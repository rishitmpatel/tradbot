"""Walk-forward validation.

Development data (everything before the locked test set) is cut into folds:

    |---- train (expanding) ----|embargo| val 1y |
    |------- train (expanding) -------|embargo| val 1y |
    ...

For each fold and each seed we train a fresh agent on the train window only,
then trade the validation window with 100,000 paper capital and the survival
rule. One (fold, seed) pair = one "run". Validation windows are consecutive,
so per seed they stitch into one continuous out-of-sample record.

Every run is saved to disk as soon as it finishes (JSON + CSV + model .zip),
so an interrupted pipeline resumes where it stopped.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import Sleeve, compute_metrics, cost_rate, portfolio_backtest, window_sleeve
from baselines import baseline_positions
from config_utils import ROOT, strategy_hash
from data import dev_view, get_split, load_prices
from features import build_assets

log = logging.getLogger(__name__)


@dataclass
class Fold:
    k: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str


def load_dev_assets(cfg: dict, tickers: list[str] | None = None):
    """Assets truncated strictly before the locked test set."""
    base_prices = load_prices(cfg)
    split = get_split(cfg, base_prices)
    prices = base_prices if tickers is None else load_prices(cfg, tickers)
    return build_assets(dev_view(prices, split), cfg["features"]), split


def make_folds(cfg: dict, assets: dict, split) -> list[Fold]:
    wf = cfg["walk_forward"]
    # Fold dates depend only on price history (not on which features an experiment
    # uses), so every experiment and robustness test shares identical folds.
    data_start = max(a.close.index[0] for a in assets.values()) + pd.DateOffset(years=int(wf.get("warmup_years", 2)))
    dev_end = split.test_start - pd.Timedelta(days=1)
    vs = data_start + pd.DateOffset(years=int(wf["min_train_years"])) + pd.Timedelta(days=int(wf.get("date_shift_days", 0)))
    min_cal_days = int(wf.get("min_val_days", 120) * 365 / 252)
    folds = []
    while vs < dev_end:
        ve = min(vs + pd.DateOffset(years=int(wf["val_years"])) - pd.Timedelta(days=1), dev_end)
        if (ve - vs).days < min_cal_days:
            break
        te = vs - pd.tseries.offsets.BDay(int(wf.get("embargo_days", 5)))
        ts = data_start if wf.get("scheme", "expanding") == "expanding" else \
            max(data_start, vs - pd.DateOffset(years=int(wf["min_train_years"])))
        folds.append(Fold(len(folds), str(ts.date()), str(te.date()), str(vs.date()), str(ve.date())))
        vs = vs + pd.DateOffset(years=int(wf["step_years"]))
    if wf.get("max_folds"):
        folds = folds[: int(wf["max_folds"])]
    for f in folds:  # hard guard: nothing may touch the test set
        assert pd.Timestamp(f.val_end) < split.test_start, "fold overlaps test set"
    return folds


# --------------------------------------------------------------------------
# One run (executed in a worker process)
# --------------------------------------------------------------------------
def run_task(task: dict) -> dict:
    """Train (or load) one agent for one fold+seed, evaluate on its validation window."""
    t0 = time.time()
    out_json = Path(task["out_json"])
    try:
        from agents import agent_positions, load_agent, train_agent, make_segment

        cfg, fold, seed = task["cfg"], Fold(**task["fold"]), int(task["seed"])
        train_cfg_rate = cost_rate(cfg)
        eval_rate = cost_rate(cfg, float(task.get("cost_mult", 1.0)))
        assets, split = load_dev_assets(cfg)
        model_path = Path(task["model_path"])
        if model_path.exists():
            model = load_agent(cfg, model_path)
            trained = False
        elif task.get("eval_only"):
            raise FileNotFoundError(f"model not found for eval-only task: {model_path}")
        else:
            w = int(cfg["env"].get("obs_window", 1))
            segs = [make_segment(a, pd.Timestamp(fold.train_start), pd.Timestamp(fold.train_end), w)
                    for a in assets.values()]
            segs = [s for s in segs if s is not None]
            model = train_agent(cfg, segs, train_cfg_rate, seed, model_path.with_suffix(""))
            trained = True

        eval_assets = assets
        if task.get("eval_tickers"):
            eval_assets, _ = load_dev_assets(cfg, task["eval_tickers"])
        vs, ve = pd.Timestamp(fold.val_start), pd.Timestamp(fold.val_end)
        sleeves = []
        for a in eval_assets.values():
            res = agent_positions(model, cfg, a, vs, ve, eval_rate)
            if res is None:
                continue
            dec, pos = res
            sl = window_sleeve(a, vs, ve, lambda d: pos)
            assert sl is not None and sl.dates.equals(dec), "decision-date mismatch"
            sleeves.append(sl)
        bt = portfolio_backtest(sleeves, eval_rate, float(cfg["capital"]["initial"]),
                                float(cfg["survival"]["max_drawdown"]))
        csv = out_json.with_suffix(".csv")
        pd.DataFrame({"equity": bt.equity, "ret": bt.returns,
                      "exposure": bt.positions.abs().mean(axis=1)}).to_csv(csv)
        rec = {"status": "ok", "fold": asdict(fold), "seed": seed, "hash": task["hash"],
               "cost_mult": task.get("cost_mult", 1.0), "eval_tickers": [s.ticker for s in sleeves],
               "metrics": bt.metrics, "trained": trained, "seconds": round(time.time() - t0, 1),
               "csv": csv.name}
    except Exception as e:
        rec = {"status": "error", "fold": task["fold"], "seed": task["seed"], "hash": task["hash"],
               "cost_mult": task.get("cost_mult", 1.0), "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc(), "seconds": round(time.time() - t0, 1)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def _is_done(path: Path, h: str, cost_mult: float) -> bool:
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return rec.get("status") == "ok" and rec.get("hash") == h and float(rec.get("cost_mult", 1)) == float(cost_mult)


def run_experiment(cfg: dict, runs_dir: Path, model_dir: Path, folds: list[Fold], n_jobs: int = 1,
                   cost_mult: float = 1.0, eval_only: bool = False, eval_tickers: list[str] | None = None,
                   label: str = "") -> list[dict]:
    """Run every (fold, seed); skip runs already finished with the same config hash."""
    h = strategy_hash(cfg)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for f in folds:
        for s in cfg["agent"]["seeds"]:
            out = runs_dir / f"fold{f.k:02d}_seed{s}.json"
            if _is_done(out, h, cost_mult):
                continue
            tasks.append({"cfg": cfg, "fold": asdict(f), "seed": s, "hash": h, "out_json": str(out),
                          "model_path": str(model_dir / h / f"fold{f.k:02d}_seed{s}.zip"),
                          "cost_mult": cost_mult, "eval_only": eval_only, "eval_tickers": eval_tickers})
    total = len(folds) * len(cfg["agent"]["seeds"])
    log.info("[%s] %d/%d runs to do (%d already done)", label or cfg["name"], len(tasks), total, total - len(tasks))
    done = 0

    def _report(rec):
        nonlocal done
        done += 1
        if rec["status"] == "ok":
            m = rec["metrics"]
            log.info("  [%s] %d/%d fold%s seed%s ret=%+.1f%% sharpe=%.2f mdd=%.1f%% %s (%.0fs)",
                     label or cfg["name"], done, len(tasks), rec["fold"]["k"], rec["seed"],
                     100 * m["total_return"], m["sharpe"], 100 * m["max_drawdown"],
                     "DEAD" if m["dead"] else "alive", rec["seconds"])
        else:
            log.error("  [%s] fold%s seed%s FAILED: %s", label or cfg["name"], rec["fold"]["k"], rec["seed"], rec["error"])

    if n_jobs <= 1 or len(tasks) <= 1:
        for t in tasks:
            _report(run_task(t))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(run_task, t): t for t in tasks}
            for fu in as_completed(futs):
                try:
                    _report(fu.result())
                except Exception as e:  # worker process crashed
                    t = futs[fu]
                    rec = {"status": "error", "fold": t["fold"], "seed": t["seed"], "hash": h,
                           "error": f"worker crashed: {e}"}
                    Path(t["out_json"]).write_text(json.dumps(rec, indent=2), encoding="utf-8")
                    _report(rec)
    return load_runs(runs_dir, folds, cfg["agent"]["seeds"])


def load_runs(runs_dir: Path, folds: list[Fold], seeds) -> list[dict]:
    recs = []
    for f in folds:
        for s in seeds:
            p = runs_dir / f"fold{f.k:02d}_seed{s}.json"
            if p.exists():
                rec = json.loads(p.read_text(encoding="utf-8"))
                rec["_dir"] = str(runs_dir)
                recs.append(rec)
            else:
                recs.append({"status": "missing", "fold": asdict(f), "seed": s})
    return recs


# --------------------------------------------------------------------------
# Baselines on the same walk-forward windows
# --------------------------------------------------------------------------
def run_baseline(cfg: dict, name: str, folds: list[Fold], runs_dir: Path, assets, cost_mult: float = 1.0) -> list[dict]:
    rate = cost_rate(cfg, cost_mult)
    params = cfg["baselines"].get(name) or {}
    runs_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in folds:
        vs, ve = pd.Timestamp(f.val_start), pd.Timestamp(f.val_end)
        sleeves = []
        for a in assets.values():
            full_pos = baseline_positions(name, a.close, params)
            sl = window_sleeve(a, vs, ve, lambda d, fp=full_pos: fp.reindex(d).fillna(0.0).values)
            if sl is not None:
                sleeves.append(sl)
        bt = portfolio_backtest(sleeves, rate, float(cfg["capital"]["initial"]), float(cfg["survival"]["max_drawdown"]))
        csv = runs_dir / f"fold{f.k:02d}_seed0.csv"
        pd.DataFrame({"equity": bt.equity, "ret": bt.returns,
                      "exposure": bt.positions.abs().mean(axis=1)}).to_csv(csv)
        recs.append({"status": "ok", "fold": asdict(f), "seed": 0, "metrics": bt.metrics,
                     "csv": csv.name, "_dir": str(runs_dir)})
    return recs


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def stitched_returns(runs: list[dict], seed) -> pd.Series | None:
    parts = []
    for r in sorted([r for r in runs if r.get("seed") == seed], key=lambda r: r["fold"]["k"]):
        if r.get("status") != "ok":
            return None  # incomplete record for this seed
        df = pd.read_csv(Path(r["_dir"]) / r["csv"], index_col=0, parse_dates=True)
        parts.append(df["ret"])
    if not parts:
        return None
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="first")]


def summarize(name: str, runs: list[dict]) -> dict:
    ok = [r for r in runs if r.get("status") == "ok"]
    seeds = sorted({r["seed"] for r in runs})
    per_seed = []
    for s in seeds:
        rets = stitched_returns(runs, s)
        if rets is None or len(rets) < 20:
            continue
        m = compute_metrics(rets)
        x = rets.values
        sd = x.std(ddof=1)
        m.update({"seed": s,
                  "skew": float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0,
                  "kurt": float(((x - x.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0})
        per_seed.append(m)
    alive = sum(1 for r in ok if not r["metrics"]["dead"])
    out = {"name": name, "n_runs": len(runs), "n_ok": len(ok),
           "n_failed": sum(1 for r in runs if r.get("status") == "error"),
           "n_missing": sum(1 for r in runs if r.get("status") == "missing"),
           "survived": alive, "survival_rate": alive / len(ok) if ok else 0.0,
           "per_seed": per_seed, "complete": len(ok) == len(runs) and len(runs) > 0}
    if per_seed:
        sh = [p["sharpe"] for p in per_seed]
        tr = [p["total_return"] for p in per_seed]
        out.update({
            "mean_sharpe": float(np.mean(sh)), "min_sharpe": float(np.min(sh)),
            "sharpe_std_seeds": float(np.std(sh)),
            "mean_return": float(np.mean(tr)), "min_return": float(np.min(tr)),
            "mean_cagr": float(np.mean([p["cagr"] for p in per_seed])),
            "mean_sortino": float(np.mean([p["sortino"] for p in per_seed])),
            "mean_stitched_mdd": float(np.mean([p["max_drawdown"] for p in per_seed])),
            "n_days": int(per_seed[0]["n_days"]), "years": per_seed[0]["years"],
            "skew": float(np.mean([p["skew"] for p in per_seed])),
            "kurt": float(np.mean([p["kurt"] for p in per_seed])),
        })
    if ok:
        out["mean_run_mdd"] = float(np.mean([r["metrics"]["max_drawdown"] for r in ok]))
        out["mean_turnover"] = float(np.mean([r["metrics"].get("annual_turnover", 0) for r in ok]))
        out["mean_exposure"] = float(np.mean([r["metrics"].get("avg_exposure", 0) for r in ok]))
    errs = [r.get("error") for r in runs if r.get("status") == "error"]
    if errs:
        out["errors"] = errs[:5]
    return out
