"""Run the whole research pipeline unattended. Safe to interrupt and re-run:
every finished walk-forward run is saved and skipped next time.

    python run_all.py                  # real Yahoo data, all configs/*.yaml
    python run_all.py --smoke          # 5-10 minute check that everything works
    python run_all.py --synthetic      # fake data (code testing only)
    python run_all.py --only ppo_base dqn_base
    python run_all.py --n-jobs 6

Stages
  1. load / download data, lock test split, data-quality report
  2. self-checks (lookahead, engine timing, env==engine, kill switch) -> abort on failure
  3. baselines on every walk-forward window
  4. every experiment in configs/ (max `experiment_cap`), every fold x seed
  5. multiple-testing penalty, pass/fail per experiment
  6. robustness tests on the top candidates
  7. RESULTS.md + plots + selection.json (input for final_test.py)
The locked test set is never touched here.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from backtest import cost_rate
from config_utils import (CONFIGS_DIR, ROOT, deep_merge, list_experiment_files, load_base, load_experiment,
                          resolve_n_jobs, strategy_hash)
from data import get_split, load_prices, quality_report
from evaluation import base_checks, final_verdict, penalize
from reporting import plot_experiment, plot_overview, save_json, write_results_md
from robustness import run_robustness
from selfcheck import run_all_checks
from walkforward import load_dev_assets, make_folds, run_baseline, run_experiment, stitched_returns, summarize

log = logging.getLogger("run_all")

SMOKE = {
    "agent": {"total_timesteps": 3000, "seeds": [0, 1],
              "ppo": {"n_steps": 512, "batch_size": 64, "n_epochs": 3},
              "dqn": {"learning_starts": 500, "buffer_size": 5000}},
    "walk_forward": {"max_folds": 2},
    "robustness": {"other_seeds": [101], "shift_seeds": [0], "top_k": 2, "run_full_only_if_base_pass": False},
}


def prepare_cfg(cfg: dict, synthetic: bool, smoke: bool) -> dict:
    if synthetic:
        cfg = deep_merge(cfg, {"data": {"source": "synthetic"}})
    if smoke:
        cfg = deep_merge(cfg, SMOKE)
    return cfg


def mode_name(synthetic: bool, smoke: bool) -> str:
    return ("smoke_" if smoke else "") + ("synthetic" if synthetic else "real")


def mode_dirs(cfg: dict, synthetic: bool, smoke: bool) -> tuple[Path, Path]:
    m = mode_name(synthetic, smoke)
    return ROOT / cfg["runtime"]["results_dir"] / m, ROOT / cfg["runtime"]["models_dir"] / m


def results_md_path(synthetic: bool, smoke: bool) -> Path:
    if not synthetic and not smoke:
        return ROOT / "RESULTS.md"
    return ROOT / f"RESULTS_{mode_name(synthetic, smoke).upper()}.md"


def setup_logging(cfg: dict, mode: str) -> Path:
    logs = ROOT / cfg["runtime"]["logs_dir"]
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"run_all_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    for noisy in ("yfinance", "matplotlib", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def update_ledger(path: Path, cfgs: list[dict], rnd: int) -> int:
    """Every distinct (name, config hash) ever validated counts as one trial, across all rounds."""
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"trials": {}}
    for c in cfgs:
        key = f"{c['name']}@{strategy_hash(c)}"
        ledger["trials"].setdefault(key, {"round": rnd, "first_seen": time.strftime("%Y-%m-%d %H:%M")})
    save_json(path, ledger)
    return len(ledger["trials"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated fake prices (code testing only)")
    ap.add_argument("--smoke", action="store_true", help="tiny budget quick check")
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--configs-dir", default=str(CONFIGS_DIR))
    args = ap.parse_args()

    base_raw = load_base()
    base = prepare_cfg(base_raw, args.synthetic, args.smoke)
    mode = mode_name(args.synthetic, args.smoke)
    log_path = setup_logging(base, mode)
    res_root, model_root = mode_dirs(base, args.synthetic, args.smoke)
    res_root.mkdir(parents=True, exist_ok=True)
    n_jobs = args.n_jobs or resolve_n_jobs(base["runtime"].get("n_jobs"))
    pc = base["pass_criteria"]
    t_start = time.time()
    log.info("=== quant-rl-bot run_all | mode=%s | n_jobs=%d | log=%s", mode, n_jobs, log_path)
    if args.synthetic:
        log.warning("SYNTHETIC DATA: results are NOT real and say nothing about markets.")

    # 1. data ---------------------------------------------------------------
    prices = load_prices(base)
    split = get_split(base, prices)
    quality = quality_report(prices)
    for q in quality:
        log.info("data %s: %d rows %s -> %s jumps=%s", q["ticker"], q["rows"], q["start"], q["end"], q["big_jumps"])
    log.info("locked test set %s -> %s (not used by this script)", split.test_start.date(), split.data_end.date())
    if not args.synthetic:
        try:  # cache robustness assets up front (avoid parallel downloads)
            load_prices(base, base["robustness"]["other_assets"])
        except Exception as e:
            log.warning("could not download robustness assets: %s", e)

    # 2. self-checks ----------------------------------------------------------
    try:
        dev_prices = {t: df[df.index < split.test_start] for t, df in prices.items()}
        for c in run_all_checks(dev_prices, base):
            log.info("self-check OK: %s", c)
    except AssertionError as e:
        log.error("SELF-CHECK FAILED: %s -- aborting, results would be untrustworthy", e)
        return 2

    # 3. experiments to run ---------------------------------------------------
    files = list_experiment_files(args.configs_dir)
    if args.only:
        files = [f for f in files if f.stem in set(args.only)]
    cap = int(base_raw.get("experiment_cap", 30))
    skipped = [f.stem for f in files[cap:]]
    files = files[:cap]
    if skipped:
        log.warning("experiment cap %d reached; skipping %s", cap, skipped)
    cfgs, load_errors = [], []
    for f in files:
        try:
            cfgs.append(prepare_cfg(load_experiment(f, base_raw), args.synthetic, args.smoke))
        except Exception as e:
            load_errors.append({"name": f.stem, "status": f"config error: {e}", "verdict": "FAIL",
                                "reasons": [f"config error: {e}"]})
            log.error("config %s invalid: %s", f.name, e)
    n_trials = update_ledger(res_root / "trial_ledger.json", cfgs, int(base_raw.get("round", 1)))
    log.info("%d experiments this run; %d variants tried in total (all rounds)", len(cfgs), n_trials)

    assets, split = load_dev_assets(base)
    base_folds = make_folds(base, assets, split)
    log.info("%d walk-forward folds: first val %s, last val %s", len(base_folds),
             base_folds[0].val_start, base_folds[-1].val_end)

    # 4. baselines --------------------------------------------------------------
    baseline_summaries, baseline_runs = [], {}
    for name in base["baselines"]:
        runs = run_baseline(base, name, base_folds, res_root / "baselines" / name, assets)
        baseline_runs[name] = runs
        s = summarize(f"baseline:{name}", runs)
        s["algo"] = "rule"
        baseline_summaries.append(s)
        log.info("baseline %-15s return=%+.1f%% sharpe=%.2f survival=%d/%d", name, 100 * s.get("mean_return", 0),
                 s.get("mean_sharpe", 0), s["survived"], s["n_ok"])

    # 5. RL experiments -----------------------------------------------------------
    summaries, all_runs = [], {}
    for i, cfg in enumerate(cfgs, 1):
        name = cfg["name"]
        log.info("--- experiment %d/%d: %s (%s) %s", i, len(cfgs), name, cfg["agent"]["algo"], cfg.get("description", ""))
        exp_dir = res_root / "experiments" / name
        try:
            folds = make_folds(cfg, assets, split)
            runs = run_experiment(cfg, exp_dir / "runs", model_root / name, folds, n_jobs)
            all_runs[name] = runs
            s = summarize(name, runs)
            s["status"] = "ok" if s["complete"] else "incomplete"
        except KeyboardInterrupt:
            log.warning("interrupted - re-run run_all.py to resume")
            raise
        except Exception as e:
            log.error("experiment %s crashed: %s\n%s", name, e, traceback.format_exc())
            s = {"name": name, "status": f"error: {type(e).__name__}: {e}", "n_runs": 0, "n_ok": 0, "survived": 0}
        s["algo"] = cfg["agent"]["algo"].upper()
        s["hash"] = strategy_hash(cfg)
        s["description"] = cfg.get("description", "")
        summaries.append(s)
        save_json(exp_dir / "summary.json", s)

    # 6. penalty + base checks ------------------------------------------------------
    cross = [s["mean_sharpe"] for s in summaries if "mean_sharpe" in s]
    for s in summaries + baseline_summaries:
        penalize(s, n_trials, cross)
        s["base_pass"], s["reasons"] = base_checks(s, pc)

    # 7. robustness ---------------------------------------------------------------------
    rcfg = base["robustness"]
    ranked = sorted([s for s in summaries if "penalized_sharpe" in s], key=lambda s: -s["penalized_sharpe"])
    robustness = {}
    if not args.skip_robustness:
        by_name = {c["name"]: c for c in cfgs}
        for s in ranked[: int(rcfg["top_k"])]:
            cfg = by_name[s["name"]]
            tests = ["costs_2x", "other_assets"]
            if s["base_pass"] or not rcfg.get("run_full_only_if_base_pass", True):
                tests += ["other_seeds", "shifted_dates"]
            log.info("--- robustness for %s: %s", s["name"], tests)
            try:
                robustness[s["name"]] = run_robustness(cfg, res_root / "experiments" / s["name"],
                                                       model_root / s["name"], n_jobs, tests)
            except Exception as e:
                log.error("robustness for %s crashed: %s", s["name"], e)
                robustness[s["name"]] = {"error": {"status": f"error: {e}", "summary": None}}
            for t in ("other_seeds", "shifted_dates"):
                robustness[s["name"]].setdefault(t, {"status": "not run (failed base criteria)", "summary": None})

    # 8. final verdicts -------------------------------------------------------------------
    suspicious, passing = [], []
    for s in summaries:
        if s.get("mean_sharpe", 0) > pc["suspicious_sharpe"]:
            suspicious.append(f"{s['name']} (Sharpe {s['mean_sharpe']:.2f})")
        v = final_verdict(s, robustness.get(s["name"], {}), rcfg["required_tests"], pc)
        s["verdict"] = "PASS" if v["pass"] else "FAIL"
        s["reasons"] = v["reasons"]
        if v["pass"]:
            passing.append(s)
    for s in baseline_summaries:
        if s.get("mean_sharpe", 0) > pc["suspicious_sharpe"]:
            suspicious.append(f"{s['name']} (Sharpe {s['mean_sharpe']:.2f})")
        s["verdict"] = "benchmark" + (" (meets base criteria)" if s["base_pass"] else "")
    passing.sort(key=lambda s: -s["penalized_sharpe"])
    selected = passing[0]["name"] if passing else None
    save_json(res_root / "selection.json", {
        "selected": selected, "hash": passing[0]["hash"] if passing else None,
        "passing": [s["name"] for s in passing], "n_trials": n_trials,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
        "note": "final_test.py evaluates ONLY the selected strategy, ONCE."})

    # 9. report -------------------------------------------------------------------------------
    plots_dir = res_root / "plots"
    md_path = results_md_path(args.synthetic, args.smoke)
    rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()  # noqa: E731
    plots, overview = {}, {}
    for s in summaries:
        runs = all_runs.get(s["name"])
        if not runs:
            continue
        p = plots_dir / f"{s['name']}.png"
        try:
            plot_experiment(s["name"], runs, baseline_runs.get("buy_and_hold"), p, float(base["capital"]["initial"]))
            plots[s["name"]] = rel(p)
        except Exception as e:
            log.warning("plot %s failed: %s", s["name"], e)
        seeds = sorted({r["seed"] for r in runs})
        parts = [stitched_returns(runs, sd) for sd in seeds]
        parts = [x for x in parts if x is not None]
        if parts:
            overview[s["name"]] = pd.concat(parts, axis=1).fillna(0).mean(axis=1)
    for name, runs in baseline_runs.items():
        r = stitched_returns(runs, 0)
        if r is not None:
            overview[f"baseline:{name}"] = r
    ov_path = plots_dir / "overview.png"
    plot_overview(overview, ov_path, float(base["capital"]["initial"]),
                  "Stitched OOS equity (RL: average over seeds)" + (" - SYNTHETIC DATA" if args.synthetic else ""))

    lock = res_root / "final_test" / "LOCK.json"
    ft_note = (f"final_test.py was already run: {json.loads(lock.read_text(encoding='utf-8')).get('run_at')}."
               if lock.exists() else "final_test.py has not been run yet.")
    write_results_md(md_path, {
        "source": base["data"]["source"], "smoke": args.smoke, "generated": time.strftime("%Y-%m-%d %H:%M"),
        "round": base_raw.get("round", 1), "n_trials": n_trials, "dev_end": str((split.test_start - pd.Timedelta(days=1)).date()),
        "test_start": str(split.test_start.date()), "data_end": str(split.data_end.date()),
        "cost_bps": cost_rate(base) * 1e4, "max_dd": float(base["survival"]["max_drawdown"]),
        "initial": float(base["capital"]["initial"]), "passing": [s["name"] for s in passing],
        "selected": selected, "suspicious": suspicious, "pc": pc,
        "experiments": summaries + load_errors, "baselines": baseline_summaries, "robustness": robustness,
        "skipped": skipped, "cap": cap, "folds": base_folds, "quality": quality,
        "overview_plot": rel(ov_path), "plots": plots, "final_test_note": ft_note})
    save_json(res_root / "all_summaries.json", {"experiments": summaries, "baselines": baseline_summaries,
                                                "robustness": robustness, "load_errors": load_errors})
    mins = (time.time() - t_start) / 60
    log.info("=== done in %.1f min. Report: %s", mins, md_path)
    log.info("PASSING: %s | SELECTED for final test: %s", [s["name"] for s in passing] or "none", selected)
    if suspicious:
        log.warning("SUSPICIOUS results (assume leakage): %s", suspicious)
    return 0


if __name__ == "__main__":
    sys.exit(main())
