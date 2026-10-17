"""Round 4 research stage: exploration -> finalists -> confirmation -> report.

    python research.py               (also called automatically by run_all.py)
    python research.py --synthetic   (code testing only)

1. EXPLORATION (development data only): every hypothesis in research/plan_round4.yaml is
   run on the broad ETF universe, 2007 -> the day before the locked test set, in one-year
   windows (fresh 100,000, 15% kill switch) and stitched. Every hypothesis is a TRIAL in the
   ledger. Hypotheses not in the first-run registration are marked EXPLORATORY.
2. FINALISTS: at most 3, chosen by the pre-registered rule in the plan, from exploration only.
3. CONFIRMATION: finalists + benchmarks on a universe and period exploration never touched
   (mutual funds, 1991-2006). Runs only once the owner sets
   research.confirmation_approved: true in base_config.yaml. The finalists are then locked.
4. Report: RESEARCH.md (also appended to RESULTS.md by run_all.py).
The locked test set (data/raw/split_lock_*.json) is never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from allocation import benchmark_weights, build_weights, rebalance_days, rebalance_rule
from backtest import compute_metrics, weights_backtest
from config_utils import ROOT, deep_merge, load_base
from data import get_split, load_prices, quality_report
from evaluation import (base_checks, bootstrap_sharpe_diff, cagr_at_vol, deflated_sharpe_prob, penalize,
                        robust_check)
from reporting import save_json

log = logging.getLogger("research")
PLAN_PATH = ROOT / "research" / "plan_round4.yaml"


# --------------------------------------------------------------------------
# Plan, registration and trial ledger
# --------------------------------------------------------------------------
def load_plan(path: Path = PLAN_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec_hash(spec: dict) -> str:
    """Hash of what a hypothesis DOES (reason text excluded)."""
    core = {k: v for k, v in spec.items() if k != "reason"}
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:10]


def register(plan: dict, res_dir: Path) -> dict:
    """First real run freezes the list of registered hypotheses. Anything added or changed later
    is exploratory. Returns {name: 'registered' | 'exploratory'}."""
    reg_path = res_dir / "registration.json"
    hashes = {n: spec_hash(s) for n, s in plan["hypotheses"].items()}
    if not reg_path.exists():
        save_json(reg_path, {"registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "plan_sha": hashlib.sha1(PLAN_PATH.read_bytes()).hexdigest()[:12],
                             "hypotheses": hashes})
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["hypotheses"]
    return {n: ("registered" if reg.get(n) == h else "exploratory") for n, h in hashes.items()}


def update_research_ledger(ledger_path: Path, plan: dict, status: dict) -> int:
    """Every round-4 hypothesis definition ever run is one trial, forever (keys 'r4:name@hash')."""
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"trials": {}}
    for n, spec in plan["hypotheses"].items():
        ledger["trials"].setdefault(f"r4:{n}@{spec_hash(spec)}", {
            "round": plan.get("round", 4), "kind": status[n], "first_seen": time.strftime("%Y-%m-%d %H:%M")})
    save_json(ledger_path, ledger)
    return sum(1 for k in ledger["trials"] if k.startswith("r4:"))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_universe(base: dict, uni: dict, rf_ticker: str, until: pd.Timestamp, synthetic: bool):
    """Close panel (union calendar, NaN before each fund's first day), daily risk-free rate,
    data-quality rows and the tickers that could not be loaded."""
    tickers = list(uni["tickers"])
    if synthetic:
        from synthetic_data import generate_prices
        prices = generate_prices(tickers, uni["start"], None, int(base["data"].get("synthetic_seed", 7)) + 11)
        rf = pd.Series(0.02 / 252, index=pd.bdate_range(uni["start"], "2026-10-16"))
        missing = []
    else:
        cfg = deep_merge(base, {"data": {"start": uni["start"], "cache_dir": "data/raw_research", "source": "yahoo"}})
        prices, missing = {}, []
        for t in tickers:
            try:
                prices.update(load_prices(cfg, [t]))
            except Exception as e:
                log.warning("could not load %s: %s - excluded", t, e)
                missing.append(t)
        try:
            irx = load_prices(cfg, [rf_ticker])[rf_ticker]["close"]
            rf = (irx / 100.0 / 252.0).clip(lower=0.0)
        except Exception as e:
            log.warning("could not load %s (%s): financing/T-bill filter use a flat 3%%", rf_ticker, e)
            rf = pd.Series(0.03 / 252, index=pd.bdate_range(uni["start"], until))
    prices = {t: df[df.index < until] for t, df in prices.items()}
    prices = {t: df for t, df in prices.items() if len(df) > 300}
    panel = pd.DataFrame({t: df["close"] for t, df in prices.items()}).sort_index()
    panel = panel.ffill().where(panel.bfill().notna())   # fill holidays, never before inception
    rf = rf.reindex(panel.index).ffill().fillna(0.0)
    assert panel.index.max() < until, "data beyond the allowed cut-off"
    return panel, rf, quality_report(prices), missing


def yearly_windows(index: pd.DatetimeIndex, start: str, end: pd.Timestamp, shift_days: int = 0):
    ws = pd.Timestamp(start) + pd.Timedelta(days=shift_days)
    out = []
    while ws < end:
        we = min(ws + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        if (we - ws).days >= 120:
            out.append((ws, we))
        ws = ws + pd.DateOffset(years=1)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def run_windows(W: pd.DataFrame, rebal: np.ndarray, panel: pd.DataFrame, rf: pd.Series, windows, common: dict,
                cost_mult: float = 1.0, credit_cash: bool | None = None, min_live: int = 1,
                kill: bool = True) -> list[dict]:
    rate = float(common["costs_bps"]) / 1e4 * cost_mult
    credit = common.get("credit_cash_interest", False) if credit_cash is None else credit_cash
    fwd_all = (panel.shift(-1) / panel - 1.0)
    reb = pd.Series(rebal, index=panel.index)
    out = []
    for ws, we in windows:
        days = panel.index[(panel.index >= ws) & (panel.index <= we)]
        if len(days) < 60 or int(panel.loc[days[0]].notna().sum()) < min_live:
            continue
        dec, real = days[:-1], days[1:]
        fwd = fwd_all.loc[dec]
        tgt = W.loc[dec].values
        if np.any((tgt != 0) & fwd.isna().values):
            raise AssertionError("weight on an asset without a price (lookahead/data bug)")
        tradable = np.repeat(reb.loc[dec].values[:, None], panel.shape[1], axis=1)
        tradable[0] = True                                   # every window starts invested
        bt = weights_backtest(dec, real, tgt, fwd.fillna(0.0).values, tradable, rate, 1e5,
                              float(common["max_drawdown_kill"]) if kill else None, list(panel.columns), rf=rf.loc[dec].values,
                              borrow_spread=float(common["borrow_spread"]), credit_cash=credit)
        out.append({"start": str(ws.date()), "end": str(we.date()), "metrics": bt.metrics,
                    "rets": bt.returns.iloc[1:], "gross": float(np.abs(tgt).sum(axis=1).mean())})
    return out


def summarize_windows(name: str, wins: list[dict], rf: pd.Series) -> dict:
    if not wins:
        return {"name": name, "n_runs": 0, "n_ok": 0, "survived": 0, "complete": False}
    rets = pd.concat([w["rets"] for w in wins])
    rets = rets[~rets.index.duplicated()]
    m = compute_metrics(rets)
    x = rets.values
    sd = x.std(ddof=1)
    alive = sum(1 for w in wins if not w["metrics"]["dead"])
    cagr10, k10 = cagr_at_vol(x, rf.reindex(rets.index).fillna(0.0).values, 0.10)
    return {"name": name, "n_runs": len(wins), "n_ok": len(wins), "complete": True,
            "survived": alive, "survival_rate": alive / len(wins),
            "mean_sharpe": m["sharpe"], "mean_return": m["total_return"], "cagr": m["cagr"],
            "ann_vol": m["ann_vol"], "mean_stitched_mdd": m["max_drawdown"], "sortino": m["sortino"],
            "worst_window": min(w["metrics"]["total_return"] for w in wins),
            "mean_turnover": float(np.mean([w["metrics"]["annual_turnover"] for w in wins])),
            "avg_gross": float(np.mean([w["gross"] for w in wins])),
            "cagr_at_10vol": cagr10, "scale_to_10vol": k10,
            "n_days": int(m["n_days"]), "years": m["years"],
            "skew": float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0,
            "kurt": float(((x - x.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0,
            "_rets": rets}


def strategy_set(plan: dict, panel: pd.DataFrame, rf: pd.Series, names=None, overrides=None):
    """{name: (weights, rebalance mask)} for hypotheses (+ optional parameter overrides)."""
    specs = plan["hypotheses"]
    cache: dict = {}
    out = {}
    for n in names or specs:
        W = build_weights(n, specs[n], specs, panel, rf, (overrides or {}).get(n), cache)
        out[n] = (W, rebalance_days(panel.index, rebalance_rule(specs[n], specs)))
    return out


def benchmark_set(uni: dict, panel: pd.DataFrame) -> dict:
    eq, bd = uni["benchmarks"]["equity"], uni["benchmarks"]["bonds"]
    hold = np.zeros(len(panel), dtype=bool)                  # buy & hold: trade only at window start
    return {"bench:buy_and_hold": (benchmark_weights("buy_and_hold", panel, eq, bd), hold),
            "bench:60_40": (benchmark_weights("sixty_forty", panel, eq, bd), rebalance_days(panel.index))}


# --------------------------------------------------------------------------
# Main stage
# --------------------------------------------------------------------------
def run_research(base: dict, res_root: Path, synthetic: bool, legacy_trials: int, approved: bool) -> dict:
    plan = load_plan()
    common = plan["common"]
    rdir = res_root / "research"
    rdir.mkdir(parents=True, exist_ok=True)
    status = register(plan, rdir)
    n_r4 = update_research_ledger(res_root / "trial_ledger.json", plan, status)
    n_total = legacy_trials + n_r4
    log.info("round 4: %d hypotheses (%d exploratory); trials so far: %d earlier + %d round-4 = %d",
             len(status), sum(v == "exploratory" for v in status.values()), legacy_trials, n_r4, n_total)

    # ---------------- exploration ----------------
    split = get_split(base, load_prices(base))
    E = plan["universes"]["exploration"]
    e_end = pd.Timestamp(E["eval_end"]) if E.get("eval_end") else split.test_start - pd.Timedelta(days=1)
    assert e_end < split.test_start
    panel, rf, quality_e, missing_e = load_universe(base, E, plan["universes"]["risk_free"], split.test_start, synthetic)
    wins = yearly_windows(panel.index, E["eval_start"], e_end)
    strats = strategy_set(plan, panel, rf)
    bench = benchmark_set(E, panel)
    summ, win_tables = {}, {}
    for n, (W, reb) in {**strats, **bench}.items():
        w = run_windows(W, reb, panel, rf, wins, common)
        summ[n] = summarize_windows(n, w, rf)
        win_tables[n] = [(x["start"], x["metrics"]["total_return"], x["metrics"]["dead"]) for x in w]
        log.info("exploration %-28s Sharpe %.2f CAGR %+.1f%% maxDD %.1f%% survived %d/%d", n,
                 summ[n].get("mean_sharpe", float("nan")), 100 * summ[n].get("cagr", float("nan")),
                 100 * summ[n].get("mean_stitched_mdd", float("nan")), summ[n]["survived"], summ[n]["n_ok"])

    # penalty, significance vs 60/40, robustness
    cross = [summ[n]["mean_sharpe"] for n in strats if "mean_sharpe" in summ[n]]
    b60, bbh = summ["bench:60_40"], summ["bench:buy_and_hold"]
    pc = base["pass_criteria"]
    for n in strats:
        s = summ[n]
        penalize(s, n_total, cross)
        s["status"] = status[n]
        a, b = s["_rets"].align(b60["_rets"], join="inner")
        s["p_vs_60_40"] = bootstrap_sharpe_diff(a.values, b.values)
        s["sharpe_minus_60_40"] = s["mean_sharpe"] - b60["mean_sharpe"]
        s["sharpe_minus_bh"] = s["mean_sharpe"] - bbh["mean_sharpe"]
        rob = {}
        rob["costs_2x"] = summarize_windows(n, run_windows(*strats[n], panel, rf, wins, common, cost_mult=2.0), rf)
        shifted = yearly_windows(panel.index, E["eval_start"], e_end, shift_days=182)
        rob["shifted_windows"] = summarize_windows(n, run_windows(*strats[n], panel, rf, shifted, common), rf)
        neigh = plan["hypotheses"][n].get("neighborhood") or []
        nres = []
        for ov in neigh:
            ns = strategy_set(plan, panel, rf, [n], {n: ov})[n]
            nres.append((ov, summarize_windows(n, run_windows(*ns, panel, rf, wins, common), rf)))
        s["robust"] = {"costs_2x": robust_check(rob["costs_2x"], pc),
                       "shifted_windows": robust_check(rob["shifted_windows"], pc),
                       "param_neighborhood": (all(robust_check(x, pc)[0] for _, x in nres),
                                              [f"{ov}: {', '.join(robust_check(x, pc)[1])}"
                                               for ov, x in nres if not robust_check(x, pc)[0]])}
        s["costs_2x_sharpe"], s["costs_2x_return"] = rob["costs_2x"]["mean_sharpe"], rob["costs_2x"]["mean_return"]
        s["shifted_sharpe"] = rob["shifted_windows"]["mean_sharpe"]
        s["neighbor_min_sharpe"] = min((x["mean_sharpe"] for _, x in nres), default=None)
        s["with_cash_interest_sharpe"] = summarize_windows(
            n, run_windows(*strats[n], panel, rf, wins, common, credit_cash=True), rf)["mean_sharpe"]

    # Round-5 measurement checks (informational): (a) without the kill switch, (b) P5 recheck
    for n in list(strats) + list(bench):
        W, reb = {**strats, **bench}[n]
        nk = summarize_windows(n, run_windows(W, reb, panel, rf, wins, common, kill=False), rf)
        summ[n]["nokill_sharpe"], summ[n]["nokill_cagr"] = nk.get("mean_sharpe"), nk.get("cagr")
        ex_n = nk["_rets"] - rf.reindex(nk["_rets"].index).fillna(0.0)
        summ[n]["nokill_excess_sharpe"] = float(ex_n.mean() / ex_n.std(ddof=1) * np.sqrt(252))
        summ[n]["_nokill_ex"] = ex_n
    p5 = None
    if "H3_trend_following" in strats:
        cash = summarize_windows("p5", run_windows(*strats["H3_trend_following"], panel, rf, wins, common,
                                                   credit_cash=True), rf)
        ex_h3 = cash["_rets"] - rf.reindex(cash["_rets"].index).fillna(0.0)
        ex_60 = b60["_rets"] - rf.reindex(b60["_rets"].index).fillna(0.0)
        a, c = ex_h3.align(ex_60, join="inner")
        p5 = {"raw_sharpe_cash_0": summ["H3_trend_following"]["mean_sharpe"], "raw_sharpe_cash_tbill": cash["mean_sharpe"],
              "excess_sharpe_cash_tbill": float(a.mean() / a.std(ddof=1) * np.sqrt(252)),
              "excess_sharpe_60_40": float(c.mean() / c.std(ddof=1) * np.sqrt(252)),
              "p_vs_60_40_excess": bootstrap_sharpe_diff(a.values, c.values)}
        log.info("P5 recheck (H3, cash at T-bill, excess Sharpe): %s", {k: round(v, 3) for k, v in p5.items()})

    # correlation of member returns (does diversifying across RULES help?)
    corr_names = [n for n in ("H2_inverse_vol", "H3_trend_following", "H4_dual_momentum", "bench:60_40") if n in summ]
    corr = pd.concat({n: summ[n]["_rets"] for n in corr_names}, axis=1).corr().round(2) if corr_names else None

    # ---------------- finalists (pre-registered rule, exploration only) ----------------
    fs = plan["finalist_selection"]
    eligible = [n for n in strats if summ[n]["survival_rate"] >= 0.8 and summ[n]["costs_2x_return"] > 0]
    finalists = sorted(eligible, key=lambda n: -summ[n]["mean_sharpe"])[: int(fs["max_finalists"])]
    fin_path = rdir / "finalists.json"
    conf_lock = rdir / "confirmation_LOCK.json"
    if conf_lock.exists():
        locked = json.loads(conf_lock.read_text(encoding="utf-8"))["finalists"]
        if locked != finalists:
            log.error("FINALISTS CHANGED after confirmation was run (%s -> %s). Using the locked ones.", locked, finalists)
        finalists = locked
    save_json(fin_path, {"finalists": finalists, "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "rule": fs, "plan_sha": hashlib.sha1(PLAN_PATH.read_bytes()).hexdigest()[:12]})
    log.info("finalists (pre-registered rule): %s", finalists)

    # ---------------- confirmation (owner-approved, once) ----------------
    conf = {"status": "not approved yet (set research.confirmation_approved: true in base_config.yaml)"}
    if approved and finalists:
        C = plan["universes"]["confirmation"]
        c_until = pd.Timestamp(C["eval_end"]) + pd.Timedelta(days=1)
        assert c_until <= pd.Timestamp(E["eval_start"]), "confirmation period overlaps exploration"
        cpanel, crf, quality_c, missing_c = load_universe(base, C, plan["universes"]["risk_free"], c_until, synthetic)
        cwins = yearly_windows(cpanel.index, C["eval_start"], pd.Timestamp(C["eval_end"]))
        cstr = {**strategy_set(plan, cpanel, crf, finalists), **benchmark_set(C, cpanel)}
        cs = {n: summarize_windows(n, run_windows(W, reb, cpanel, crf, cwins, common,
                                                  min_live=int(C.get("min_live_assets", 1))), crf)
              for n, (W, reb) in cstr.items()}
        crit = plan["confirmation_criteria"]
        verdicts = {}
        for n in finalists:
            s, why = cs[n], []
            if s.get("mean_return", -1) <= crit["return_after_costs_gt"]:
                why.append(f"return {s.get('mean_return', float('nan')):+.1%}")
            if crit.get("sharpe_gt_benchmark_60_40") and s.get("mean_sharpe", -9) <= cs["bench:60_40"].get("mean_sharpe", 9):
                why.append(f"Sharpe {s.get('mean_sharpe', float('nan')):.2f} not above 60/40 "
                           f"{cs['bench:60_40'].get('mean_sharpe', float('nan')):.2f}")
            if s.get("survival_rate", 0) < crit["survival_rate_ge"]:
                why.append(f"survival {s['survived']}/{s['n_ok']}")
            if s.get("mean_stitched_mdd", 1) > crit["max_drawdown_le"]:
                why.append(f"max DD {s['mean_stitched_mdd']:.1%}")
            verdicts[n] = {"pass": not why, "reasons": why}
        conf = {"status": "ok", "summaries": cs, "verdicts": verdicts, "quality": quality_c, "missing": missing_c,
                "windows": [(str(a.date()), str(b.date())) for a, b in cwins]}
        if not conf_lock.exists():
            save_json(conf_lock, {"finalists": finalists, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        for n, v in verdicts.items():
            log.info("confirmation %s: %s %s", n, "PASS" if v["pass"] else "FAIL", "; ".join(v["reasons"]))
    elif approved:
        conf = {"status": "no finalists"}

    # ---------------- verdicts ----------------
    for n in strats:
        s = summ[n]
        ok, why = base_checks(s, pc)
        if s["mean_sharpe"] <= bbh["mean_sharpe"]:
            ok, why = False, why + [f"Sharpe {s['mean_sharpe']:.2f} not above buy-and-hold {bbh['mean_sharpe']:.2f}"]
        if s["mean_sharpe"] <= b60["mean_sharpe"]:
            ok, why = False, why + [f"Sharpe {s['mean_sharpe']:.2f} not above 60/40 {b60['mean_sharpe']:.2f}"]
        for t in ("costs_2x", "param_neighborhood"):
            p, r = s["robust"][t]
            if not p:
                ok, why = False, why + [f"robustness {t} failed: {', '.join(r) if r else ''}"]
        if n not in finalists:
            ok, why = False, why + ["not a finalist"]
        elif conf.get("status") != "ok":
            ok, why = False, why + ["confirmation not run"]
        elif not conf["verdicts"][n]["pass"]:
            ok, why = False, why + ["confirmation failed: " + "; ".join(conf["verdicts"][n]["reasons"])]
        s["verdict"], s["reasons"] = ("PASS" if ok else "FAIL"), why
        # PROPOSAL P4 (not active): profit at equal risk + significance + confirmation
        dsr = deflated_sharpe_prob(s["mean_sharpe"], s["sharpe_haircut"], s["n_days"], s["years"], s["skew"], s["kurt"])
        s["p4"] = bool(s["cagr_at_10vol"] > b60["cagr_at_10vol"] and s["p_vs_60_40"] < 0.05 and dsr > 0.95
                       and s["survival_rate"] >= 0.8 and s["robust"]["costs_2x"][0]
                       and n in finalists and conf.get("status") == "ok" and conf["verdicts"][n]["pass"])

    result = {"plan": plan, "status": status, "n_total": n_total, "n_r4": n_r4, "legacy_trials": legacy_trials,
              "summ": summ, "wins": win_tables, "finalists": finalists, "confirmation": conf, "corr": corr,
              "quality": quality_e, "missing": missing_e, "windows": [(str(a.date()), str(b.date())) for a, b in wins],
              "synthetic": synthetic, "approved": approved, "p5": p5,
              "plan_sha": hashlib.sha1(PLAN_PATH.read_bytes()).hexdigest()[:12]}
    save_json(rdir / "summary.json", {k: _clean(v) for k, v in result.items() if k not in ("corr",)})
    return result


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def stitched_equity(res: dict, names) -> dict:
    return {n: (1 + res["summ"][n]["_rets"]).cumprod() * 1e5 for n in names if "_rets" in res["summ"][n]}


def write_report(res: dict, path: Path, plot_path: Path) -> str:
    """Markdown for RESEARCH.md (and the round-4 section of RESULTS.md)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reporting import _num, _pct, _plain, _signed

    summ, plan = res["summ"], res["plan"]
    names = list(plan["hypotheses"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for n, eq in stitched_equity(res, names + ["bench:60_40", "bench:buy_and_hold"]).items():
        ax.plot(eq, "--" if n.startswith("bench") else "-", lw=1.1, label=n)
    ax.set_yscale("log")
    ax.set_title("Round 4 exploration: stitched one-year windows (log scale)" + (" - SYNTHETIC" if res["synthetic"] else ""))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=90)
    plt.close(fig)

    L = ["## Round 4 research: diversified allocation rules (pre-registered plan)\n"]
    if res["synthetic"]:
        L.append("> **SYNTHETIC DATA - NOT REAL RESULTS.**\n")
    st = res["status"]
    L.append(f"- Plan `research/plan_round4.yaml` sha `{res['plan_sha']}`; hypotheses registered: "
             f"{sum(v == 'registered' for v in st.values())}, exploratory (added/changed after registration): "
             f"{sum(v == 'exploratory' for v in st.values())}  \n"
             f"- **Trials counted for the Sharpe penalty: {res['n_total']}** ({res['legacy_trials']} from rounds 1-3 + "
             f"{res['n_r4']} round-4)  \n"
             f"- Exploration: ETF universe, one-year windows {res['windows'][0][0]} -> {res['windows'][-1][1]}, "
             f"fresh 100,000 each window, 15 bps costs, 15% kill switch, leverage pays T-bill + 1%, idle cash earns 0  \n"
             f"- Tickers not loaded: {', '.join(res['missing']) or 'none'}\n")
    L.append("| strategy | trial | Sharpe | CAGR | vol | max DD (stitched) | worst year | survived | turnover/yr | "
             "avg exposure | Sharpe - 60/40 | p (Sharpe <= 60/40) | CAGR at 10% vol | penalized Sharpe | DSR prob | "
             "Sharpe 2x costs | min neighbour Sharpe | shifted windows Sharpe | Sharpe if cash earned T-bill (P5) | "
             "finalist | verdict | P4 (NOT active) |")
    L.append("|" + "---|" * 22)
    for n in names + ["bench:60_40", "bench:buy_and_hold"]:
        s = summ[n]
        is_h = n in names
        L.append(f"| {n} | {st.get(n, 'benchmark')} | {_num(s.get('mean_sharpe'))} | {_pct(s.get('cagr'))} | "
                 f"{_plain(s.get('ann_vol'), 1)} | {_plain(s.get('mean_stitched_mdd'), 1)} | {_pct(s.get('worst_window'))} | "
                 f"{s['survived']}/{s['n_ok']} | {_num(s.get('mean_turnover'), 1)} | {_num(s.get('avg_gross'))} | "
                 f"{_signed(s.get('sharpe_minus_60_40'))} | {_num(s.get('p_vs_60_40'), 3)} | {_pct(s.get('cagr_at_10vol'))} | "
                 f"{_num(s.get('penalized_sharpe'))} | {_num(s.get('dsr_prob'))} | {_num(s.get('costs_2x_sharpe'))} | "
                 f"{_num(s.get('neighbor_min_sharpe'))} | {_num(s.get('shifted_sharpe'))} | "
                 f"{_num(s.get('with_cash_interest_sharpe'))} | "
                 f"{('yes' if n in res['finalists'] else 'no') if is_h else '-'} | {s.get('verdict', 'benchmark')} | "
                 f"{('yes' if s.get('p4') else 'no') if is_h else '-'} |")
    L.append("\n**Round-5 measurement checks (informational, see PROPOSALS P7/P8):** performance WITHOUT the "
             "kill switch (what a holder would actually have had) and the excess-return Sharpe (returns above T-bills).\n")
    L.append("| strategy | Sharpe w/o kill switch | CAGR w/o kill switch | excess Sharpe w/o kill switch |\n|---|---|---|---|")
    for n in names + ["bench:60_40", "bench:buy_and_hold"]:
        s = summ[n]
        L.append(f"| {n} | {_num(s.get('nokill_sharpe'))} | {_pct(s.get('nokill_cagr'))} | {_num(s.get('nokill_excess_sharpe'))} |")
    if res.get("p5"):
        p = res["p5"]
        L.append(f"\n**P5 recheck (diagnostic trial R5_D1, H3 trend):** Sharpe with idle cash at 0 = {_num(p['raw_sharpe_cash_0'])}; "
                 f"with cash at T-bill, RAW Sharpe = {_num(p['raw_sharpe_cash_tbill'])} but EXCESS-return Sharpe = "
                 f"{_num(p['excess_sharpe_cash_tbill'])} vs 60/40 {_num(p['excess_sharpe_60_40'])}, bootstrap p (not better) = "
                 f"{_num(p['p_vs_60_40_excess'], 3)}. Raw Sharpe counts T-bill interest as if it were skill; the excess-return "
                 "Sharpe removes it. Compare the two to see how much of P5's improvement was accounting.\n")
    L.append("\n*CAGR at 10% vol*: what the strategy would have earned run at 10% annual volatility (borrowing at "
             "T-bill + 1% if it needs leverage, idle cash at 0). It compares profit at EQUAL risk. *p*: paired block "
             "bootstrap probability that the Sharpe advantage over 60/40 is not real (smaller = more convincing).\n")
    L.append("### Why each hypothesis passed or failed\n")
    for n in names:
        L.append(f"- **{n}** ({st[n]}): {summ[n]['verdict']} - " + ("; ".join(summ[n]["reasons"]) or "all checks met"))
    if res["corr"] is not None:
        L.append("\n### Correlation of daily returns (exploration)\n")
        c = res["corr"]
        L.append("| | " + " | ".join(c.columns) + " |\n|---|" + "---|" * len(c.columns))
        for i in c.index:
            L.append(f"| {i} | " + " | ".join(f"{v:.2f}" for v in c.loc[i]) + " |")
    L.append(f"\n### Finalists (pre-registered rule: survival >= 80%, positive return at 2x costs, top 3 by Sharpe)\n\n"
             f"{', '.join(res['finalists']) or 'none'}\n")
    conf = res["confirmation"]
    L.append("### Confirmation (mutual funds, 1991-2006, never used before)\n")
    if conf.get("status") != "ok":
        L.append(f"Not run: {conf.get('status')}.\n")
    else:
        L.append("| strategy | Sharpe | CAGR | max DD | survived | verdict |\n|---|---|---|---|---|---|")
        for n, s in conf["summaries"].items():
            v = conf["verdicts"].get(n)
            L.append(f"| {n} | {_num(s.get('mean_sharpe'))} | {_pct(s.get('cagr'))} | {_plain(s.get('mean_stitched_mdd'), 1)} | "
                     f"{s.get('survived', 0)}/{s.get('n_ok', 0)} | "
                     f"{('PASS' if v['pass'] else 'FAIL: ' + '; '.join(v['reasons'])) if v else 'benchmark'} |")
        L.append(f"\nFunds not loaded: {', '.join(conf['missing']) or 'none'}. Data quality:\n")
        L.append("| fund | rows | start | end | jumps > 25% |\n|---|---|---|---|---|")
        for q in conf["quality"]:
            L.append(f"| {q['ticker']} | {q['rows']} | {q['start']} | {q['end']} | {', '.join(q['big_jumps']) or '-'} |")
    L.append("\n### Return per one-year window (exploration; DEAD = kill switch)\n")
    cols = names + ["bench:60_40", "bench:buy_and_hold"]
    L.append("| window | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    starts = [w[0] for w in res["wins"][cols[0]]]
    for i, ws in enumerate(starts):
        cells = []
        for n in cols:
            row = res["wins"][n][i] if i < len(res["wins"][n]) else None
            cells.append("-" if row is None else f"{row[1]:+.1%}" + (" DEAD" if row[2] else ""))
        L.append(f"| {ws} | " + " | ".join(cells) + " |")
    L.append("\n### Data quality (exploration universe)\n")
    L.append("| ticker | rows | start | end | jumps > 25% |\n|---|---|---|---|---|")
    for q in res["quality"]:
        L.append(f"| {q['ticker']} | {q['rows']} | {q['start']} | {q['end']} | {', '.join(q['big_jumps']) or '-'} |")
    L.append(f"\n![round 4]({plot_path.relative_to(ROOT).as_posix()})\n")
    md = "\n".join(L) + "\n"
    path.write_text("# RESEARCH - round 4\n\n" + md, encoding="utf-8")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    from run_all import legacy_trial_count, mode_dirs, prepare_cfg, setup_logging
    base_raw = load_base()
    base = prepare_cfg(base_raw, args.synthetic, smoke=False)
    setup_logging(base, "research_" + ("synthetic" if args.synthetic else "real"))
    res_root, _ = mode_dirs(base, args.synthetic, False)
    approved = bool((base_raw.get("research") or {}).get("confirmation_approved", False))
    res = run_research(base, res_root, args.synthetic, legacy_trial_count(res_root, base_raw), approved)
    md = ROOT / ("RESEARCH.md" if not args.synthetic else "RESEARCH_SYNTHETIC.md")
    write_report(res, md, res_root / "plots" / "round4.png")
    log.info("wrote %s", md.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
