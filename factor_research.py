"""Round 5 research stage: long-history premia on Ken French data (1926-2024).

    python factor_research.py            (also called automatically by run_all.py)
    python factor_research.py --synthetic

Why long history: with 18 years a Sharpe ratio has a standard error of ~0.24, so round 4
could not tell 0.67 from 0.60. With 61 years of exploration it is ~0.13, and the paired
test of two related strategies is sharper still. More data beats more cleverness.

Every strategy AND the benchmark (the market) is run the same way:
  * sleeve return = mix of portfolios - their yearly cost drag (fund fees + internal turnover)
  * optional trend filter (market's trailing 12-month return must beat T-bills, else T-bills)
  * optional turn-of-month timing (daily data; in the market only around month end)
  * scaled each month to ~10% volatility (trailing 12 months), no leverage, rest in T-bills
  * 15 bps on every change of our own position, same 15% kill switch per calendar year
Comparison = Sharpe of EXCESS returns (return minus T-bills). At equal risk, a higher
excess Sharpe means more profit; that is the honest "does it beat the market" test.

Exploration: US 1963-07 -> month before the locked test set. Confirmation (after owner approval,
once): US 1927-1963 (never used) + other developed markets 1991-2024 (factor tilts never used).
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

from allocation import tom_position_mask
from config_utils import ROOT, load_base
from data import get_split, load_prices
from evaluation import base_checks, bootstrap_sharpe_diff, deflated_sharpe_prob, penalize
from reporting import save_json

log = logging.getLogger("factor_research")
PLAN5 = ROOT / "research" / "plan_round5.yaml"


# --------------------------------------------------------------------------
# Plan, registration, ledger
# --------------------------------------------------------------------------
def load_plan(path: Path = PLAN5) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def spec_hash(spec: dict) -> str:
    core = {k: v for k, v in spec.items() if k != "reason"}
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:10]


def all_trials(plan: dict) -> dict:
    """Hypotheses + diagnostics: everything that counts as a trial."""
    return {**plan["hypotheses"], **(plan.get("diagnostics") or {})}


def register_round5(plan: dict, res_root: Path) -> tuple[dict, int]:
    """Freeze the registered list on the first run; add every trial to the ledger ('r5:' keys).
    Returns ({name: registered|exploratory|diagnostic}, number of r5 trials in the ledger)."""
    rdir = res_root / "research5"
    rdir.mkdir(parents=True, exist_ok=True)
    reg_path = rdir / "registration.json"
    trials = all_trials(plan)
    hashes = {n: spec_hash(s) for n, s in trials.items()}
    if not reg_path.exists():
        save_json(reg_path, {"registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "plan_sha": hashlib.sha1(PLAN5.read_bytes()).hexdigest()[:12], "trials": hashes})
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["trials"]
    status = {}
    for n, h in hashes.items():
        if reg.get(n) != h:
            status[n] = "exploratory"
        else:
            status[n] = "diagnostic" if n in (plan.get("diagnostics") or {}) else "registered"
    lp = res_root / "trial_ledger.json"
    ledger = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {"trials": {}}
    for n, h in hashes.items():
        ledger["trials"].setdefault(f"r5:{n}@{h}", {"round": 5, "kind": status[n],
                                                    "first_seen": time.strftime("%Y-%m-%d %H:%M")})
    save_json(lp, ledger)
    return status, sum(1 for k in ledger["trials"] if k.startswith("r5:"))


def count_prefix(res_root: Path, prefix: str) -> int:
    lp = res_root / "trial_ledger.json"
    if not lp.exists():
        return 0
    return sum(1 for k in json.loads(lp.read_text(encoding="utf-8"))["trials"] if k.startswith(prefix))


# --------------------------------------------------------------------------
# Signals (all known at the END of the previous month; selfcheck verifies)
# --------------------------------------------------------------------------
def tom_sleeve_monthly(daily: pd.DataFrame, before: int, after: int, cost: float) -> pd.Series:
    """Monthly return of 'market on turn-of-month days, T-bills otherwise' at full exposure,
    including 15 bps per switch. Position decided at close d earns day d+1's return."""
    on = pd.Series(tom_position_mask(daily.index, before, after), index=daily.index).astype(float)
    pos = on.shift(1).fillna(0.0)                               # position held during day t
    switch = (pos - pos.shift(1).fillna(0.0)).abs()
    r = pos * daily["mkt"] + (1 - pos) * daily["rf"] - switch * cost
    g = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    g.index = g.index.to_timestamp(how="end").normalize()
    return g


def sleeve_returns(spec: dict, common: dict, data: dict, drag_mult: float = 1.0, cost_mult: float = 1.0) -> pd.Series:
    """Monthly sleeve return at full exposure (before vol targeting / trend)."""
    m = data["monthly"]
    if spec.get("turn_of_month"):
        tom = common.get("tom_days", {})
        s = tom_sleeve_monthly(data["daily"], int(tom.get("before", 1)), int(tom.get("after", 3)),
                               common["switch_cost_bps"] / 1e4 * cost_mult)
        return (s - common["sleeve_drag_bps_per_year"]["market"] / 1e4 / 12 * drag_mult).reindex(m.index)
    drag = common["sleeve_drag_bps_per_year"]
    s = sum(w * (m[k] - drag[k] / 1e4 / 12 * drag_mult) for k, w in spec["sleeve"].items())
    return s


def positions(spec: dict, common: dict, data: dict, sleeve: pd.Series) -> pd.Series:
    """Fraction of equity in the sleeve for each month, decided at the previous month end."""
    m = data["monthly"]
    ex = sleeve - m["rf"]
    lb = int(common["vol_lookback_months"])
    vol = ex.rolling(lb, min_periods=lb).std().shift(1) * np.sqrt(12)        # known at previous month end
    w = (float(common["vol_target"]) / vol).clip(upper=float(common["max_gross"]))
    if spec.get("trend"):
        L = int(spec["trend"]["lookback_months"])
        mk = (1 + m["market"]).rolling(L, min_periods=L).apply(np.prod, raw=True)
        rf = (1 + m["rf"]).rolling(L, min_periods=L).apply(np.prod, raw=True)
        on = (mk - rf > 0).astype(float).where(mk.notna()).shift(1)
        w = w * on
    return w


# --------------------------------------------------------------------------
# Monthly engine: calendar-year windows, fresh 100,000, kill switch, costs
# --------------------------------------------------------------------------
def run_monthly(sleeve: pd.Series, rf: pd.Series, pos: pd.Series, cost: float, kill: float | None,
                start: str, end: pd.Timestamp, single_window: bool = False) -> tuple[pd.Series, list[dict]]:
    idx = sleeve.index[(sleeve.index >= pd.Timestamp(start)) & (sleeve.index <= end)]
    ok = sleeve.reindex(idx).notna() & pos.reindex(idx).notna() & rf.reindex(idx).notna()
    idx = idx[ok.values]
    rets, wins = [], []
    groups = [(0, pd.Series(idx, index=idx))] if single_window else pd.Series(idx, index=idx).groupby(idx.year)
    for year, months in groups:
        if len(months) < 6 and not single_window:
            continue
        eq, peak, held, dead = 1e5, 1e5, 0.0, False
        start_eq = eq
        for t in months.index:
            if dead:
                rets.append((t, 0.0))
                continue
            p, s, f = float(pos[t]), float(sleeve[t]), float(rf[t])
            c = abs(p - held) * cost
            port = f + p * (s - f)
            r = (1 - c) * (1 + port) - 1
            held = p * (1 + s) / (1 + port) if 1 + port > 0 else 0.0
            new = eq * (1 + r)
            peak = max(peak, new)
            if kill is not None and 1 - new / peak > kill:
                new *= 1 - held * cost
                dead, held = True, 0.0
            rets.append((t, new / eq - 1))
            eq = new
        wins.append({"year": int(year), "return": eq / start_eq - 1, "dead": dead})
    s = pd.Series(dict(rets)).sort_index()
    return s, wins


def stats(rets: pd.Series, rf: pd.Series, wins: list[dict]) -> dict:
    if len(rets) < 24:
        return {"n_ok": 0, "survived": 0, "complete": False}
    ex = rets - rf.reindex(rets.index).fillna(0.0)
    years = len(rets) / 12
    eq = (1 + rets).cumprod()
    sd = ex.std(ddof=1)
    x = ex.values
    alive = sum(1 for w in wins if not w["dead"])
    return {"mean_sharpe": float(ex.mean() / sd * np.sqrt(12)) if sd > 0 else 0.0,     # EXCESS-return Sharpe
            "cagr": float(eq.iloc[-1] ** (1 / years) - 1), "excess_cagr": float((1 + ex).prod() ** (1 / years) - 1),
            "mean_return": float(eq.iloc[-1] - 1), "ann_vol": float(rets.std(ddof=1) * np.sqrt(12)),
            "mean_stitched_mdd": float((1 - eq / eq.cummax()).max()),
            "survived": alive, "n_ok": len(wins), "n_runs": len(wins), "complete": True,
            "survival_rate": alive / len(wins) if wins else 0.0,
            "n_days": len(rets), "years": years,
            "skew": float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0,
            "kurt": float(((x - x.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0,
            "first": str(rets.index[0].date()), "last": str(rets.index[-1].date()), "_rets": rets, "_ex": ex}


def evaluate(spec: dict, common: dict, data: dict, start: str, end: pd.Timestamp, kill=True,
             cost_mult: float = 1.0) -> dict:
    sl = sleeve_returns(spec, common, data, drag_mult=cost_mult, cost_mult=cost_mult)
    pos = positions(spec, common, data, sl)
    k = float(common["kill_switch"]) if kill else None
    rets, wins = run_monthly(sl, data["monthly"]["rf"], pos, common["switch_cost_bps"] / 1e4 * cost_mult, k, start, end)
    out = stats(rets, data["monthly"]["rf"], wins)
    out["avg_exposure"] = float(pos.reindex(rets.index).mean()) if len(rets) else float("nan")
    out["_wins"] = wins
    return out


def compare(s: dict, b: dict) -> tuple[float, float]:
    """(excess-Sharpe difference vs benchmark, paired block-bootstrap p-value of 'not better')."""
    a, c = s["_ex"].align(b["_ex"], join="inner")
    return s["mean_sharpe"] - b["mean_sharpe"], bootstrap_sharpe_diff(a.values, c.values, block=12)


SUBPERIODS = [("1963-07", "1990-12"), ("1991-01", "2006-12"), ("2007-01", "2100-01")]


# --------------------------------------------------------------------------
# Main stage
# --------------------------------------------------------------------------
def load_data(plan: dict, cutoff: pd.Timestamp, synthetic: bool, region: str | None = None) -> dict:
    from factor_data import load_market, synthetic_market
    if synthetic:
        seed = 5 if region is None else 50 + len(region)
        start = "1926-07-01" if region is None else "1990-07-01"
        d = synthetic_market(seed, start, str((cutoff - pd.Timedelta(days=1)).date()))
        return {k: v[v.index < cutoff] for k, v in d.items()}
    spec = plan["data"]["us"] if region is None else plan["data"]["regions"][region]
    return load_market(spec, plan["data"]["base_url"], cutoff)


def run_round5(base: dict, res_root: Path, synthetic: bool, legacy_trials: int, approved: bool) -> dict:
    plan = load_plan()
    common = plan["common"]
    status, n_r5 = register_round5(plan, res_root)
    n_r4 = count_prefix(res_root, "r4:")
    n_total = legacy_trials + n_r4 + n_r5
    rdir = res_root / "research5"
    split = get_split(base, load_prices(base))
    cutoff = pd.Timestamp(split.test_start.year, split.test_start.month, 1)      # whole months before the test set
    log.info("round 5: %d trials (%s); total trials = %d + %d + %d = %d", n_r5, status, legacy_trials, n_r4, n_r5, n_total)

    us = load_data(plan, cutoff, synthetic)
    ex = plan["periods"]["exploration"]
    ex_end = cutoff - pd.Timedelta(days=1)
    hyps = plan["hypotheses"]
    bench_spec = {"sleeve": {"market": 1.0}}
    summ = {"bench:market": evaluate(bench_spec, common, us, ex["start"], ex_end)}
    nokill = {"bench:market": evaluate(bench_spec, common, us, ex["start"], ex_end, kill=False)}
    for n, spec in hyps.items():
        summ[n] = evaluate(spec, common, us, ex["start"], ex_end)
        nokill[n] = evaluate(spec, common, us, ex["start"], ex_end, kill=False)
        log.info("exploration %-28s excess Sharpe %.2f CAGR %+.1f%% maxDD %.1f%% survived %d/%d", n,
                 summ[n].get("mean_sharpe", float("nan")), 100 * summ[n].get("cagr", float("nan")),
                 100 * summ[n].get("mean_stitched_mdd", float("nan")), summ[n]["survived"], summ[n]["n_ok"])
    b = summ["bench:market"]
    cross = [summ[n]["mean_sharpe"] for n in hyps]
    pc = base["pass_criteria"]
    for n, spec in hyps.items():
        s = summ[n]
        penalize(s, n_total, cross)
        s["status"] = status[n]
        s["sharpe_minus_bench"], s["p_vs_bench"] = compare(s, b)
        s["nokill_sharpe"], s["nokill_cagr"] = nokill[n].get("mean_sharpe"), nokill[n].get("cagr")
        c2 = evaluate(spec, common, us, ex["start"], ex_end, cost_mult=2.0)
        s["costs_2x_sharpe"], s["costs_2x_excess_cagr"] = c2.get("mean_sharpe"), c2.get("excess_cagr")
        s["subperiods"] = {}
        for lo, hi in SUBPERIODS:
            sub = evaluate(spec, common, us, max(lo, ex["start"]), min(pd.Timestamp(hi), ex_end))
            bsub = evaluate(bench_spec, common, us, max(lo, ex["start"]), min(pd.Timestamp(hi), ex_end))
            s["subperiods"][f"{lo[:4]}-{hi[:4] if hi < '2100' else 'now'}"] = (
                sub.get("mean_sharpe", float("nan")) - bsub.get("mean_sharpe", float("nan")))
    b["nokill_sharpe"], b["nokill_cagr"] = nokill["bench:market"].get("mean_sharpe"), nokill["bench:market"].get("cagr")

    # finalists (pre-registered rule)
    fs = plan["finalist_selection"]
    eligible = [n for n in hyps if summ[n]["survival_rate"] >= 0.8 and (summ[n].get("costs_2x_excess_cagr") or -1) > 0]
    finalists = sorted(eligible, key=lambda n: -summ[n]["mean_sharpe"])[: int(fs["max_finalists"])]
    lock = rdir / "confirmation_LOCK.json"
    if lock.exists():
        locked = json.loads(lock.read_text(encoding="utf-8"))["finalists"]
        if locked != finalists:
            log.error("FINALISTS CHANGED after confirmation (%s -> %s). Using the locked ones.", locked, finalists)
        finalists = locked
    save_json(rdir / "finalists.json", {"finalists": finalists, "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "rule": fs})
    log.info("round 5 finalists: %s", finalists)

    # confirmation (owner-approved)
    conf = {"status": "not approved yet (set research5.confirmation_approved: true in base_config.yaml)"}
    if approved and finalists:
        crit = plan["confirmation_criteria"]
        sets = [("US 1927-1963", None, plan["periods"]["confirmation_us"], True)]
        for reg, rspec in plan["data"]["regions"].items():
            sets.append((reg, reg, plan["periods"]["confirmation_regions"], bool(rspec.get("required"))))
        conf = {"status": "ok", "sets": {}, "verdicts": {}, "errors": {}}
        for label, region, per, required in sets:
            try:
                d = us if region is None else load_data(plan, cutoff, synthetic, region)
            except Exception as e:
                conf["errors"][label] = str(e)
                conf["sets"][label] = {"required": required, "loaded": False}
                log.error("confirmation data %s not loaded: %s", label, e)
                continue
            end = pd.Timestamp(per["end"]) + pd.offsets.MonthEnd(0) if per.get("end") else ex_end
            assert region is not None or end < pd.Timestamp(ex["start"]), "US confirmation overlaps exploration"
            res = {"bench:market": evaluate(bench_spec, common, d, per["start"], end)}
            for n in finalists:
                res[n] = evaluate(hyps[n], common, d, per["start"], end)
            ver = {}
            for n in finalists:
                s, why = res[n], []
                if s.get("mean_sharpe", -9) <= res["bench:market"].get("mean_sharpe", 9):
                    why.append(f"excess Sharpe {s.get('mean_sharpe', float('nan')):.2f} <= market "
                               f"{res['bench:market'].get('mean_sharpe', float('nan')):.2f}")
                if s.get("excess_cagr", -1) <= crit["excess_return_gt"]:
                    why.append(f"excess return {s.get('excess_cagr', float('nan')):+.1%}/yr")
                if s.get("survival_rate", 0) < crit["survival_rate_ge"]:
                    why.append(f"survival {s.get('survived', 0)}/{s.get('n_ok', 0)}")
                if region is None and crit.get("us_p_value_lt") is not None and "mean_sharpe" in s:
                    _, p = compare(s, res["bench:market"])
                    s["p_vs_bench"] = p
                    if not p < float(crit["us_p_value_lt"]):
                        why.append(f"advantage not significant (p {p:.2f} >= {crit['us_p_value_lt']})")
                ver[n] = {"pass": not why, "reasons": why}
            conf["sets"][label] = {"required": required, "loaded": True, "summ": res, "verdicts": ver}
        for n in finalists:
            req = [v for v in conf["sets"].values() if v["required"]]
            ok = all(v["loaded"] and v["verdicts"][n]["pass"] for v in req)
            conf["verdicts"][n] = {"pass": ok, "reasons": [f"{lab}: " + ("; ".join(v["verdicts"][n]["reasons"]) or "ok")
                                                           if v["loaded"] else f"{lab}: NOT LOADED (required)"
                                                           for lab, v in conf["sets"].items() if v["required"]]}
        if not lock.exists():
            save_json(lock, {"finalists": finalists, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    elif approved:
        conf = {"status": "no finalists"}

    # verdicts under the ACTIVE bar, plus proposal P9 (not active)
    n_h = len(hyps)
    for n in hyps:
        s = summ[n]
        ok, why = base_checks(s, pc)
        if s["mean_sharpe"] <= b["mean_sharpe"]:
            ok, why = False, why + [f"excess Sharpe {s['mean_sharpe']:.2f} not above the market's {b['mean_sharpe']:.2f}"]
        if (s.get("costs_2x_excess_cagr") or -1) <= 0:
            ok, why = False, why + ["not profitable over T-bills at 2x costs"]
        if n not in finalists:
            ok, why = False, why + ["not a finalist"]
        elif conf.get("status") != "ok":
            ok, why = False, why + ["confirmation not run"]
        elif not conf["verdicts"][n]["pass"]:
            ok, why = False, why + ["confirmation failed: " + " | ".join(conf["verdicts"][n]["reasons"])]
        s["verdict"], s["reasons"] = ("PASS" if ok else "FAIL"), why
        dsr = deflated_sharpe_prob(s["mean_sharpe"], s["sharpe_haircut"], s["n_days"], s["years"], s["skew"], s["kurt"])
        s["p9"] = bool(s["p_vs_bench"] < 0.05 / n_h and dsr > 0.95 and s["survival_rate"] >= 0.8
                       and (s.get("costs_2x_excess_cagr") or -1) > 0 and n in finalists
                       and conf.get("status") == "ok" and conf["verdicts"][n]["pass"])

    # Indian retail cost/tax reality check for finalists + market (last 20 years of exploration)
    from india_costs import india_reality
    india = {}
    passive = {"sleeve": {"market": 1.0}, "passive": True}
    for n in finalists + ["bench:market", "passive:buy_and_hold_market"]:
        spec = bench_spec if n == "bench:market" else passive if n.startswith("passive") else hyps[n]
        try:
            india[n] = india_reality(spec, common, us, ex_end)
        except Exception as e:
            log.error("india cost model failed for %s: %s", n, e)

    res = {"plan": plan, "status": status, "n_total": n_total, "n_r5": n_r5, "summ": summ, "finalists": finalists,
           "confirmation": conf, "india": india, "synthetic": synthetic, "explore_end": str(ex_end.date()),
           "plan_sha": hashlib.sha1(PLAN5.read_bytes()).hexdigest()[:12]}
    save_json(rdir / "summary.json", _clean(res))
    return res


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(res: dict, path: Path) -> str:
    from reporting import _num, _pct, _plain, _signed
    summ, plan, st = res["summ"], res["plan"], res["status"]
    names = list(plan["hypotheses"])
    L = ["## Round 5 research: century-long premia (Ken French data, pre-registered plan)\n"]
    if res["synthetic"]:
        L.append("> **SYNTHETIC DATA - NOT REAL RESULTS.**\n")
    b = summ["bench:market"]
    L.append(f"- Plan `research/plan_round5.yaml` sha `{res['plan_sha']}`; trial status: "
             + ", ".join(f"{n}={v}" for n, v in st.items()) + "  \n"
             f"- **Trials counted for the Sharpe penalty: {res['n_total']}** (all rounds, incl. {res['n_r5']} round-5)  \n"
             f"- Exploration: US {b.get('first')} -> {b.get('last')} ({b.get('years', 0):.0f} years, monthly). Every strategy "
             "AND the market are scaled to ~10% volatility (rest in T-bills, no leverage), 15 bps per position change, "
             "fund/turnover drag per sleeve, 15% kill switch per calendar year.  \n"
             "- *Excess Sharpe* = Sharpe of returns ABOVE T-bills (correct accounting; idle cash earns T-bills here).\n")
    L.append("| strategy | trial | excess Sharpe | vs market | p (not better) | CAGR | excess CAGR | vol | max DD | survived | "
             "avg exposure | Sharpe w/o kill switch | CAGR w/o kill switch | excess Sharpe 2x costs | penalized Sharpe | "
             "DSR prob | finalist | verdict | P9 (NOT active) |")
    L.append("|" + "---|" * 19)
    for n in names + ["bench:market"]:
        s = summ[n]
        h = n in names
        L.append(f"| {n} | {st.get(n, 'benchmark')} | {_num(s.get('mean_sharpe'))} | {_signed(s.get('sharpe_minus_bench'))} | "
                 f"{_num(s.get('p_vs_bench'), 3)} | {_pct(s.get('cagr'))} | {_pct(s.get('excess_cagr'))} | "
                 f"{_plain(s.get('ann_vol'), 1)} | {_plain(s.get('mean_stitched_mdd'), 1)} | {s.get('survived', 0)}/{s.get('n_ok', 0)} | "
                 f"{_num(s.get('avg_exposure'))} | {_num(s.get('nokill_sharpe'))} | {_pct(s.get('nokill_cagr'))} | "
                 f"{_num(s.get('costs_2x_sharpe'))} | {_num(s.get('penalized_sharpe'))} | {_num(s.get('dsr_prob'))} | "
                 f"{('yes' if n in res['finalists'] else 'no') if h else '-'} | {s.get('verdict', 'benchmark')} | "
                 f"{('yes' if s.get('p9') else 'no') if h else '-'} |")
    L.append("\n*p*: paired block-bootstrap (12-month blocks) probability that the strategy's excess Sharpe is NOT above "
             "the market's. *P9* (proposal, not active): p < 0.05 / number of round-5 hypotheses, deflated-Sharpe prob > 0.95 "
             "with ALL trials, survives, profitable at 2x costs, finalist, confirmed.\n")
    L.append("### Does the edge decay? Excess Sharpe minus the market's, by sub-period\n")
    periods = list(next(iter(summ[n]["subperiods"] for n in names)).keys())
    L.append("| strategy | " + " | ".join(periods) + " |\n|---|" + "---|" * len(periods))
    for n in names:
        L.append(f"| {n} | " + " | ".join(_signed(summ[n]["subperiods"][p]) for p in periods) + " |")
    L.append("\n### Why each hypothesis passed or failed\n")
    for n in names:
        L.append(f"- **{n}** ({st[n]}): {summ[n]['verdict']} - " + ("; ".join(summ[n]["reasons"]) or "all checks met"))
    L.append(f"\n### Finalists (pre-registered: survival >= 80%, beats T-bills at 2x costs, top 3 by excess Sharpe)\n\n"
             f"{', '.join(res['finalists']) or 'none'}\n")
    conf = res["confirmation"]
    L.append("### Confirmation (US 1927-1963 + other developed markets; never used before)\n")
    if conf.get("status") != "ok":
        L.append(f"Not run: {conf.get('status')}.\n")
    else:
        for lab, v in conf["sets"].items():
            if not v["loaded"]:
                L.append(f"- **{lab}** ({'required' if v['required'] else 'informational'}): NOT LOADED - "
                         f"{conf['errors'].get(lab, '')}")
                continue
            L.append(f"\n**{lab}** ({'required' if v['required'] else 'informational'})\n")
            L.append("| strategy | period | excess Sharpe | p (not better) | CAGR | max DD | survived | verdict |\n"
                     "|---|---|---|---|---|---|---|---|")
            for n, s in v["summ"].items():
                ver = v["verdicts"].get(n)
                L.append(f"| {n} | {s.get('first', '?')} -> {s.get('last', '?')} | {_num(s.get('mean_sharpe'))} | "
                         f"{_num(s.get('p_vs_bench'), 3)} | "
                         f"{_pct(s.get('cagr'))} | {_plain(s.get('mean_stitched_mdd'), 1)} | {s.get('survived', 0)}/{s.get('n_ok', 0)} | "
                         f"{('PASS' if ver['pass'] else 'FAIL: ' + '; '.join(ver['reasons'])) if ver else 'benchmark'} |")
        L.append("\nOverall: " + "; ".join(f"{n}: {'CONFIRMED' if v['pass'] else 'NOT confirmed'}"
                                           for n, v in conf["verdicts"].items()) + "\n")
    if res["india"]:
        L.append("### What an Indian retail investor would keep (last 20 years of the exploration pattern)\n")
        L.append("US returns used as a stand-in for the TRADING PATTERN; taxes/charges per india_costs.py (FY2025-26 rules, "
                 "slab 30%, liquidated at the end so buy & hold also pays its tax).\n")
        L.append("| strategy | pre-tax CAGR | route | Rs 1 lakh | Rs 10 lakh | Rs 1 crore |\n|---|---|---|---|---|---|")
        for n, r in res["india"].items():
            for reg, caps in r["regimes"].items():
                cells = " | ".join(f"{caps[c]['cagr']:+.1%} (tax {caps[c]['tax_paid'] / c:.0%}, costs {caps[c]['charges'] / c:.1%})"
                                   for c in sorted(caps))
                L.append(f"| {n} | {_pct(r['pre_tax_cagr'])} | {reg} | {cells} |")
        L.append("\nCells: after-cost, after-tax CAGR (total tax and charges as a share of starting capital). "
                 "*bench:market* is the market scaled to 10% volatility (it trades monthly); "
                 "*passive:buy_and_hold_market* is simply holding an index fund at full risk and selling once at the end "
                 "- the 'do nothing' alternative (higher risk, so compare its CAGR with care).\n")
    md = "\n".join(L) + "\n"
    path.write_text("# RESEARCH - round 5\n\n" + md, encoding="utf-8")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    from run_all import legacy_trial_count, mode_dirs, prepare_cfg, setup_logging
    base_raw = load_base()
    base = prepare_cfg(base_raw, args.synthetic, smoke=False)
    setup_logging(base, "research5_" + ("synthetic" if args.synthetic else "real"))
    res_root, _ = mode_dirs(base, args.synthetic, False)
    approved = bool((base_raw.get("research5") or {}).get("confirmation_approved", False))
    res = run_round5(base, res_root, args.synthetic, legacy_trial_count(res_root, base_raw) + count_prefix(res_root, "r6:"),
                     approved)
    md = ROOT / ("RESEARCH5.md" if not args.synthetic else "RESEARCH5_SYNTHETIC.md")
    write_report(res, md)
    log.info("wrote %s", md.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def final_test_round5(name: str, base: dict, synthetic: bool) -> dict:
    """ONE-TIME evaluation of a round-5 finalist on the locked test period (called only by final_test.py).
    Ken French data lags 1-2 months, so the covered period is reported."""
    plan = load_plan()
    common, spec = plan["common"], plan["hypotheses"][name]
    split = get_split(base, load_prices(base))
    cutoff = split.data_end + pd.Timedelta(days=1)
    d = load_data(plan, cutoff, synthetic)
    start = pd.Timestamp(split.test_start.year, split.test_start.month, 1) + pd.offsets.MonthEnd(0)
    out = {}
    for n, sp in ((name, spec), ("bench:market", {"sleeve": {"market": 1.0}})):
        sl = sleeve_returns(sp, common, d)
        pos = positions(sp, common, d, sl)
        rets, wins = run_monthly(sl, d["monthly"]["rf"], pos, common["switch_cost_bps"] / 1e4,
                                 float(common["kill_switch"]), str(start.date()), split.data_end, single_window=True)
        out[n] = stats(rets, d["monthly"]["rf"], wins) if len(rets) >= 24 else {
            "first": str(rets.index[0].date()) if len(rets) else None, "last": str(rets.index[-1].date()) if len(rets) else None,
            "months": len(rets), "total_return": float((1 + rets).prod() - 1) if len(rets) else None,
            "dead": bool(wins and wins[0]["dead"])}
    return out
