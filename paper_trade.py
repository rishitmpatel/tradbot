"""Forward paper-trading tracker (round 4) - the only evidence that can't be overfit.

    python paper_trade.py           (run_all.py also calls it; run it any day you like)

What it does each time it runs:
  * downloads fresh daily prices for the exploration universe (cache: data/live)
  * for every trading day since the last logged day, simulates the finalists from
    results/real/research/finalists.json plus the 60/40 and buy-and-hold benchmarks:
    same rules, same costs (15 bps), same leverage financing, same 15% kill switch
  * APPENDS one row per day to paper/<strategy>.csv:
        date, equity, peak, drawdown, dead, cash, gross, w_<ticker>... (weights held after the close)
    and the closes used to paper/prices.csv
Rows already written are never changed. Commit the paper/ folder after each run, so git
history proves nothing was edited afterwards.

Missed a few days? The next run fills them in, using only data up to each day (the rules
never look ahead). That gives the same result as running every day.

The tracker starts AFTER the locked test set ends (data/raw/split_lock_yahoo.json). It reads
earlier prices only as look-back inputs for signals, and never records performance
for any day inside the test set.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from allocation import benchmark_weights, build_weights, rebalance_days, rebalance_rule
from config_utils import ROOT, deep_merge, load_base

log = logging.getLogger("paper_trade")
PAPER_DIR = ROOT / "paper"


def strategy_weights(plan: dict, names: list[str], panel: pd.DataFrame, rf: pd.Series) -> dict:
    """{name: (target weights, rebalance mask)} for finalists + benchmarks."""
    specs, cache, out = plan["hypotheses"], {}, {}
    for n in names:
        if n.startswith("bench:"):
            E = plan["universes"]["exploration"]["benchmarks"]
            kind = "buy_and_hold" if n == "bench:buy_and_hold" else "sixty_forty"
            W = benchmark_weights(kind, panel, E["equity"], E["bonds"])
            reb = np.zeros(len(panel), bool) if kind == "buy_and_hold" else rebalance_days(panel.index)
        else:
            W = build_weights(n, specs[n], specs, panel, rf, None, cache)
            reb = rebalance_days(panel.index, rebalance_rule(specs[n], specs))
        out[n] = (W, reb)
    return out


def update_tracker(panel: pd.DataFrame, rf: pd.Series, plan: dict, names: list[str], out_dir: Path,
                   start: pd.Timestamp, initial: float = 100_000.0) -> dict:
    """Append new days to each strategy's CSV. Pure function of the data -> testable.
    Returns {name: number of rows appended}."""
    common = plan["common"]
    rate = float(common["costs_bps"]) / 1e4
    spread = float(common["borrow_spread"]) / 252.0
    kill = float(common["max_drawdown_kill"])
    out_dir.mkdir(parents=True, exist_ok=True)
    days = panel.index[panel.index >= start]
    if len(days) == 0:
        return {n: 0 for n in names}
    strats = strategy_weights(plan, names, panel, rf)
    cols = list(panel.columns)
    appended = {}
    for n in names:
        path = out_dir / f"{n.replace(':', '_')}.csv"
        if path.exists():
            hist = pd.read_csv(path, parse_dates=["date"])
            last = hist.iloc[-1]
            last_date = pd.Timestamp(last["date"])
            equity, peak, dead = float(last["equity"]), float(last["peak"]), bool(last["dead"])
            held = np.array([float(last.get(f"w_{c}", 0.0)) for c in cols])
            new_days = days[days > last_date]
            prev = last_date
        else:
            equity, peak, dead, held = initial, initial, False, np.zeros(len(cols))
            new_days, prev = days, None
        W, reb = strats[n]
        reb = pd.Series(reb, index=panel.index)
        rows = []
        for t in new_days:
            if prev is not None and not dead:                       # 1) earn yesterday-close -> today-close
                r = (panel.loc[t] / panel.loc[prev] - 1.0).fillna(0.0).values
                gross_ret = float(held @ r)
                inv = float(held.sum())
                if inv > 1.0:
                    gross_ret -= (inv - 1.0) * (float(rf.get(prev, 0.0)) + spread)
                equity *= 1.0 + gross_ret
                held = held * (1.0 + r) / (1.0 + gross_ret) if 1.0 + gross_ret > 0 else np.zeros(len(cols))
                peak = max(peak, equity)
                if 1.0 - equity / peak > kill:                       # the agent dies: liquidate, stay flat
                    equity *= 1.0 - np.abs(held).sum() * rate
                    held, dead = np.zeros(len(cols)), True
            if not dead and (prev is None or reb.loc[t]):            # 2) decide at today's close
                want = W.loc[t].fillna(0.0).values
                equity *= 1.0 - np.abs(want - held).sum() * rate
                held = want
            peak = max(peak, equity)
            rows.append({"date": t.date().isoformat(), "equity": round(equity, 2), "peak": round(peak, 2),
                         "drawdown": round(1.0 - equity / peak, 5), "dead": dead,
                         "cash": round(1.0 - float(held.sum()), 5), "gross": round(float(held.sum()), 5),
                         **{f"w_{c}": round(float(w), 6) for c, w in zip(cols, held)}})
            prev = t
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
        appended[n] = len(rows)
    px = out_dir / "prices.csv"
    done = pd.read_csv(px, parse_dates=["date"])["date"].max() if px.exists() else None
    new_px = panel.loc[days if done is None else days[days > done]]
    if len(new_px):
        new_px.round(6).rename_axis("date").reset_index().assign(
            date=lambda d: d["date"].dt.date.astype(str)).to_csv(px, mode="a", header=not px.exists(), index=False)
    return appended


def write_summary(out_dir: Path, names: list[str], state: dict) -> str:
    L = ["# Forward paper trading (append-only log)\n",
         f"Started {state['start']} (after the locked test set ended {state['test_end']}). Strategies registered "
         f"{state['registered_at']}. Paper money only.\n",
         "| strategy | days | equity | return since start | max drawdown | status |", "|---|---|---|---|---|---|"]
    for n in names:
        p = out_dir / f"{n.replace(':', '_')}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        eq = d["equity"]
        mdd = float((1 - eq / eq.cummax()).max())
        L.append(f"| {n} | {len(d)} | {eq.iloc[-1]:,.0f} | {eq.iloc[-1] / eq.iloc[0] - 1:+.2%} | {mdd:.1%} | "
                 f"{'DEAD' if d['dead'].iloc[-1] else 'alive'} |")
    L.append("\nUnder about a year this says almost nothing: one year of returns cannot separate skill from luck. "
             "Judge after 12+ months, against the 60/40 row.\n")
    (out_dir / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def run_tracker(base_raw: dict, synthetic: bool = False, panel=None, rf=None) -> str | None:
    """Load finalists + fresh data, append new days. Returns the summary markdown (or None)."""
    from data import get_split, load_prices
    from research import load_plan
    from run_all import mode_dirs, prepare_cfg

    base = prepare_cfg(base_raw, synthetic, smoke=False)
    res_root, _ = mode_dirs(base, synthetic, False)
    fin_path = res_root / "research" / "finalists.json"
    if not fin_path.exists():
        log.warning("paper tracker: no finalists yet (run run_all.py first)")
        return None
    finalists = json.loads(fin_path.read_text(encoding="utf-8"))["finalists"]
    plan = load_plan()
    out_dir = PAPER_DIR if not synthetic else ROOT / "paper_synthetic"
    state_path = out_dir / "state.json"
    split = get_split(base, load_prices(base))                 # the locked test set ends at split.data_end
    if panel is None:
        E = plan["universes"]["exploration"]
        cfg = deep_merge(base, {"data": {"start": E["start"], "cache_dir": "data/live", "source": "yahoo"}})
        prices = {}
        for t in E["tickers"]:
            try:
                prices.update(load_prices(cfg, [t], refresh=True))
            except Exception as e:
                log.warning("paper tracker: %s not loaded (%s)", t, e)
        panel = pd.DataFrame({t: df["close"] for t, df in prices.items()}).sort_index().ffill()
        try:
            irx = load_prices(cfg, [plan["universes"]["risk_free"]], refresh=True)[plan["universes"]["risk_free"]]["close"]
            rf = (irx / 100.0 / 252.0).clip(lower=0.0).reindex(panel.index).ffill().fillna(0.03 / 252)
        except Exception:
            rf = pd.Series(0.03 / 252, index=panel.index)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        after_lock = panel.index[panel.index > split.data_end]
        if len(after_lock) == 0:
            log.warning("paper tracker: no trading day after the locked test set yet")
            return None
        state = {"start": str(after_lock[-1].date()), "test_end": str(split.data_end.date()),
                 "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "strategies": finalists + ["bench:60_40", "bench:buy_and_hold"]}
        out_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    start = pd.Timestamp(state["start"])
    assert start > split.data_end, "paper tracker would record days inside the locked test set"
    names = state["strategies"]
    if set(names) - {"bench:60_40", "bench:buy_and_hold"} != set(finalists):
        log.warning("paper tracker: finalists changed since tracking started (%s); keeping the original list",
                    finalists)
    added = update_tracker(panel, rf, plan, names, out_dir, start)
    log.info("paper tracker: appended %s", added)
    return write_summary(out_dir, names, state)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    md = run_tracker(load_base(), args.synthetic)
    if md:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
