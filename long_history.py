"""Long-history check for RULES (round 3) - INFORMATIONAL, not part of pass/fail
unless you approve proposal P2 in PROPOSALS.md.

Why: rounds 1-2 validated on 2016-2024, a strong bull market for this basket (buy & hold
Sharpe 1.06 - far above long-run norms) with no 2008-style crash. Rules need no
training window, so they can be checked on ALL history after a feature warm-up. ETF data
for the basket goes back to late 2004 (GLD started Nov 2004), so this covers 2007-2024,
including 2008 and 2011.

The 2007-2015 part has NEVER been used to evaluate any rule. The rules' parameters were
fixed in round 2 without looking at it, so it works as a second, untouched hold-out for them -
as long as nobody tunes on it afterwards.

The locked test set stays locked: the cut-off date is read from the SAME split lock as
the main pipeline (data/raw/split_lock_yahoo.json) and asserted.

    python long_history.py            (downloads to data/raw_long the first time)
    python long_history.py --synthetic
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from backtest import compute_metrics
from config_utils import CONFIGS_DIR, ROOT, deep_merge, list_experiment_files, load_base, load_experiment
from data import dev_view, get_split, load_prices, quality_report
from features import build_assets
from rules import is_rule, run_rule_strategy
from run_all import mode_dirs, mode_name, prepare_cfg, setup_logging
from walkforward import make_folds, run_baseline, stitched_returns

log = logging.getLogger("long_history")

PERIODS = {"2007-2015 (never used before)": ("2000-01-01", "2015-12-31"),
           "2016-2024 (rounds 1-2 validation period)": ("2016-01-01", "2100-01-01"),
           "all": ("2000-01-01", "2100-01-01")}


def period_stats(rets: pd.Series, runs: list[dict], lo: str, hi: str) -> dict:
    r = rets[(rets.index >= lo) & (rets.index <= hi)]
    if len(r) < 60:
        return {}
    m = compute_metrics(r)
    in_p = [x for x in runs if lo <= x["fold"]["val_start"] <= hi]
    alive = sum(1 for x in in_p if not x["metrics"]["dead"])
    return {"sharpe": m["sharpe"], "cagr": m["cagr"], "mdd": m["max_drawdown"], "survived": f"{alive}/{len(in_p)}",
            "years": m["years"]}


def run_long_history(base_raw: dict, synthetic: bool, start: str = "2005-01-01", smoke: bool = False):
    """Evaluate every baseline and every rule config over all history before the locked test set.
    Returns (strategies {name: runs}, folds, quality, split)."""
    main_cfg = prepare_cfg(base_raw, synthetic, smoke=smoke)
    res_root, _ = mode_dirs(main_cfg, synthetic, smoke)
    out_dir = res_root / "long_history"
    split = get_split(main_cfg, load_prices(main_cfg))          # SAME lock as the main pipeline
    over = {"data": {"start": start, "cache_dir": "data/raw_long"},
            "walk_forward": {"min_train_years": 0}}
    long_cfg = deep_merge(main_cfg, over)
    prices = load_prices(long_cfg)
    quality = quality_report(prices)
    dev = dev_view(prices, split)
    assert all(df.index.max() < split.test_start for df in dev.values()), "test set leaked into long history"
    assets = build_assets(dev, long_cfg["features"])
    folds = make_folds(long_cfg, assets, split)
    log.info("long history: %d one-year windows %s -> %s; test set (%s+) untouched",
             len(folds), folds[0].val_start, folds[-1].val_end, split.test_start.date())

    strategies = {}
    for name in long_cfg["baselines"]:
        strategies[f"baseline:{name}"] = run_baseline(long_cfg, name, folds, out_dir / "baselines" / name, assets)
    for f in list_experiment_files(CONFIGS_DIR):
        c = deep_merge(prepare_cfg(load_experiment(f, base_raw), synthetic, smoke=smoke), over)
        if is_rule(c):
            strategies[c["name"]] = run_rule_strategy(c, assets, folds, out_dir / "rules" / c["name"])
    return strategies, folds, quality, split


def early_period_runs(runs: list[dict], before: str = "2016-01-01") -> list[dict]:
    """Windows that start before the rounds 1-2 validation period (2007-2015)."""
    return [r for r in runs if r["fold"]["val_start"] < before]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", default="2005-01-01")
    args = ap.parse_args()
    base_raw = load_base()
    setup_logging(prepare_cfg(base_raw, args.synthetic, False), "long_history_" + mode_name(args.synthetic, False))
    strategies, folds, quality, split = run_long_history(base_raw, args.synthetic, args.start)

    L = ["# LONG HISTORY (rules only) - informational, NOT part of pass/fail\n"]
    if args.synthetic:
        L.append("> **SYNTHETIC DATA - NOT REAL RESULTS.**\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. One-year windows from {folds[0].val_start} to "
             f"{folds[-1].val_end}, fresh 100,000 each window, same costs and 15% kill switch as the main "
             f"pipeline. Locked test set ({split.test_start.date()} onwards) NOT used.\n")
    L.append("Rules need no training, so every year after the 2-year warm-up is out-of-sample for them. "
             "**2007-2015 was never used to evaluate or choose any rule** - treat it as a second hold-out and "
             "do not tune on it.\n")
    for pname, (lo, hi) in PERIODS.items():
        L.append(f"## {pname}\n")
        L.append("| strategy | Sharpe | CAGR | max DD (stitched) | survived windows |\n|---|---|---|---|---|")
        for sname, runs in strategies.items():
            st = period_stats(stitched_returns(runs, 0), runs, lo, hi)
            if st:
                L.append(f"| {sname} | {st['sharpe']:.2f} | {st['cagr']:+.1%} | {st['mdd']:.1%} | {st['survived']} |")
        L.append("")
    L.append("## Return per one-year window (DEAD = kill switch hit)\n")
    names = list(strategies)
    L.append("| window start | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for i, f in enumerate(folds):
        cells = []
        for n in names:
            m = strategies[n][i]["metrics"]
            cells.append(f"{m['total_return']:+.1%}" + (" DEAD" if m["dead"] else ""))
        L.append(f"| {f.val_start} | " + " | ".join(cells) + " |")
    L.append("\n## Data quality (long history)\n")
    L.append("| ticker | rows | start | end | jumps > 25% |\n|---|---|---|---|---|")
    for q in quality:
        L.append(f"| {q['ticker']} | {q['rows']} | {q['start']} | {q['end']} | {', '.join(q['big_jumps']) or '-'} |")
    md = ROOT / ("LONG_HISTORY.md" if not args.synthetic else "LONG_HISTORY_SYNTHETIC.md")
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("wrote %s", md.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
