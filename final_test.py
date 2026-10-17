"""ONE-TIME evaluation on the locked test set (final `data.test_years` years).

Rules enforced here:
  * Only the strategy selected by run_all.py (passed ALL validation + robustness
    checks) is evaluated. If nothing passed, the test set is left untouched.
  * It runs ONCE. A LOCK file is written; running again is refused.
    Looking at the test set and then changing the strategy turns it into a
    validation set and the result stops meaning anything.
  * Nothing here feeds back into configs. Do not tune on these numbers.

Procedure: retrain the selected config on ALL development data (each seed),
trade the test period with 100,000 paper capital and the survival rule, and
compare with the baselines over the same period.

    python final_test.py
    python final_test.py --synthetic        (code testing only)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agents import agent_positions, make_segment, train_agent  # noqa: E402
from backtest import cost_rate, portfolio_backtest, window_sleeve  # noqa: E402
from baselines import baseline_positions  # noqa: E402
from config_utils import CONFIGS_DIR, ROOT, load_base, load_experiment, strategy_hash  # noqa: E402
from data import get_split, load_prices, test_view  # noqa: E402
from features import build_assets  # noqa: E402
from reporting import save_json  # noqa: E402
from rules import evaluate_window, is_rule  # noqa: E402
from run_all import mode_dirs, mode_name, prepare_cfg, setup_logging  # noqa: E402

log = logging.getLogger("final_test")


def final_test_research(name: str, base: dict, out_dir, lock, md, synthetic: bool, sel: dict) -> int:
    """One-time test of a round-4 allocation rule on the locked period (exploration universe)."""
    from research import (benchmark_set, load_plan, load_universe, register, run_windows, spec_hash,
                          strategy_set, summarize_windows)

    plan = load_plan()
    reg = register(plan, out_dir.parent / "research")
    if reg.get(name) != "registered":
        log.error("%s is not a registered, unchanged hypothesis (%s). Refusing.", name, reg.get(name))
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(lock, {"strategy": name, "hash": spec_hash(plan["hypotheses"][name]),
                     "run_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    split = get_split(base, load_prices(base))
    E = plan["universes"]["exploration"]
    panel, rf, _, missing = load_universe(base, E, plan["universes"]["risk_free"],
                                          split.data_end + pd.Timedelta(days=1), synthetic)
    win = [(split.test_start, split.data_end)]
    rows = {}
    for n, (W, reb) in {**strategy_set(plan, panel, rf, [name]), **benchmark_set(E, panel)}.items():
        rows[n] = summarize_windows(n, run_windows(W, reb, panel, rf, win, plan["common"]), rf)
    fc = base.get("final_test_criteria") or {}
    s = rows[name]
    ok = (s.get("mean_return", -1) > fc.get("min_return", 0.0) and s.get("mean_sharpe", -9) > fc.get("min_sharpe", 0.0)
          and (s.get("survived", 0) == s.get("n_ok", 1) or not fc.get("must_survive", True)))
    L = ["# FINAL TEST (one-time, locked test set) - round 4\n"]
    if synthetic:
        L.append("> **SYNTHETIC DATA - NOT REAL RESULTS.**\n")
    L.append(f"Strategy **{name}** (registered round-4 hypothesis), selected from {sel['n_trials']} trials. "
             f"Test period {split.test_start.date()} -> {split.data_end.date()}. Fixed rule, no training. "
             f"Tickers not loaded: {', '.join(missing) or 'none'}.\n")
    L.append(f"**Pre-registered final-test check** (return > {fc.get('min_return', 0.0):+.0%}, Sharpe > "
             f"{fc.get('min_sharpe', 0.0)}, survives): **{'MET' if ok else 'NOT MET'}**. Two years is short: this can "
             "catch a broken strategy but cannot prove a good one.\n")
    L.append("| strategy | return | CAGR | Sharpe | max DD | survived |\n|---|---|---|---|---|---|")
    for n, r in rows.items():
        L.append(f"| {n} | {r.get('mean_return', float('nan')):+.1%} | {r.get('cagr', float('nan')):+.1%} | "
                 f"{r.get('mean_sharpe', float('nan')):.2f} | {r.get('mean_stitched_mdd', float('nan')):.1%} | "
                 f"{r.get('survived', 0)}/{r.get('n_ok', 0)} |")
    L.append("\nDo NOT change the strategy based on this result. The test set is now used up.\n")
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info("Wrote %s", md.name)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="use smoke-mode results/settings (pipeline testing)")
    args = ap.parse_args()

    base = prepare_cfg(load_base(), args.synthetic, args.smoke)
    setup_logging(base, "final_test_" + mode_name(args.synthetic, args.smoke))
    res_root, model_root = mode_dirs(base, args.synthetic, args.smoke)
    out_dir = res_root / "final_test"
    lock = out_dir / "LOCK.json"
    md = ROOT / ("FINAL_TEST.md" if not (args.synthetic or args.smoke)
                 else f"FINAL_TEST_{mode_name(args.synthetic, args.smoke).upper()}.md")

    if lock.exists():
        log.error("Final test already run (%s). Refusing to run again - see %s", lock, md.name)
        return 1
    sel_path = res_root / "selection.json"
    if not sel_path.exists():
        log.error("No selection.json - run run_all.py first.")
        return 1
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    if not sel.get("selected"):
        msg = ("# FINAL TEST - NOT RUN\n\nNo strategy passed validation + robustness, so the locked test set "
               "was **not** touched. It stays clean for a future research round.\n")
        md.write_text(msg, encoding="utf-8")
        log.info("No passing strategy. Test set left untouched. Wrote %s", md.name)
        return 0

    name = sel["selected"]
    if sel.get("kind") == "research":     # round-4 allocation rule
        return final_test_research(name, base, out_dir, lock, md, args.synthetic, sel)
    cfg = prepare_cfg(load_experiment(CONFIGS_DIR / f"{name}.yaml"), args.synthetic, args.smoke)
    if strategy_hash(cfg) != sel["hash"]:
        log.error("Config %s changed since validation (hash %s != %s). Refusing: re-run run_all.py.",
                  name, strategy_hash(cfg), sel["hash"])
        return 1

    # Write the lock BEFORE looking at any result, so a crash cannot be retried for a better number.
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(lock, {"strategy": name, "hash": sel["hash"], "run_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    prices = load_prices(cfg)
    split = get_split(cfg, prices)
    assets = build_assets(test_view(prices, split), cfg["features"])
    data_start = max(a.features.index[0] for a in assets.values())
    train_end = split.test_start - pd.tseries.offsets.BDay(int(cfg["walk_forward"].get("embargo_days", 5)))
    ts, te = split.test_start, split.data_end
    rate = cost_rate(cfg)
    initial, max_dd = float(cfg["capital"]["initial"]), float(cfg["survival"]["max_drawdown"])
    log.info("FINAL TEST of %s: train %s -> %s, test %s -> %s", name, data_start.date(), train_end.date(), ts.date(), te.date())

    rows, curves = [], {}
    if is_rule(cfg):   # a fixed rule: nothing to train, just trade the locked period once
        bt = evaluate_window(cfg, assets, ts, te)
        rows.append({"name": f"{name} (rule)", **bt.metrics})
        curves[f"{name} (rule)"] = bt.equity
        log.info("rule: return %+.1f%% sharpe %.2f mdd %.1f%% %s", 100 * bt.metrics["total_return"],
                 bt.metrics["sharpe"], 100 * bt.metrics["max_drawdown"], "DEAD" if bt.dead else "alive")
    for seed in ([] if is_rule(cfg) else cfg["agent"]["seeds"]):
        segs = [s for s in (make_segment(a, data_start, train_end, cfg) for a in assets.values()) if s is not None]
        model = train_agent(cfg, segs, rate, int(seed), model_root / name / "final" / f"seed{seed}")
        sleeves = []
        for a in assets.values():
            res = agent_positions(model, cfg, a, ts, te, rate)
            if res is None:
                continue
            _, pos = res
            sleeves.append(window_sleeve(a, ts, te, lambda d, p=pos: p))
        bt = portfolio_backtest(sleeves, rate, initial, max_dd)
        rows.append({"name": f"{name} seed {seed}", **bt.metrics})
        curves[f"{name} seed {seed}"] = bt.equity
        log.info("seed %s: return %+.1f%% sharpe %.2f mdd %.1f%% %s", seed, 100 * bt.metrics["total_return"],
                 bt.metrics["sharpe"], 100 * bt.metrics["max_drawdown"], "DEAD" if bt.dead else "alive")

    for bname, params in cfg["baselines"].items():
        sleeves = []
        for a in assets.values():
            fp = baseline_positions(bname, a.close, params)
            sl = window_sleeve(a, ts, te, lambda d, fp=fp: fp.reindex(d).fillna(0.0).values)
            if sl is not None:
                sleeves.append(sl)
        bt = portfolio_backtest(sleeves, rate, initial, max_dd)
        rows.append({"name": f"baseline:{bname}", **bt.metrics})
        curves[f"baseline:{bname}"] = bt.equity

    df = pd.DataFrame(rows).set_index("name")
    df.to_csv(out_dir / "final_test_metrics.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, eq in curves.items():
        ax.plot(eq, "--" if k.startswith("baseline") else "-", lw=1.1, label=k)
    ax.set_title(f"FINAL TEST (locked period) - {name}" + (" - SYNTHETIC" if args.synthetic else ""))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "final_test_equity.png", dpi=90)
    plt.close(fig)

    rl = df.loc[[i for i in df.index if not i.startswith("baseline")]]
    alive = int((~rl["dead"].astype(bool)).sum())
    L = ["# FINAL TEST (one-time, locked test set)\n"]
    if args.synthetic:
        L.append("> **SYNTHETIC DATA - NOT REAL RESULTS.**\n")
    L.append(f"Strategy: **{name}** (hash {sel['hash']}), selected from {sel['n_trials']} variants tried.  ")
    trained = "Fixed rule, no training." if is_rule(cfg) else f"Trained on {data_start.date()} -> {train_end.date()}."
    L.append(f"Test period: {ts.date()} -> {te.date()}. {trained}  ")
    L.append(f"Mean return {rl['total_return'].mean():+.1%}, mean Sharpe {rl['sharpe'].mean():.2f}, "
             f"survived {alive}/{len(rl)} run(s).\n")
    fc = cfg.get("final_test_criteria") or {}
    ok = (rl["total_return"].mean() > fc.get("min_return", 0.0) and rl["sharpe"].mean() > fc.get("min_sharpe", 0.0)
          and (alive == len(rl) or not fc.get("must_survive", True)))
    L.append(f"**Pre-registered final-test check** (written in base_config.yaml before the test was run: return > "
             f"{fc.get('min_return', 0.0):+.0%}, Sharpe > {fc.get('min_sharpe', 0.0)}, survives): "
             f"**{'MET' if ok else 'NOT MET'}**. Two years is short: a Sharpe estimate from 2 years has a standard "
             "error of about 0.7, so this can catch a broken strategy but cannot prove a good one.\n")
    L.append("Do NOT change the strategy based on this result. The test set is now used up; a new round needs new "
             "unseen data (e.g. wait and paper-trade forward).\n")
    cols = ["total_return", "cagr", "sharpe", "sortino", "max_drawdown", "annual_turnover", "dead", "death_date"]
    L.append("| run | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for i, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c in ("total_return", "cagr"):
                vals.append(f"{v:+.1%}")
            elif c == "max_drawdown":
                vals.append(f"{v:.1%}")
            elif c == "death_date":
                vals.append(str(v) if isinstance(v, str) else "-")
            elif c == "dead":
                vals.append("DEAD" if bool(v) else "alive")
            else:
                vals.append(f"{v:.2f}")
        L.append(f"| {i} | " + " | ".join(vals) + " |")
    L.append(f"\n![equity]({(out_dir / 'final_test_equity.png').relative_to(ROOT).as_posix()})\n")
    md.write_text("\n".join(L), encoding="utf-8")
    log.info("Wrote %s", md.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
