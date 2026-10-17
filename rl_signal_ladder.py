"""Learning experiment: how strong must a signal be before PPO can find it?

Synthetic prices only (this says nothing about real markets). One asset whose daily
return has a PLANTED, known amount of predictability:

    r_t = mu + phi * (r_{t-1} - mu) + noise        noise ~ 1% a day, mu ~ 7.5% a year

phi = 0 means tomorrow is unpredictable from today; phi = 0.2 is a strong, easy edge.
For each phi we train PPO exactly like the main pipeline (same features, network,
60k steps, 8 years of training data), then trade 3 unseen years and compare with:
  * buy & hold
  * the ideal rule for this process: long when the expected next-day return
    mu + phi * (r_today - mu) is positive, else flat (it KNOWS the planted structure;
    PPO has to discover it from scratch)
Run with and without trading costs.

    python rl_signal_ladder.py              (~5-15 min on a laptop; writes RL_SIGNAL_LADDER.md)

How to read it: if PPO only finds signals far stronger than real markets show (see the
lag-1 autocorrelation column in RESULTS.md's data-quality table), RL on these features
cannot be expected to add anything - which is what rounds 1-2 found.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from config_utils import ROOT, deep_merge, load_base, resolve_n_jobs

PHIS = [0.0, 0.05, 0.10, 0.20]
COSTS = [0.0, 0.0015]
SEEDS = [0, 1, 2]


def make_prices(phi: float, n_years: int = 13, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)               # same noise for every phi -> fair comparison
    n = n_years * 252
    mu, eps = 0.0003, rng.normal(0, 0.01, n)
    r = np.empty(n)
    prev = mu
    for i in range(n):
        r[i] = mu + phi * (prev - mu) + eps[i]
        prev = r[i]
    idx = pd.bdate_range("2000-01-03", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(np.log1p(r))), index=idx)
    return pd.DataFrame({"close": close})


def _job(args) -> dict:
    phi, cost, seed, steps = args
    import torch

    from agents import agent_positions, make_segment, train_agent
    from backtest import portfolio_backtest, window_sleeve
    from features import build_assets

    torch.set_num_threads(1)
    cfg = deep_merge(load_base(), {
        "costs": {"commission": cost, "slippage": 0.0},
        "agent": {"algo": "PPO", "total_timesteps": steps},
        "env": {"actions": [0, 1]},
        "reward": {"drawdown_increase_penalty": 0.0, "death_penalty": 0.0, "terminate_on_death": False},
    })
    a = build_assets({"X": make_prices(phi)}, cfg["features"])["X"]
    idx = a.features.index
    split = idx[int(len(idx) * 0.73)]
    train_end, test_start, end = idx[idx < split][-1], split, idx[-1]
    seg = make_segment(a, idx[0], train_end, cfg)
    model = train_agent(cfg, [seg], cost, seed)
    _, pos = agent_positions(model, cfg, a, test_start, end, cost)

    def sharpe(pos_fn):
        sl = window_sleeve(a, test_start, end, pos_fn)
        return portfolio_backtest([sl], cost, 100_000.0, None).metrics["sharpe"]

    return {"phi": phi, "cost_bps": cost * 1e4, "seed": seed,
            "ppo": sharpe(lambda d: pos),
            "ppo_exposure": float(np.mean(pos)),
            "buy_hold": sharpe(lambda d: np.ones(len(d))),
            "ideal_rule": sharpe(lambda d: ((0.0003 + phi * (a.close.pct_change().reindex(d) - 0.0003)) > 0)
                                 .astype(float).values)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    jobs = [(p, c, s, args.steps) for p in PHIS for c in COSTS for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(resolve_n_jobs(args.n_jobs), mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_job, jobs))
    df = pd.DataFrame(rows)
    g = df.groupby(["cost_bps", "phi"]).agg(ppo_sharpe=("ppo", "mean"), ppo_worst_seed=("ppo", "min"),
                                             ppo_exposure=("ppo_exposure", "mean"),
                                             buy_hold=("buy_hold", "mean"), ideal_rule=("ideal_rule", "mean"))
    g = g.round(2).reset_index()
    lines = ["# RL signal ladder (SYNTHETIC data - a learning experiment, not market results)\n",
             "Out-of-sample Sharpe over 3 unseen years after training PPO on 8 years "
             f"({args.steps} steps, {len(SEEDS)} seeds). phi = planted lag-1 autocorrelation of daily returns.\n",
             "| costs (bps per trade) | phi | PPO Sharpe (mean) | PPO worst seed | PPO avg exposure | "
             "buy & hold | ideal rule (knows the structure) |", "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r.cost_bps:.0f} | {r.phi:.2f} | {r.ppo_sharpe:.2f} | {r.ppo_worst_seed:.2f} | "
                     f"{r.ppo_exposure:.2f} | {r.buy_hold:.2f} | {r.ideal_rule:.2f} |")
    lines.append(f"\nRuntime {time.time() - t0:.0f}s. Compare phi with the real lag-1 autocorrelations in "
                 "RESULTS.md (data-quality table).")
    out = ROOT / "RL_SIGNAL_LADDER.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
