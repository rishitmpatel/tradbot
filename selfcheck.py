"""Sanity checks run at the start of every pipeline run (and by the tests).
If any fails, run_all.py stops: results from a broken engine are worthless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import Sleeve, cost_rate, portfolio_backtest, sleeve_returns, window_sleeve
from features import build_assets, check_causality
from trading_env import Segment, TradingEnv


def check_features_causal(prices: dict, cfg: dict) -> None:
    for t, df in prices.items():
        check_causality(df["close"], cfg["features"])


def check_engine_timing(asset, cfg: dict) -> None:
    """(1) Constant long = buy & hold minus one entry cost.
    (2) A cheating 'oracle' that knows tomorrow's return must look absurdly good;
        a shifted (non-cheating) version of it must not. This proves positions
        are applied to the NEXT day's return, not today's."""
    rate = cost_rate(cfg)
    idx = asset.features.index
    start, end = idx[len(idx) // 3], idx[len(idx) // 3 + 300]
    sl = window_sleeve(asset, start, end, lambda d: np.ones(len(d)))
    bt = portfolio_backtest([sl], rate, 100_000.0, None)
    expect = 100_000.0 * (1 - rate) * asset.close.loc[end] / asset.close.loc[start]
    assert abs(bt.equity.iloc[-1] - expect) / expect < 1e-9, "buy&hold accounting mismatch"

    oracle = window_sleeve(asset, start, end, lambda d: np.sign(asset.fwd_ret.loc[d].values))
    bt_o = portfolio_backtest([oracle], 0.0, 100_000.0, None)
    assert bt_o.metrics["sharpe"] > 5, "oracle not detected: engine timing is wrong"
    honest = window_sleeve(asset, start, end,
                           lambda d: np.sign(asset.fwd_ret.shift(1).reindex(d).fillna(0).values))
    bt_h = portfolio_backtest([honest], 0.0, 100_000.0, None)
    assert bt_h.metrics["sharpe"] < 3, "lagged signal looks like an oracle: possible leakage"


def check_env_matches_engine(asset, cfg: dict, seed: int = 0) -> None:
    """Random actions through the env must give the same P&L as the backtest engine."""
    rate = cost_rate(cfg)
    idx = asset.features.index
    sel = slice(len(idx) // 2, len(idx) // 2 + 200)
    seg = Segment(asset.features.values[sel], asset.fwd_ret.values[sel], idx[sel], first=0)
    env_cfg = dict(cfg["env"], obs_window=1)
    env = TradingEnv([seg], env_cfg, dict(cfg["reward"], terminate_on_death=False), rate, 0.99, train=False)
    env.reset(options={"segment": 0})
    rng = np.random.default_rng(seed)
    nets, poss = [], []
    while True:
        _, _, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
        nets.append(info["net"])
        poss.append(info["pos"])
        if term or trunc:
            break
    net_engine, _ = sleeve_returns(np.array(poss), seg.fwd_ret[: len(poss)], rate)
    assert np.allclose(nets, net_engine, atol=1e-12), "env P&L differs from backtest engine"


def check_survival_rule() -> None:
    d = pd.bdate_range("2020-01-01", periods=60)
    fwd = np.full(59, 0.01)
    fwd[20:30] = -0.03            # crash: ~26% fall
    sl = Sleeve("X", d[:-1], d[1:], np.ones(59), fwd)
    bt = portfolio_backtest([sl], 0.001, 100_000.0, 0.15)
    assert bt.dead, "survival rule did not trigger"
    k = bt.equity.index.get_loc(bt.death_date)
    assert np.allclose(bt.equity.iloc[k + 1:], bt.equity.iloc[k + 1]), "equity moved after death"
    assert bt.equity.iloc[-1] < bt.equity.cummax().max() * 0.85 + 1e-6


def run_all_checks(prices: dict, cfg: dict) -> list[str]:
    done = []
    check_features_causal(prices, cfg)
    done.append("features causal (truncation test)")
    assets = build_assets(prices, cfg["features"])
    a = next(iter(assets.values()))
    check_engine_timing(a, cfg)
    done.append("engine timing (buy&hold accounting, oracle detection)")
    check_env_matches_engine(a, cfg)
    done.append("env P&L == engine P&L")
    check_survival_rule()
    done.append("survival kill switch")
    return done
