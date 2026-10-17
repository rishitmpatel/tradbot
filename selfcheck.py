"""Sanity checks run at the start of every pipeline run (and by the tests).
If any fails, run_all.py stops: results from a broken engine are worthless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import Sleeve, cost_rate, portfolio_backtest, sleeve_returns, window_sleeve
from features import build_assets, check_causality


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
    """Random actions through the env must give the same P&L as the backtest engine,
    with this config's overlay, rebalance band, decision interval and reward settings."""
    from agents import build_env, make_segment

    rate = cost_rate(cfg)
    idx = asset.features.index
    seg = make_segment(asset, idx[len(idx) // 2], idx[len(idx) // 2 + 200], cfg)
    c = dict(cfg, reward=dict(cfg["reward"], terminate_on_death=False))
    env = build_env([seg], c, rate, train=False)
    env.reset(options={"segment": 0})
    rng = np.random.default_rng(seed)
    nets, poss = [], []
    while True:
        _, reward, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
        assert np.isfinite(reward), "non-finite reward"
        nets.extend(info["nets"])
        poss.extend(info["positions"])
        if term or trunc:
            break
    start = max(seg.first, env.window - 1)
    assert len(poss) == seg.last_decision - start + 1, "env skipped or repeated days"
    assert max(abs(p) for p in poss) <= 1.0 + 1e-12, "position above 100% (leverage)"
    net_engine, _ = sleeve_returns(np.array(poss), seg.fwd_ret[start: start + len(poss)], rate)
    assert np.allclose(nets, net_engine, atol=1e-12), "env P&L differs from backtest engine"


def check_overlay_causal(close: pd.Series, cfg: dict, n_checks: int = 5) -> None:
    """The rule's base position and benchmark on day t must not change when future prices are removed."""
    from overlay import base_position, reference_position

    full_b, full_r = base_position(close, cfg), reference_position(close, cfg)
    rng = np.random.default_rng(1)
    for i in rng.choice(np.arange(300, len(close) - 1), size=n_checks, replace=False):
        tr = close.iloc[: i + 1]
        assert np.isclose(full_b.iloc[i], base_position(tr, cfg).iloc[-1], equal_nan=True), \
            f"LOOKAHEAD in overlay base position at {close.index[i].date()}"
        if full_r is not None:
            assert np.isclose(full_r.iloc[i], reference_position(tr, cfg).iloc[-1], equal_nan=True), \
                f"LOOKAHEAD in reward benchmark at {close.index[i].date()}"


OVERLAY_TEST = {"env": {"base_signal": "momentum", "vol_target": 0.12, "actions": [0, 0.5, 1],
                        "rebalance_band": 0.1, "decision_interval": 5, "obs_window": 3},
                "reward": {"relative_to": "base", "type": "differential_sharpe"}}


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
    from config_utils import deep_merge
    ocfg = deep_merge(cfg, OVERLAY_TEST)
    check_overlay_causal(a.close, ocfg)
    check_env_matches_engine(a, ocfg)
    done.append("overlay: causal base/benchmark, env == engine with band + weekly decisions")
    check_survival_rule()
    done.append("survival kill switch")
    return done
