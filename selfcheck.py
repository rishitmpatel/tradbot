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


def check_weights_engine(asset, cfg: dict) -> None:
    """Portfolio-weights engine: (1) 100% in one asset == buy & hold minus one entry cost,
    identical to the sleeve engine; (2) two assets, constant 50/50 rebalanced daily, matches
    a hand calculation including rebalancing costs; (3) kill switch freezes equity."""
    from backtest import weights_backtest

    rate = cost_rate(cfg)
    idx = asset.features.index
    start, end = idx[len(idx) // 3], idx[len(idx) // 3 + 250]
    days = idx[(idx >= start) & (idx <= end)]
    dec, real = days[:-1], days[1:]
    fwd = asset.fwd_ret.loc[dec].values[:, None]
    bt_w = weights_backtest(dec, real, np.ones((len(dec), 1)), fwd, np.ones((len(dec), 1), bool), rate, 1e5, None)
    sl = window_sleeve(asset, start, end, lambda d: np.ones(len(d)))
    bt_s = portfolio_backtest([sl], rate, 1e5, None)
    assert np.isclose(bt_w.equity.iloc[-1], bt_s.equity.iloc[-1], rtol=1e-10), "weights engine != sleeve engine"

    r = np.array([[0.01, -0.02], [0.03, 0.01], [-0.01, 0.0]])
    d = pd.bdate_range("2021-01-04", periods=4)
    bt2 = weights_backtest(d[:-1], d[1:], np.full((3, 2), 0.5), r, np.ones((3, 2), bool), 0.001, 1.0, None)
    eq, held = 1.0, np.zeros(2)
    for i in range(3):
        eq *= (1 - np.abs(0.5 - held).sum() * 0.001) * (1 + 0.5 * r[i].sum())
        held = 0.5 * (1 + r[i]) / (1 + 0.5 * r[i].sum())
    assert np.isclose(bt2.equity.iloc[-1], eq, rtol=1e-12), "weights engine rebalancing cost mismatch"

    crash = np.r_[np.full(10, 0.01), np.full(10, -0.03), np.full(9, 0.02)][:, None]
    d3 = pd.bdate_range("2021-01-04", periods=30)
    bt3 = weights_backtest(d3[:-1], d3[1:], np.ones((29, 1)), crash, np.ones((29, 1), bool), 0.001, 1e5, 0.15)
    assert bt3.dead and bt3.equity[bt3.equity.index > bt3.death_date].nunique() == 1, "weights kill switch broken"


def check_portfolio_rule_causal(assets: dict, n_checks: int = 4) -> None:
    """Weights on day t must not change when prices after t are removed."""
    from rules import close_panel, portfolio_vol_target_weights

    panel = close_panel(assets)
    full = portfolio_vol_target_weights(panel)
    rng = np.random.default_rng(2)
    for i in rng.choice(np.arange(200, len(panel) - 1), size=n_checks, replace=False):
        trunc = portfolio_vol_target_weights(panel.iloc[: i + 1])
        assert np.allclose(full.iloc[i].values, trunc.iloc[-1].values, atol=1e-9), \
            f"LOOKAHEAD in portfolio rule weights at {panel.index[i].date()}"
    assert (full.values >= -1e-12).all() and (full.sum(axis=1) <= 1.0 + 1e-9).all(), "portfolio rule leverage/short"


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
    check_weights_engine(a, cfg)
    check_portfolio_rule_causal(assets)
    done.append("portfolio engine accounting + portfolio rule causal, no leverage")
    return done
