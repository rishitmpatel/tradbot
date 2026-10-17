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


ALLOC_SPECS = {
    "ew": {"rule": "equal_weight", "params": {}},
    "iv": {"rule": "inverse_vol", "params": {"window": 60}},
    "tr": {"rule": "trend", "params": {"lookback": 252, "window": 60}},
    "xs": {"rule": "xs_momentum", "params": {"lookback": 252, "skip": 21, "top_frac": 0.34, "window": 60}},
    "pv": {"rule": "per_asset_vol_target", "params": {"target": 0.12, "window": 20, "band": 0.02}},
    "en": {"rule": "ensemble", "params": {"members": ["iv", "tr", "xs"]}},
    "lv": {"rule": "vol_target_portfolio", "params": {"base": "en", "target": 0.10, "max_gross": 1.5}},
}


def check_allocation_rules(panel: pd.DataFrame, n_checks: int = 3) -> None:
    """Round-4 rules: weights on day t identical when later prices are deleted (no lookahead),
    long only, total <= 1 (or <= max_gross for the leveraged rule)."""
    from allocation import build_weights

    rf = pd.Series(0.0001, index=panel.index)
    full = {n: build_weights(n, sp, ALLOC_SPECS, panel, rf, None, {}) for n, sp in ALLOC_SPECS.items()}
    rng = np.random.default_rng(3)
    for i in rng.choice(np.arange(300, len(panel) - 1), size=n_checks, replace=False):
        tp, trf = panel.iloc[: i + 1], rf.iloc[: i + 1]
        for n, sp in ALLOC_SPECS.items():
            t = build_weights(n, sp, ALLOC_SPECS, tp, trf, None, {})
            assert np.allclose(full[n].iloc[i].values, t.iloc[-1].values, atol=1e-9), \
                f"LOOKAHEAD in allocation rule {sp['rule']} at {panel.index[i].date()}"
    for n, w in full.items():
        cap = ALLOC_SPECS[n]["params"].get("max_gross", 1.0)
        assert (w.values >= -1e-12).all(), f"{n}: short position"
        assert (w.sum(axis=1) <= cap + 1e-9).all(), f"{n}: exposure above {cap}"


def check_tracker_matches_engine(panel: pd.DataFrame) -> None:
    """The forward paper tracker must reproduce the backtest engine's equity exactly."""
    import tempfile
    from pathlib import Path

    from allocation import build_weights, rebalance_days
    from backtest import weights_backtest
    from paper_trade import update_tracker

    plan = {"hypotheses": {"lv": ALLOC_SPECS["lv"], **ALLOC_SPECS},
            "common": {"costs_bps": 15, "borrow_spread": 0.01, "max_drawdown_kill": 0.15},
            "universes": {"exploration": {"benchmarks": {"equity": panel.columns[0], "bonds": panel.columns[1]}}}}
    rf = pd.Series(0.0001, index=panel.index)
    days = panel.index[len(panel) // 2: len(panel) // 2 + 260]
    W = build_weights("lv", ALLOC_SPECS["lv"], ALLOC_SPECS, panel, rf, None, {})
    reb = rebalance_days(panel.index)
    dec, real = days[:-1], days[1:]
    tradable = np.repeat(pd.Series(reb, index=panel.index).loc[dec].values[:, None], panel.shape[1], axis=1)
    tradable[0] = True
    fwd = (panel.shift(-1) / panel - 1.0).loc[dec].fillna(0.0).values
    bt = weights_backtest(dec, real, W.loc[dec].values, fwd, tradable, 0.0015, 1e5, 0.15, rf=rf.loc[dec].values,
                          borrow_spread=0.01)
    with tempfile.TemporaryDirectory() as d:
        update_tracker(panel.loc[: days[-1]], rf, plan, ["lv"], Path(d), days[0])
        log = pd.read_csv(Path(d) / "lv.csv")
    # the engine applies trading cost at the decision; the tracker logs equity after it
    assert np.isclose(log["equity"].iloc[-1], bt.equity.iloc[-1], rtol=1e-6), \
        f"paper tracker {log['equity'].iloc[-1]:.2f} != engine {bt.equity.iloc[-1]:.2f}"


FRENCH_SAMPLE = """Header text line
  Average Value Weighted Returns -- Monthly
,SMALL LoPRIOR,ME1 PRIOR2,SMALL HiPRIOR,BIG LoPRIOR,ME2 PRIOR2,BIG HiPRIOR
192701,   2.12,   0.53,   1.10,  -0.33,   0.55,   1.02
192702, -99.99,   4.68,   5.73,   3.69,   4.35,   4.76

  Annual
,SMALL LoPRIOR,ME1 PRIOR2,SMALL HiPRIOR,BIG LoPRIOR,ME2 PRIOR2,BIG HiPRIOR
  1927,   9.99,   9.99,   9.99,   9.99,   9.99,   9.99
"""


def check_round5() -> None:
    """Round 5: Ken French parser, turn-of-month calendar, no lookahead in monthly positions
    (future-poison test), monthly engine accounting + kill switch, Indian tax simulator."""
    from allocation import tom_position_mask
    from factor_data import parse_french_csv, synthetic_market
    from factor_research import load_plan, positions, run_monthly, sleeve_returns
    from india_costs import simulate

    df = parse_french_csv(FRENCH_SAMPLE)
    assert len(df) == 2 and np.isnan(df.iloc[1, 0]) and np.isclose(df["BIG HiPRIOR"].iloc[0], 0.0102), "French parser"
    days = pd.bdate_range("2024-10-01", "2024-11-08")
    on = pd.Series(tom_position_mask(days), index=days)
    expect = pd.DatetimeIndex(["2024-10-01", "2024-10-02", "2024-10-30", "2024-10-31", "2024-11-01", "2024-11-04"])
    assert list(on[on].index) == list(expect), f"turn-of-month calendar wrong: {list(on[on].index)}"

    plan = load_plan()
    common = plan["common"]
    data = synthetic_market(1, "1990-01-01", "2010-12-31")
    rng = np.random.default_rng(4)
    for spec in plan["hypotheses"].values():
        if spec.get("turn_of_month"):
            continue                                      # calendar-only signal, checked above
        full = positions(spec, common, data, sleeve_returns(spec, common, data))
        for i in rng.choice(np.arange(30, len(data["monthly"]) - 1), size=3, replace=False):
            t = data["monthly"].index[i]
            bad = {"monthly": data["monthly"].copy(), "daily": data["daily"]}
            bad["monthly"].loc[bad["monthly"].index >= t, ["market", "big_momentum", "big_value"]] = 9.0
            pos = positions(spec, common, bad, sleeve_returns(spec, common, bad))
            assert np.isclose(full[t], pos[t], equal_nan=True), f"LOOKAHEAD in round-5 position at {t.date()}"

    idx = pd.date_range("2001-01-31", periods=24, freq="ME")
    sl = pd.Series(0.01, index=idx)
    rf = pd.Series(0.0, index=idx)
    r, w = run_monthly(sl, rf, pd.Series(1.0, index=idx), 0.0, None, "2001-01-31", idx[-1])
    assert np.isclose((1 + r).prod(), 1.01 ** 24), "monthly engine accounting"
    crash = pd.Series([-0.06] * 12 + [0.02] * 12, index=idx)
    r, w = run_monthly(crash, rf, pd.Series(1.0, index=idx), 0.001, 0.15, "2001-01-31", idx[-1])
    assert w[0]["dead"] and (r[r.index.year == 2001].iloc[3:] == 0).all(), "monthly kill switch"

    m = pd.date_range("2020-04-30", periods=24, freq="ME")
    t = simulate(pd.DataFrame({"market": 1.0, "cash": 0.0}, index=m), pd.DataFrame({"market": 0.01, "cash": 0.0}, index=m),
                 1e6, "india_listed")
    gain = 1e6 * (1.01 ** 24 - 1)
    assert abs(t["tax_paid"] - 0.125 * (gain - 125_000) * 1.04) < 0.01 * gain, "Indian LTCG tax simulator"


def check_r5_sleeve_causal(panel: pd.DataFrame) -> None:
    from allocation import build_weights
    c = list(panel.columns)
    spec = {"x": {"rule": "r5_sleeve", "params": {"mix": {c[0]: 0.5, c[1]: 0.5}, "cash": c[2], "trend_signal": c[0],
                                                  "tom": True, "vol_target": 0.10}}}
    rf = pd.Series(0.0001, index=panel.index)
    full = build_weights("x", spec["x"], spec, panel, rf, None, {})
    for i in (400, 700, 900):
        tr = build_weights("x", spec["x"], spec, panel.iloc[: i + 1], rf.iloc[: i + 1], None, {})
        assert np.allclose(full.iloc[i].values, tr.iloc[-1].values, atol=1e-9), "LOOKAHEAD in r5_sleeve (paper tracker rule)"
    assert (full.sum(axis=1) <= 1 + 1e-9).all() and (full.values >= -1e-12).all(), "r5_sleeve leverage/short"


def check_round6() -> None:
    """Round 6: overnight/intraday return timing (future-poison), stress fills, open-price reliability check,
    daily kill switch, momentum guard decided at the previous month end, Indian tax arithmetic."""
    import overnight_research as o6
    from factor_data import synthetic_market

    plan = o6.load_plan()
    th = plan["overnight"]["reliability"]
    df = o6.synthetic_ohlc(3, "2015-01-01", "2016-12-31")
    on = o6.day_returns(df, "overnight")
    t1 = df.index[1]
    assert np.isclose(on[t1], df["open"].iloc[1] / df["close"].iloc[0] - 1), "overnight return timing"
    assert np.isclose(o6.day_returns(df, "intraday")[t1], df["close"].iloc[1] / df["open"].iloc[1] - 1), "intraday timing"
    for i in (50, 200, 400):
        bad = df.copy()
        bad.iloc[i + 1:] *= 3.0
        r = o6.day_returns(bad, "overnight")
        assert np.allclose(on.iloc[:i], r.iloc[:i]), "LOOKAHEAD in overnight returns"
    st = o6.day_returns(df, "overnight", 0.0, -0.002)
    assert (st <= on + 1e-12).all(), "stress fill better than the open"
    hit = on <= -0.002
    assert hit.any() and np.allclose(st[hit], (df["low"] / df["close"].shift(1) - 1)[hit.index][hit]), "stress fill != low"
    assert o6.open_quality(df, th)["reliable"], "good open prices flagged unreliable"
    fake = o6.synthetic_ohlc(3, "2015-01-01", "2016-12-31", fake_open=True)
    assert not o6.open_quality(fake, th)["reliable"], "fake open prices (= previous close) NOT flagged"
    broken = df.copy()
    broken.loc[broken.index[::10], "open"] = broken["high"] * 1.05
    assert not o6.open_quality(broken, th)["reliable"], "open outside the day's range NOT flagged"
    split = df.copy()
    split.iloc[300:] /= 10.0                                                   # an unadjusted 1:10 split
    assert not o6.open_quality(split, th)["reliable"], "unadjusted split NOT flagged"

    idx = pd.bdate_range("2001-01-01", "2001-12-31")
    r = pd.Series(0.001, index=idx)
    r.iloc[100] = -0.20
    out, wins = o6.run_daily(r, 0.15, idx[0], idx[-1])
    assert wins[0]["dead"] and (out.iloc[101:] == 0).all() and np.isclose(out.iloc[100], -0.20), "daily kill switch"

    m = synthetic_market(2, "1990-01-01", "2010-12-31")["monthly"]
    spec = plan["hypotheses"]["R6_H2_momentum_crash_guard"]
    g = o6.guard_state(spec, m)
    for i in (60, 120, 200):
        t = m.index[i]
        bad = m.copy()
        bad.loc[bad.index >= t, ["market", "big_momentum"]] = -0.5
        assert np.isclose(g[t], o6.guard_state(spec, bad)[t], equal_nan=True), f"LOOKAHEAD in momentum guard at {t.date()}"

    fy = pd.bdate_range("2020-04-01", "2022-03-31")
    rr = pd.Series(np.where(fy < "2021-04-01", np.log(0.9) / 252, np.log(1.1) / 252), index=fy)
    rr = np.exp(rr) - 1
    assert o6.after_tax_cagr(rr, 1e6, 0.2, 0.04)["tax_share"] == 0.0, "loss carry-forward"
    c = plan["overnight"]["costs"]["india"]
    assert 3.0 < o6.india_round_trip(c, 1e6) * 1e4 < 6.0, "India round-trip cost out of range"
    d = pd.bdate_range("2010-01-01", "2020-12-31")
    px = pd.Series(100 * 1.12 ** (np.arange(len(d)) / 252), index=d)
    h, w = o6.tax_harvest_sim(px, 1e5, False, c), o6.tax_harvest_sim(px, 1e5, True, c)
    gain = 1e5 * (px.iloc[-1] / px.iloc[0]) - 1e5
    assert abs(h["tax_share"] * 1e5 - 0.125 * (gain - 125_000) * 1.04) < 0.02 * gain, "buy & hold LTCG"
    assert w["final"] > h["final"] and w["harvests"] >= 9, "tax harvesting does not help on a rising market"
    flat = pd.Series(100.0, index=d)
    assert o6.tax_harvest_sim(flat, 1e6, True, c)["harvests"] == 0, "harvested without a gain"


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
    from rules import close_panel
    panel = close_panel(assets)
    check_allocation_rules(panel)
    check_tracker_matches_engine(panel.dropna())
    done.append("round-4 allocation rules causal/long-only/capped; paper tracker == engine")
    check_round5()
    check_r5_sleeve_causal(panel.dropna())
    done.append("round-5: French parser, turn-of-month calendar, no lookahead (future-poison), engine, kill switch, tax sim")
    check_round6()
    done.append("round-6: overnight timing (future-poison), stress fills, fake-open detection, daily kill switch, "
                "momentum guard causal, loss carry-forward, tax harvesting")
    return done
