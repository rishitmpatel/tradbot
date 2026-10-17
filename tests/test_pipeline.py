"""Run with:  python -m pytest -q
All tests use synthetic data (no internet needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import Sleeve, compute_metrics, cost_rate, portfolio_backtest, sleeve_returns  # noqa: E402
from config_utils import deep_merge, list_experiment_files, load_base, load_experiment, strategy_hash  # noqa: E402
from data import dev_view, get_split, load_prices  # noqa: E402
from evaluation import expected_max_sharpe, penalize  # noqa: E402
from features import build_assets, check_causality  # noqa: E402
from selfcheck import run_all_checks  # noqa: E402
from walkforward import load_dev_assets, make_folds  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return deep_merge(load_base(), {"data": {"source": "synthetic"}})


@pytest.fixture(scope="module")
def prices(cfg):
    return load_prices(cfg)


def test_all_configs_load():
    """Active AND archived configs must still load (archived ones still count as trials)."""
    from config_utils import CONFIGS_DIR
    files = list_experiment_files() + sorted(CONFIGS_DIR.glob("round*/*.yaml"))
    assert len(files) >= 22
    names = [load_experiment(f)["name"] for f in files]
    assert len(set(names)) == len(names)


def test_self_checks(prices, cfg):
    assert len(run_all_checks(prices, cfg)) == 8


def test_lookahead_feature_is_caught(prices, cfg):
    close = next(iter(prices.values()))["close"]
    import features as F
    orig = F.raw_feature

    def leaky(c, name):
        return c.shift(-1) / c - 1 if name == "logret_1" else orig(c, name)
    F.raw_feature = leaky
    try:
        with pytest.raises(AssertionError):
            check_causality(close, cfg["features"])
    finally:
        F.raw_feature = orig


def test_dev_view_excludes_test(prices, cfg):
    split = get_split(cfg, prices)
    for df in dev_view(prices, split).values():
        assert df.index.max() < split.test_start
    assets, split = load_dev_assets(cfg)
    folds = make_folds(cfg, assets, split)
    assert len(folds) >= 5
    for a, b in zip(folds, folds[1:]):
        assert pd.Timestamp(a.val_end) < pd.Timestamp(b.val_start)
    for f in folds:
        assert pd.Timestamp(f.train_end) < pd.Timestamp(f.val_start)
        assert pd.Timestamp(f.val_end) < split.test_start


def test_costs_reduce_returns():
    pos = np.array([0, 1, 0, 1, 0, 1], float)
    r = np.full(6, 0.001)
    net0, _ = sleeve_returns(pos, r, 0.0)
    net1, traded = sleeve_returns(pos, r, 0.0015)
    assert traded.sum() == 5
    assert (net1 <= net0).all() and net1.sum() < net0.sum()


def test_survival_freezes_equity():
    d = pd.bdate_range("2021-01-01", periods=40)
    fwd = np.r_[np.full(10, 0.01), np.full(10, -0.03), np.full(19, 0.02)]
    bt = portfolio_backtest([Sleeve("X", d[:-1], d[1:], np.ones(39), fwd)], 0.001, 100_000, 0.15)
    assert bt.dead
    after = bt.equity[bt.equity.index > bt.death_date]
    assert after.nunique() == 1  # no more P&L after death, profits kept
    alive = portfolio_backtest([Sleeve("X", d[:-1], d[1:], np.ones(39), fwd)], 0.001, 100_000, 0.50)
    assert not alive.dead and alive.equity.iloc[-1] > bt.equity.iloc[-1]


def test_multicalendar_portfolio():
    d1 = pd.bdate_range("2021-01-01", periods=30)
    d2 = d1[::2]
    s1 = Sleeve("A", d1[:-1], d1[1:], np.ones(29), np.full(29, 0.001))
    s2 = Sleeve("B", d2[:-1], d2[1:], np.zeros(len(d2) - 1), np.full(len(d2) - 1, 0.05))
    bt = portfolio_backtest([s1, s2], 0.0, 100.0, None)
    assert abs(bt.equity.iloc[-1] - (50 * 1.001 ** 29 + 50)) < 1e-9


def test_metrics_basic():
    idx = pd.bdate_range("2020-01-01", periods=505)
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, len(idx)), index=idx)
    m = compute_metrics(r)
    assert -1 < m["max_drawdown"] <= 1 and np.isfinite(m["sharpe"]) and np.isfinite(m["sortino"])


def test_multiple_testing_penalty_grows():
    assert expected_max_sharpe(1, 0.5) == 0
    assert expected_max_sharpe(30, 0.5) > expected_max_sharpe(5, 0.5) > 0
    s = {"mean_sharpe": 1.5, "n_days": 2000, "years": 8.0, "skew": 0.0, "kurt": 3.0}
    penalize(s, 30, [])
    assert s["penalized_sharpe"] < 1.5


def test_hash_ignores_runtime(cfg):
    a = strategy_hash(cfg)
    assert strategy_hash(deep_merge(cfg, {"runtime": {"n_jobs": 7}})) == a
    assert strategy_hash(deep_merge(cfg, {"reward": {"risk_penalty": 1.23}})) != a


def test_train_and_eval_tiny(cfg, tmp_path):
    """One very short PPO + DQN training and evaluation through the real code path."""
    from walkforward import run_experiment, summarize
    for algo in ("PPO", "DQN"):
        c = deep_merge(cfg, {"name": f"t_{algo}", "agent": {"algo": algo, "total_timesteps": 600, "seeds": [0],
                                                            "ppo": {"n_steps": 128, "batch_size": 32, "n_epochs": 1},
                                                            "dqn": {"learning_starts": 100}},
                             "walk_forward": {"max_folds": 1}})
        assets, split = load_dev_assets(c)
        folds = make_folds(c, assets, split)
        runs = run_experiment(c, tmp_path / algo / "runs", tmp_path / algo / "models", folds, n_jobs=1)
        assert runs[0]["status"] == "ok", runs[0].get("traceback")
        s = summarize(algo, runs)
        assert s["n_ok"] == 1 and "mean_sharpe" in s
        # resume: second call must not retrain
        runs2 = run_experiment(c, tmp_path / algo / "runs", tmp_path / algo / "models", folds, n_jobs=1)
        assert runs2[0]["seconds"] == runs[0]["seconds"]


# ---------------------------------------------------------------- round 2
def _seg(cfg, prices, extra):
    from agents import make_segment
    c = deep_merge(cfg, extra)
    a = next(iter(build_assets(prices, c["features"]).values()))
    idx = a.features.index
    return c, make_segment(a, idx[400], idx[700], c)


def test_overlay_defaults_are_round1(cfg, prices):
    _, seg = _seg(cfg, prices, {})
    assert np.all(seg.base == 1.0) and seg.ref is None


def test_relative_reward_is_zero_when_agent_copies_rule(cfg, prices):
    """Reward = beat the rule. Holding exactly the rule (multiplier 1) must score 0 every day."""
    from agents import build_env
    c, seg = _seg(cfg, prices, {"env": {"base_signal": "momentum", "vol_target": 0.12, "actions": [0, 1],
                                        "rebalance_band": 0.1},
                                "reward": {"relative_to": "base", "drawdown_increase_penalty": 0.0,
                                           "death_penalty": 0.0, "terminate_on_death": False}})
    env = build_env([seg], c, cost_rate(c), train=False)
    env.reset(options={"segment": 0})
    rewards = []
    while True:
        _, r, term, trunc, _ = env.step(1)
        rewards.append(r)
        if term or trunc:
            break
    assert np.allclose(rewards, 0.0, atol=1e-9)


def test_cash_is_not_free_with_relative_reward(cfg, prices):
    """Round-1 lesson: with a benchmark-relative reward, cash scores the negative of the market."""
    from agents import build_env
    c, seg = _seg(cfg, prices, {"env": {"actions": [0, 1]},
                                "reward": {"relative_to": "buy_and_hold", "drawdown_increase_penalty": 0.0,
                                           "death_penalty": 0.0, "terminate_on_death": False, "scale": 1.0}})
    env = build_env([seg], c, cost_rate(c), train=False)
    env.reset(options={"segment": 0})
    _, r, _, _, _ = env.step(0)   # stay in cash on day 1
    t0 = env.t - 1
    expected = -np.log1p((1 - cost_rate(c)) * (1 + seg.fwd_ret[t0]) - 1)
    assert np.isclose(r, expected)


def test_decision_interval_holds_position(cfg, prices):
    from agents import build_env
    from trading_env import rollout_positions

    class Alternate:
        k = 0

        def predict(self, obs, deterministic=True):
            self.k += 1
            return self.k % 2, None
    c, seg = _seg(cfg, prices, {"env": {"actions": [0, 1], "decision_interval": 5}})
    env = build_env([seg], c, cost_rate(c), train=False)
    pos = rollout_positions(Alternate(), env, 0)
    assert len(pos) == seg.last_decision - seg.first + 1
    blocks = pos[: len(pos) // 5 * 5].reshape(-1, 5)
    assert np.all(blocks == blocks[:, :1])            # constant within each 5-day block
    assert np.any(blocks[:-1, 0] != blocks[1:, 0])    # but it does change between blocks


def test_differential_sharpe_reward_sign(cfg, prices):
    from trading_env import Segment, TradingEnv
    n = 60
    seg = Segment(np.zeros((n, 1)), np.full(n, 0.002), pd.bdate_range("2020-01-01", periods=n))
    env = TradingEnv([seg], {"actions": [0, 1]}, {"type": "differential_sharpe", "scale": 1.0}, 0.0, 0.5, train=False)
    env.reset(options={"segment": 0})
    _, r_long, *_ = env.step(1)
    assert np.isfinite(r_long) and r_long > 0


def test_band_reduces_trades():
    from baselines import apply_band
    raw = 0.5 + 0.03 * np.sin(np.arange(200))
    assert np.abs(np.diff(apply_band(raw, 0.1))).sum() < np.abs(np.diff(raw)).sum() / 10
    assert apply_band(np.array([0.5, 0.0, 0.52]), 0.1)[1] == 0.0   # exits always execute


def test_relative_checks():
    from evaluation import low_risk_match, relative_checks
    pc = load_base()["pass_criteria"]
    bench = {"mean_sharpe": 1.0, "mean_stitched_mdd": 0.30, "survival_rate": 0.5}
    s = {"mean_sharpe": 0.95, "mean_stitched_mdd": 0.12, "survival_rate": 0.9}
    ok, why = relative_checks(s, bench, {"mean_sharpe": 0.97}, pc)
    assert not ok and len(why) == 2
    assert low_risk_match(s, bench, pc) is True
    ok, _ = relative_checks({**s, "mean_sharpe": 1.2}, bench, {"mean_sharpe": 1.1}, pc)
    assert ok


# ---------------------------------------------------------------- round 3
def _rule_cfg(cfg, name, params, neighborhood=None):
    return deep_merge(cfg, {"name": f"t_{name}", "kind": "rule",
                            "rule": {"name": name, "params": params, "neighborhood": neighborhood or []}})


def test_rule_matches_baseline(cfg, tmp_path):
    """A promoted rule must give exactly the baseline's numbers (same rule, two code paths)."""
    from rules import run_rule_strategy
    from walkforward import run_baseline, summarize
    assets, split = load_dev_assets(cfg)
    folds = make_folds(cfg, assets, split)
    c = _rule_cfg(cfg, "vol_target", cfg["baselines"]["vol_target"])
    a = summarize("rule", run_rule_strategy(c, assets, folds, tmp_path / "r"))
    b = summarize("base", run_baseline(cfg, "vol_target", folds, tmp_path / "b", assets))
    assert np.isclose(a["mean_sharpe"], b["mean_sharpe"]) and np.isclose(a["mean_return"], b["mean_return"])


def test_portfolio_rule_no_leverage_and_costs(cfg, tmp_path):
    from rules import close_panel, portfolio_vol_target_weights, run_rule_strategy
    from walkforward import summarize
    assets, split = load_dev_assets(cfg)
    W = portfolio_vol_target_weights(close_panel(assets), target=0.5)   # high target -> gross cap binds
    assert W.values.min() >= 0 and W.sum(axis=1).max() <= 1 + 1e-9 and W.sum(axis=1).max() > 0.9
    folds = make_folds(cfg, assets, split)[:3]
    c = _rule_cfg(cfg, "portfolio_vol_target", {"target": 0.08, "window": 60})
    cheap = summarize("x", run_rule_strategy(c, assets, folds, tmp_path / "a"))
    dear = summarize("y", run_rule_strategy(c, assets, folds, tmp_path / "b", cost_mult=5.0))
    assert dear["mean_return"] < cheap["mean_return"] and cheap["mean_turnover"] > 0


def test_closed_market_cannot_trade():
    from backtest import weights_backtest
    d = pd.bdate_range("2021-01-04", periods=4)
    tradable = np.array([[True, False], [True, False], [True, True]])
    bt = weights_backtest(d[:-1], d[1:], np.full((3, 2), 0.5), np.zeros((3, 2)), tradable, 0.01, 1.0, None)
    held = bt.positions.values
    assert held[0, 1] == 0 and held[1, 1] == 0 and held[2, 1] == 0.5   # asset 2 only bought once its market opens


def test_rule_robustness_and_verdict(cfg, tmp_path):
    from evaluation import final_verdict
    from robustness import run_rule_robustness
    c = _rule_cfg(cfg, "vol_target", {"target": 0.12, "window": 20, "band": 0.1},
                  neighborhood=[{"target": 0.09}, {"window": 40}])
    rob = run_rule_robustness(c, tmp_path)
    assert set(rob) == {"costs_2x", "other_assets", "shifted_dates", "param_neighborhood"}
    assert all(r["status"] == "ok" for r in rob.values()), rob
    assert len(rob["param_neighborhood"]["neighbors"]) == 2
    # a neighbourhood failure must fail the verdict even if the worst-Sharpe summary looks fine
    rob["param_neighborhood"]["pass"] = False
    rob["param_neighborhood"]["reasons"] = ["{'window': 40}: survival 50%"]
    s = {"mean_sharpe": 1.5, "mean_return": 0.5, "penalized_sharpe": 1.2, "survival_rate": 1.0, "survived": 9,
         "n_ok": 9, "n_runs": 9, "complete": True, "dsr_prob": 0.99, "mean_stitched_mdd": 0.1}
    v = final_verdict(s, rob, ["costs_2x", "param_neighborhood"], c["pass_criteria"])
    assert not v["pass"] and any("param_neighborhood" in r for r in v["reasons"])


def test_proposed_bar_is_inactive_by_default(cfg):
    from evaluation import base_checks
    pc = cfg["pass_criteria"]
    assert pc["sharpe_test"] == "penalized"
    s = {"mean_sharpe": 1.46, "mean_return": 1.18, "penalized_sharpe": 0.78, "dsr_prob": 0.99,
         "survival_rate": 1.0, "survived": 9, "n_ok": 9, "n_runs": 9, "complete": True}
    assert not base_checks(s, pc)[0]                 # active bar: fails
    assert base_checks(s, pc, sharpe_test="dsr")[0]   # proposed bar: would pass (that is why it needs approval)


def test_invalid_rule_config_rejected(cfg):
    from config_utils import validate
    with pytest.raises(ValueError):
        validate(_rule_cfg(cfg, "no_such_rule", {}))
    with pytest.raises(ValueError):
        validate(_rule_cfg(cfg, "vol_target", {"target": 0.1}, neighborhood=[{"windw": 5}]))


def test_long_history_respects_test_lock(cfg):
    from long_history import early_period_runs, run_long_history
    strategies, folds, _, split = run_long_history(load_base(), synthetic=True)
    assert pd.Timestamp(folds[-1].val_end) < split.test_start
    assert pd.Timestamp(folds[0].val_start) < pd.Timestamp("2008-01-01")       # covers 2008
    runs = strategies["baseline:vol_target"]
    early = early_period_runs(runs)
    assert early and all(r["fold"]["val_start"] < "2016-01-01" for r in early)


# ---------------------------------------------------------------- round 4
def _panel(cfg):
    from rules import close_panel
    return close_panel(build_assets(load_prices(cfg), cfg["features"])).dropna()


def test_leverage_pays_financing():
    from backtest import weights_backtest
    d = pd.bdate_range("2021-01-04", periods=253)
    z = np.zeros((252, 1))
    rf = np.full(252, 0.0002)
    bt = weights_backtest(d[:-1], d[1:], np.full((252, 1), 1.5), z, np.ones((252, 1), bool), 0.0, 1.0, None,
                          rf=rf, borrow_spread=0.01)
    expect = (1 - 0.5 * (0.0002 + 0.01 / 252)) ** 252
    assert np.isclose(bt.equity.iloc[-1], expect)
    cash = weights_backtest(d[:-1], d[1:], np.full((252, 1), 0.5), z, np.ones((252, 1), bool), 0.0, 1.0, None,
                            rf=rf, credit_cash=True)
    assert np.isclose(cash.equity.iloc[-1], (1 + 0.5 * 0.0002) ** 252)


def test_equal_risk_and_bootstrap_tools():
    from evaluation import bootstrap_sharpe_diff, cagr_at_vol
    rng = np.random.default_rng(1)
    b = rng.normal(0.0002, 0.01, 2520)
    a = b + rng.normal(0.0004, 0.002, 2520)          # clearly better, highly correlated
    assert bootstrap_sharpe_diff(a, b, n_boot=500) < 0.05
    assert bootstrap_sharpe_diff(b, b, n_boot=200) == 1.0
    c10, k = cagr_at_vol(b * 0.5, np.zeros(2520), 0.10)
    assert np.isclose(k, 0.10 / (0.5 * b.std(ddof=1) * np.sqrt(252))) and np.isfinite(c10)


def test_monthly_rebalance_days():
    from allocation import rebalance_days
    idx = pd.bdate_range("2021-01-01", "2021-04-30")
    r = rebalance_days(idx)
    assert r.sum() == 4 and all(idx[r].day <= 3)


def test_tracker_append_only_and_resumable(cfg, tmp_path):
    """Running in two pieces must equal running once, and old rows are never rewritten."""
    from paper_trade import update_tracker
    from selfcheck import ALLOC_SPECS
    panel = _panel(cfg)
    rf = pd.Series(0.0001, index=panel.index)
    plan = {"hypotheses": ALLOC_SPECS, "common": {"costs_bps": 15, "borrow_spread": 0.01, "max_drawdown_kill": 0.15},
            "universes": {"exploration": {"benchmarks": {"equity": panel.columns[0], "bonds": panel.columns[1]}}}}
    names = ["lv", "tr", "bench:60_40"]
    start = panel.index[2000]
    mid, end = panel.index[2100], panel.index[2250]
    update_tracker(panel.loc[:mid], rf, plan, names, tmp_path / "a", start)
    first = pd.read_csv(tmp_path / "a" / "lv.csv")
    added = update_tracker(panel.loc[:end], rf, plan, names, tmp_path / "a", start)
    again = update_tracker(panel.loc[:end], rf, plan, names, tmp_path / "a", start)
    both = pd.read_csv(tmp_path / "a" / "lv.csv")
    update_tracker(panel.loc[:end], rf, plan, names, tmp_path / "b", start)
    once = pd.read_csv(tmp_path / "b" / "lv.csv")
    assert added["lv"] == 150 and again["lv"] == 0
    pd.testing.assert_frame_equal(both.iloc[: len(first)], first)
    assert np.allclose(both["equity"], once["equity"])


def test_research_stage_synthetic(tmp_path, monkeypatch):
    """Exploration -> finalists (pre-registered rule) -> confirmation gated by approval."""
    import research
    base = deep_merge(load_base(), {"data": {"source": "synthetic"}})
    res = research.run_research(base, tmp_path, True, legacy_trials=27, approved=False)
    assert res["n_total"] == 27 + len(res["plan"]["hypotheses"])
    assert res["confirmation"]["status"].startswith("not approved")
    fin = res["finalists"]
    assert len(fin) <= 3
    for n in fin:
        assert res["summ"][n]["survival_rate"] >= 0.8 and res["summ"][n]["costs_2x_return"] > 0
    assert all(res["summ"][n]["verdict"] == "FAIL" for n in res["plan"]["hypotheses"])   # nothing confirmed
    res2 = research.run_research(base, tmp_path, True, legacy_trials=27, approved=True)
    if fin:
        assert res2["confirmation"]["status"] == "ok"
        assert (tmp_path / "research" / "confirmation_LOCK.json").exists()
    # a changed hypothesis becomes a NEW exploratory trial
    plan = research.load_plan()
    plan["hypotheses"]["H1_equal_weight"]["params"] = {"window": 20}
    monkeypatch.setattr(research, "load_plan", lambda path=None: plan)
    res3 = research.run_research(base, tmp_path, True, legacy_trials=27, approved=False)
    assert res3["status"]["H1_equal_weight"] == "exploratory"
    assert res3["n_total"] == res["n_total"] + 1


def test_confirmation_universe_is_disjoint():
    import research
    plan = research.load_plan()
    E, C = plan["universes"]["exploration"], plan["universes"]["confirmation"]
    assert not set(E["tickers"]) & set(C["tickers"])
    assert pd.Timestamp(C["eval_end"]) < pd.Timestamp(E["eval_start"])


# ---------------------------------------------------------------- round 5
def test_french_download_failure_is_clear(tmp_path, monkeypatch):
    import factor_data
    monkeypatch.setattr(factor_data, "CACHE", tmp_path)

    def boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(factor_data.urllib.request, "urlopen", boom)
    with pytest.raises(factor_data.DataUnavailable) as e:
        factor_data.fetch("https://example.invalid/", "F-F_Research_Data_Factors_CSV.zip")
    msg = str(e.value)
    assert "data_library.html" in msg and "F-F_Research_Data_Factors_CSV.zip" in msg and str(tmp_path) in msg


def test_turn_of_month_costs_and_switches():
    from factor_research import tom_sleeve_monthly
    days = pd.bdate_range("2010-01-01", "2010-12-31")
    flat = pd.DataFrame({"mkt": 0.0, "rf": 0.0}, index=days)
    m = tom_sleeve_monthly(flat, 1, 3, 0.0015)
    assert (m < 0).all()                                   # no edge -> only switching costs
    assert np.allclose(m.iloc[1:-1], (1 - 0.0015) ** 2 - 1, atol=1e-6)   # in and out once per month


def test_india_taxes_and_fixed_charges():
    from india_costs import simulate
    m = pd.date_range("2015-04-30", periods=60, freq="ME")
    W = pd.DataFrame({"market": 1.0, "cash": 0.0}, index=m)
    R = pd.DataFrame({"market": 0.01, "cash": 0.0}, index=m)
    hold = simulate(W, R, 1e6, "india_listed")
    churn = simulate(W, R, 1e6, "india_listed", round_trip_each_month=True)
    assert churn["tax_paid"] > hold["tax_paid"] and churn["cagr"] < hold["cagr"]
    small = simulate(W, R, 1e5, "india_listed", round_trip_each_month=True)
    big = simulate(W, R, 1e7, "india_listed", round_trip_each_month=True)
    assert small["charges"] / 1e5 > big["charges"] / 1e7      # fixed DP charges hurt small accounts more


def test_round5_stage_synthetic(tmp_path, monkeypatch):
    import factor_research as fr
    base = deep_merge(load_base(), {"data": {"source": "synthetic"}})
    res = fr.run_round5(base, tmp_path, True, legacy_trials=27, approved=False)
    n_trials5 = len(fr.all_trials(res["plan"]))
    assert res["n_total"] == 27 + n_trials5
    assert res["confirmation"]["status"].startswith("not approved")
    assert len(res["finalists"]) <= 3
    for n in res["finalists"]:
        assert res["summ"][n]["survival_rate"] >= 0.8 and res["summ"][n]["costs_2x_excess_cagr"] > 0
    assert all(res["summ"][n]["verdict"] == "FAIL" for n in res["plan"]["hypotheses"])     # unconfirmed
    b = res["summ"]["bench:market"]
    assert b["last"] < "2024-10-16"                                                          # never the test set
    res2 = fr.run_round5(base, tmp_path, True, legacy_trials=27, approved=True)
    if res["finalists"]:
        assert res2["confirmation"]["status"] == "ok"
        us = res2["confirmation"]["sets"]["US 1927-1963"]["summ"]["bench:market"]
        assert us["last"] <= "1963-06-30"
        assert (tmp_path / "research5" / "confirmation_LOCK.json").exists()
    plan = fr.load_plan()
    plan["hypotheses"]["R5_H3_big_value"]["sleeve"] = {"big_value": 0.9, "market": 0.1}
    monkeypatch.setattr(fr, "load_plan", lambda path=None: plan)
    res3 = fr.run_round5(base, tmp_path, True, legacy_trials=27, approved=False)
    assert res3["status"]["R5_H3_big_value"] == "exploratory" and res3["n_total"] == res["n_total"] + 1


def test_tracker_converts_round5_finalists():
    from factor_research import load_plan as lp5
    from paper_trade import tracked_plan
    from research import load_plan as lp4
    p = tracked_plan(lp4(), lp5(), ["R5_H5_value_momentum_trend", "R5_H6_turn_of_month"])
    a = p["hypotheses"]["R5_H5_value_momentum_trend"]["params"]
    assert a["mix"] == {"IWD": 0.5, "MTUM": 0.5} and a["trend_signal"] == "SPY" and a["cash"] == "SHV"
    assert p["hypotheses"]["R5_H6_turn_of_month"]["params"]["rebalance"] == "daily"
    assert "r5_bench:market" in p["hypotheses"]
