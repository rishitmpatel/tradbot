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
    files = list_experiment_files()
    assert len(files) >= 4
    names = [load_experiment(f)["name"] for f in files]
    assert len(set(names)) == len(names)


def test_self_checks(prices, cfg):
    assert len(run_all_checks(prices, cfg)) == 5


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
