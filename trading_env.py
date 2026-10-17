"""The reinforcement-learning environment (Gymnasium API).

==========================  RL IN ONE PAGE  ==================================
An RL agent learns by trial and error inside an "environment":

    loop each trading day t:
        STATE  s_t   what the agent observes (here: today's features + its
                     own position + its current drawdown)
        ACTION a_t   what it decides (here: which position to hold until
                     tomorrow's close, chosen from env.actions, e.g. 0 or 1)
        REWARD r_t   a number saying how good that decision turned out
                     (here: tomorrow's profit AFTER costs, minus penalties for
                     drawdown and risk)
        -> environment moves to s_{t+1}

The agent's goal is to maximise the discounted sum of future rewards
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
(gamma = agent.<algo>.gamma in the config, e.g. 0.99).

* DQN learns Q(s, a) = expected G_t if it takes action a in state s and acts
  well afterwards. It then picks argmax_a Q(s, a).
* PPO learns a POLICY pi(a | s) = probability of each action, plus a value
  function V(s) = expected G_t, and nudges pi toward actions that turned out
  better than V expected ("advantage").

Episode: a random 1-year slice of one random asset from the TRAINING window.
It ends early if the agent "dies" (drawdown > survival.max_drawdown).
==============================================================================
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Segment:
    """Arrays for one asset over one date window.
    features[i]  features known at close of day i
    fwd_ret[i]   return close i -> close i+1 (NaN on the window's last day:
                 that return lies OUTSIDE the window, so it can never be used)
    first        first index at which the agent may act (obs_window history needed)
    """

    def __init__(self, features: np.ndarray, fwd_ret: np.ndarray, dates, first: int = 0, ticker: str = ""):
        self.features = np.asarray(features, dtype=np.float32)
        self.fwd_ret = np.asarray(fwd_ret, dtype=np.float64).copy()
        self.fwd_ret[-1] = np.nan
        self.dates = dates
        self.first = first
        self.ticker = ticker

    @property
    def last_decision(self) -> int:  # last index whose fwd_ret is inside the window
        return len(self.fwd_ret) - 2


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, segments: list[Segment], env_cfg: dict, reward_cfg: dict, cost_rate: float,
                 max_dd: float, train: bool = True):
        super().__init__()
        self.segments = [s for s in segments if s.last_decision - s.first >= 5]
        if not self.segments:
            raise ValueError("no usable segments for environment")
        self.actions = np.array(env_cfg["actions"], dtype=np.float64)
        self.window = int(env_cfg.get("obs_window", 1))
        self.ep_len = int(env_cfg.get("episode_length", 252))
        self.inc_pos = bool(env_cfg.get("include_position", True))
        self.inc_dd = bool(env_cfg.get("include_drawdown", True))
        self.rc = reward_cfg
        self.cost_rate = cost_rate
        self.max_dd = max_dd
        self.train = train
        n_feat = self.segments[0].features.shape[1]

        # ---- STATE (observation) space ----------------------------------
        # [features of the last `obs_window` days (flattened), position, drawdown]
        obs_dim = self.window * n_feat + int(self.inc_pos) + int(self.inc_dd)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        # ---- ACTION space -------------------------------------------------
        # Discrete(k): action i -> hold position self.actions[i] until next close.
        self.action_space = spaces.Discrete(len(self.actions))

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.train:
            # random asset, random start => many different training episodes
            self.seg = self.segments[self.np_random.integers(len(self.segments))]
            lo = max(self.seg.first, self.window - 1)
            hi = max(lo, self.seg.last_decision - self.ep_len + 1)
            self.t = int(self.np_random.integers(lo, hi + 1))
            self.end = min(self.seg.last_decision, self.t + self.ep_len - 1)
        else:
            # evaluation: walk one segment from start to end, no randomness
            self.seg = self.segments[int(options.get("segment", 0))]
            self.t = max(self.seg.first, self.window - 1)
            self.end = self.seg.last_decision
        self.pos = 0.0
        self.equity = 1.0
        self.peak = 1.0
        self.dd = 0.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        f = self.seg.features[self.t - self.window + 1: self.t + 1].reshape(-1)
        extra = []
        if self.inc_pos:
            extra.append(self.pos)
        if self.inc_dd:
            extra.append(self.dd * 10.0)  # scaled so 10% drawdown ~ 1.0
        return np.concatenate([f, np.array(extra, dtype=np.float32)]).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        target = float(self.actions[int(action)])           # ACTION -> position
        r = float(self.seg.fwd_ret[self.t])                  # what that position earns by next close
        if not np.isfinite(r):
            raise RuntimeError("env tried to use a return outside its window (lookahead bug)")
        traded = abs(target - self.pos)
        cost = traded * self.cost_rate
        net = (1.0 - cost) * (1.0 + target * r) - 1.0        # identical formula to backtest.py

        prev_dd = self.dd
        self.equity *= 1.0 + net
        self.peak = max(self.peak, self.equity)
        self.dd = 1.0 - self.equity / self.peak
        dead = self.dd > self.max_dd

        # ---- REWARD ------------------------------------------------------
        rc = self.rc
        reward = np.log1p(net)                                         # profit after costs
        reward -= float(rc.get("extra_cost_penalty", 0.0)) * cost       # optional turnover penalty
        reward -= float(rc.get("drawdown_penalty", 0.0)) * self.dd      # being under water
        reward -= float(rc.get("drawdown_increase_penalty", 0.0)) * max(0.0, self.dd - prev_dd)
        reward -= float(rc.get("risk_penalty", 0.0)) * net * net        # variance penalty
        if dead:
            reward -= float(rc.get("death_penalty", 0.0))               # the agent "dies"
        reward *= float(rc.get("scale", 1.0))

        self.pos = target
        self.t += 1
        terminated = bool(dead and rc.get("terminate_on_death", True) and self.train)
        truncated = self.t > self.end
        obs = self._obs()  # STATE at the next close
        info = {"net": net, "cost": cost, "pos": target, "equity": self.equity, "drawdown": self.dd}
        return obs, float(reward), terminated, truncated, info


def rollout_positions(model, env: TradingEnv, segment_index: int) -> np.ndarray:
    """Run a trained policy through one evaluation segment day by day.
    Returns the position decided at each decision day (deterministic = greedy)."""
    obs, _ = env.reset(options={"segment": segment_index})
    positions = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        positions.append(info["pos"])
        if terminated or truncated:
            break
    return np.array(positions)
