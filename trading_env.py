"""The reinforcement-learning environment (Gymnasium API).

==========================  RL IN ONE PAGE  ==================================
An RL agent learns by trial and error inside an "environment":

    loop each decision day t:
        STATE  s_t   what the agent observes (here: today's features, the
                     rule's base position, its own position, its drawdown)
        ACTION a_t   what it decides (here: a number from env.actions; it is
                     the position itself, or with an overlay a MULTIPLIER on the
                     rule's base position), held until the next decision
        REWARD r_t   a number saying how good that decision turned out
                     (profit AFTER costs, or a Sharpe-style score, possibly
                     measured against a benchmark, minus penalties)
        -> environment moves to s_{t+1}

The agent's goal is to maximise the discounted sum of future rewards
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
(gamma = agent.<algo>.gamma in the config, e.g. 0.99).

* DQN learns Q(s, a) = expected G_t if it takes action a in state s and acts
  well afterwards. It then picks argmax_a Q(s, a).
* PPO learns a POLICY pi(a | s) = probability of each action, plus a value
  function V(s) = expected G_t, and nudges pi toward actions that turned out
  better than V expected ("advantage").

THE REWARD DEFINES THE TASK. The agent optimises exactly what the reward
says, not what we meant. Round 1 showed this: a per-day penalty on new
drawdown made holding cash the best answer (see README "Round 2").

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
    base[i]      rule position before the agent's multiplier (1.0 = no overlay)
    ref[i]       benchmark position for a relative reward (None = absolute reward)
    first        first index at which the agent may act (obs_window history needed)
    """

    def __init__(self, features: np.ndarray, fwd_ret: np.ndarray, dates, first: int = 0, ticker: str = "",
                 base: np.ndarray | None = None, ref: np.ndarray | None = None):
        self.features = np.asarray(features, dtype=np.float32)
        self.fwd_ret = np.asarray(fwd_ret, dtype=np.float64).copy()
        self.fwd_ret[-1] = np.nan
        self.dates = dates
        self.first = first
        self.ticker = ticker
        n = len(self.fwd_ret)
        self.base = np.ones(n) if base is None else np.asarray(base, dtype=np.float64)
        self.ref = None if ref is None else np.asarray(ref, dtype=np.float64)

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
        self.interval = max(1, int(env_cfg.get("decision_interval", 1)))
        self.band = float(env_cfg.get("rebalance_band", 0.0))
        self.inc_pos = bool(env_cfg.get("include_position", True))
        self.inc_dd = bool(env_cfg.get("include_drawdown", True))
        inc_base = env_cfg.get("include_base", "auto")
        overlay = (env_cfg.get("base_signal") or "none") != "none" or bool(env_cfg.get("vol_target"))
        self.inc_base = overlay if inc_base == "auto" else bool(inc_base)
        self.rc = reward_cfg
        self.cost_rate = cost_rate
        self.max_dd = max_dd
        self.train = train
        n_feat = self.segments[0].features.shape[1]

        # ---- STATE (observation) space ----------------------------------
        # [features of the last `obs_window` days (flattened), base position, position, drawdown]
        obs_dim = self.window * n_feat + int(self.inc_base) + int(self.inc_pos) + int(self.inc_dd)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        # ---- ACTION space -------------------------------------------------
        # Discrete(k): action i -> position = base[t] * self.actions[i]
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
        self.ref_pos = 0.0
        self.equity = 1.0
        self.peak = 1.0
        self.dd = 0.0
        self.dead = False
        # running estimates for the differential Sharpe ratio (mean and mean square of returns)
        self.A = 0.0
        self.B = float(self.rc.get("dsr_init_vol", 0.01)) ** 2
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        f = self.seg.features[self.t - self.window + 1: self.t + 1].reshape(-1)
        extra = []
        if self.inc_base:
            extra.append(self.seg.base[self.t])
        if self.inc_pos:
            extra.append(self.pos)
        if self.inc_dd:
            extra.append(self.dd * 10.0)  # scaled so 10% drawdown ~ 1.0
        return np.concatenate([f, np.array(extra, dtype=np.float32)]).astype(np.float32)

    def _banded(self, want: float, current: float) -> float:
        """Skip tiny rebalances (env.rebalance_band); exits to 0 always execute."""
        if self.band > 0 and want != 0.0 and abs(want - current) <= self.band:
            return current
        return want

    # ------------------------------------------------------------------
    def _reward(self, net: float, ref_net: float | None, cost: float, prev_dd: float, died: bool) -> float:
        rc = self.rc
        kind = rc.get("type", "log_return")
        if kind == "differential_sharpe":
            # Differential Sharpe ratio (Moody & Saffell 1998): how much THIS day's
            # return raises (+) or lowers (-) a running Sharpe estimate. It rewards
            # return per unit of risk directly, instead of punishing every down day.
            x = net - (ref_net if ref_net is not None else 0.0)
            eta = float(rc.get("dsr_eta", 1 / 63))
            dA, dB = x - self.A, x * x - self.B
            var = self.B - self.A ** 2
            core = (self.B * dA - 0.5 * self.A * dB) / var ** 1.5 if var > 1e-12 else 0.0
            core = float(np.clip(core, -float(rc.get("dsr_clip", 5.0)), float(rc.get("dsr_clip", 5.0))))
            self.A += eta * dA
            self.B += eta * dB
        elif kind == "log_return":
            core = np.log1p(net)                                       # profit after costs
            if ref_net is not None:
                core -= np.log1p(ref_net)                              # ...measured against the benchmark
        else:
            raise ValueError(f"reward.type must be log_return|differential_sharpe, got {kind}")

        # Drawdown penalties only count drawdown BEYOND the soft limit, so normal
        # day-to-day wiggles are not punished (that is what pushed round 1 into cash).
        soft = float(rc.get("drawdown_soft_limit", 0.0))
        dd_eff, prev_eff = max(0.0, self.dd - soft), max(0.0, prev_dd - soft)
        reward = core
        reward -= float(rc.get("extra_cost_penalty", 0.0)) * cost       # optional turnover penalty
        reward -= float(rc.get("drawdown_penalty", 0.0)) * dd_eff       # being deep under water
        reward -= float(rc.get("drawdown_increase_penalty", 0.0)) * max(0.0, dd_eff - prev_eff)
        reward -= float(rc.get("risk_penalty", 0.0)) * net * net        # variance penalty
        if died:
            reward -= float(rc.get("death_penalty", 0.0))               # the agent "dies" (once)
        return reward * float(rc.get("scale", 1.0))

    def step(self, action):
        mult = float(self.actions[int(action)])                       # ACTION
        total_reward, nets, poss = 0.0, [], []
        terminated = False
        for _ in range(self.interval):                                # hold the decision for N days
            if self.t > self.end:
                break
            seg = self.seg
            target = self._banded(float(np.clip(seg.base[self.t] * mult, -1.0, 1.0)), self.pos)
            r = float(seg.fwd_ret[self.t])                             # what that position earns by next close
            if not np.isfinite(r):
                raise RuntimeError("env tried to use a return outside its window (lookahead bug)")
            cost = abs(target - self.pos) * self.cost_rate
            net = (1.0 - cost) * (1.0 + target * r) - 1.0              # identical formula to backtest.py
            ref_net = None
            if seg.ref is not None:                                    # benchmark held the same way
                ref_t = self._banded(float(seg.ref[self.t]), self.ref_pos)
                ref_net = (1.0 - abs(ref_t - self.ref_pos) * self.cost_rate) * (1.0 + ref_t * r) - 1.0
                self.ref_pos = ref_t

            prev_dd = self.dd
            self.equity *= 1.0 + net
            self.peak = max(self.peak, self.equity)
            self.dd = 1.0 - self.equity / self.peak
            died = self.dd > self.max_dd and not self.dead
            self.dead = self.dead or died
            total_reward += self._reward(net, ref_net, cost, prev_dd, died)   # REWARD

            self.pos = target
            self.t += 1
            nets.append(net)
            poss.append(target)
            if self.dead and self.rc.get("terminate_on_death", True) and self.train:
                terminated = True
                break
        truncated = self.t > self.end
        obs = self._obs()  # STATE at the next decision
        info = {"net": nets[-1], "pos": poss[-1], "nets": nets, "positions": poss,
                "equity": self.equity, "drawdown": self.dd}
        return obs, float(total_reward), terminated, truncated, info


def rollout_positions(model, env: TradingEnv, segment_index: int) -> np.ndarray:
    """Run a trained policy through one evaluation segment day by day.
    Returns the position held for each decision day (deterministic = greedy)."""
    obs, _ = env.reset(options={"segment": segment_index})
    positions = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        positions.extend(info["positions"])
        if terminated or truncated:
            break
    return np.array(positions)
