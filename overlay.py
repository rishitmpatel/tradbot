"""Round 2: RL as an OVERLAY on a simple rule ("residual RL").

Round 1 asked the agent to discover everything from scratch, including the
basic fact that being long earns the equity premium. It mostly failed, and
plain buy-and-hold won. Here the agent starts from a rule and only learns
when to DEVIATE from it:

    final position = base_position(t) * multiplier chosen by the agent

    base_position = base_signal (buy_and_hold / momentum / ...)  x  vol scale
    multiplier    = one of env.actions, e.g. [0, 0.5, 1]

With multiplier 1 everywhere the agent simply IS the rule, so a useless agent
should end up near the rule, not in cash. A useful one cuts risk at the right
times.

Everything here uses closes up to and including day t only (selfcheck.py verifies this).
"""
from __future__ import annotations

import pandas as pd

from baselines import baseline_positions, vol_scale


def has_overlay(cfg: dict) -> bool:
    env = cfg["env"]
    return (env.get("base_signal") or "none") != "none" or bool(env.get("vol_target"))


def base_position(close: pd.Series, cfg: dict) -> pd.Series:
    """Position of the rule before the agent's multiplier, in [-1, 1]."""
    env = cfg["env"]
    sig = env.get("base_signal") or "none"
    if sig == "none":
        b = pd.Series(1.0, index=close.index)  # round-1 behaviour: action = position
    else:
        b = baseline_positions(sig, close, cfg["baselines"].get(sig) or {})
    if env.get("vol_target"):
        b = b * vol_scale(close, float(env["vol_target"]), int(env.get("vol_window", 20)),
                          float(env.get("max_leverage", 1.0)))
    return b.clip(-1.0, 1.0)


def reference_position(close: pd.Series, cfg: dict) -> pd.Series | None:
    """Benchmark the reward is measured against (reward.relative_to):
    none -> absolute reward; base -> the rule itself; buy_and_hold -> always long."""
    rel = cfg["reward"].get("relative_to") or "none"
    if rel == "none":
        return None
    if rel == "base":
        return base_position(close, cfg)
    if rel == "buy_and_hold":
        return pd.Series(1.0, index=close.index)
    raise ValueError(f"reward.relative_to must be none|base|buy_and_hold, got {rel}")
