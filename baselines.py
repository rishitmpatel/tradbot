"""Classic rule-based baselines. Fixed textbook parameters (from base_config.yaml),
NOT tuned, so they are honest benchmarks the RL agent must beat.

Every rule returns a position series indexed by date using closes up to and
including that date only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(close: pd.Series, **_) -> pd.Series:
    return pd.Series(1.0, index=close.index)


def ma_crossover(close: pd.Series, fast: int = 50, slow: int = 200, allow_short: bool = False, **_) -> pd.Series:
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    pos = np.where(f > s, 1.0, -1.0 if allow_short else 0.0)
    return pd.Series(pos, index=close.index).where(s.notna(), 0.0)


def mean_reversion(close: pd.Series, window: int = 20, entry_z: float = -1.0, exit_z: float = 0.0, **_) -> pd.Series:
    """Long when price is stretched below its moving average; exit when it reverts."""
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    z = ((close - ma) / sd).values
    pos = np.zeros(len(close))
    holding = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            holding = 0.0
        elif holding == 0.0 and zi < entry_z:
            holding = 1.0
        elif holding == 1.0 and zi > exit_z:
            holding = 0.0
        pos[i] = holding
    return pd.Series(pos, index=close.index)


def momentum(close: pd.Series, lookback: int = 252, skip: int = 21, allow_short: bool = False, **_) -> pd.Series:
    """Time-series momentum (12-1): long if the past year's return excluding the last month is positive."""
    past = close.shift(skip) / close.shift(lookback) - 1.0
    pos = np.where(past > 0, 1.0, -1.0 if allow_short else 0.0)
    return pd.Series(pos, index=close.index).where(past.notna(), 0.0)


BASELINES = {
    "buy_and_hold": buy_and_hold,
    "ma_crossover": ma_crossover,
    "mean_reversion": mean_reversion,
    "momentum": momentum,
}


def baseline_positions(name: str, close: pd.Series, params: dict) -> pd.Series:
    return BASELINES[name](close, **(params or {}))
