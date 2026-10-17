"""Features: what the agent (and nothing else) gets to see each day.

RULE: every feature value on day t uses closes up to and including day t only.
The single forward-looking quantity, `fwd_ret` (close t -> close t+1), is the
thing the position decided at close t earns. It is used for P&L and reward,
never as an input. `check_causality` verifies this on real data at startup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_NAME = re.compile(r"^([a-z_]+?)_(\d+)$")


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(up.notna())


def raw_feature(close: pd.Series, name: str) -> pd.Series:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"Bad feature name '{name}'. Use e.g. mom_20, vol_60, rsi_14")
    kind, n = m.group(1), int(m.group(2))
    lc = np.log(close)
    lr1 = lc.diff()
    if kind in ("logret", "mom"):
        return lc - lc.shift(n)
    if kind == "vol":
        return lr1.rolling(n, min_periods=n).std() * np.sqrt(252)
    if kind == "rsi":
        return _rsi(close, n)
    if kind == "ma_ratio":
        return lc - np.log(close.rolling(n, min_periods=n).mean())
    raise ValueError(f"Unknown feature kind '{kind}' in '{name}'")


def compute_features(close: pd.Series, fcfg: dict) -> pd.DataFrame:
    feats = pd.DataFrame({name: raw_feature(close, name) for name in fcfg["list"]}, index=close.index)
    if fcfg.get("normalize", "rolling_z") == "rolling_z":
        w = int(fcfg.get("z_window", 252))
        mu = feats.rolling(w, min_periods=w).mean()
        sd = feats.rolling(w, min_periods=w).std()
        feats = (feats - mu) / sd.replace(0, np.nan)
    clip = fcfg.get("clip")
    if clip:
        feats = feats.clip(-clip, clip)
    return feats


@dataclass
class AssetData:
    ticker: str
    features: pd.DataFrame  # rows = trading days with valid features
    fwd_ret: pd.Series      # same index; return from this close to NEXT close (NaN on last day)
    close: pd.Series        # full close history (for baselines)


def build_assets(prices: dict[str, pd.DataFrame], fcfg: dict) -> dict[str, AssetData]:
    out = {}
    for t, df in prices.items():
        close = df["close"].astype(float)
        feats = compute_features(close, fcfg)
        fwd = close.shift(-1) / close - 1.0
        valid = feats.notna().all(axis=1).values
        if not valid.any():
            raise ValueError(f"{t}: not enough history to compute features")
        first = int(np.argmax(valid))
        # After warm-up keep EVERY trading day (so fwd_ret stays day-to-next-day);
        # rare interior NaNs (e.g. zero rolling std) are filled causally.
        f = feats.iloc[first:].ffill().fillna(0.0)
        out[t] = AssetData(t, f.astype(np.float32), fwd.iloc[first:], close)
    return out


def check_causality(close: pd.Series, fcfg: dict, n_checks: int = 6, seed: int = 0) -> None:
    """Recompute features on data truncated at day t; values at t must be identical.
    If a feature peeked at the future this would fail."""
    full = compute_features(close, fcfg)
    valid_idx = np.where(full.notna().all(axis=1).values)[0]
    if len(valid_idx) < 10:
        raise AssertionError("not enough data for causality check")
    rng = np.random.default_rng(seed)
    for i in rng.choice(valid_idx[:-1], size=min(n_checks, len(valid_idx) - 1), replace=False):
        trunc = compute_features(close.iloc[: i + 1], fcfg)
        a, b = full.iloc[i].values, trunc.iloc[-1].values
        if not np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True):
            raise AssertionError(f"LOOKAHEAD: features at {close.index[i].date()} change when future data is removed")
