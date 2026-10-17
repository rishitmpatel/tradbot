"""Round 4: multi-asset allocation rules (no learning).

Every rule returns a DataFrame of TARGET WEIGHTS (fraction of total equity per asset),
one row per trading day, computed only from prices up to and including that day.
The engine (backtest.weights_backtest) only trades on rebalance days; in between, holdings
drift with prices. Weights are long-only. Their sum is <= 1 (rest = cash) unless a rule
explicitly allows leverage.

Quant logic in one paragraph: a portfolio's return comes from the risk premia it holds
(equity, bond term, credit, commodity, ...) and from any timing skill. Holding many
premia that do not crash together (diversification) raises return per unit of risk
(Sharpe). Trend rules step aside from assets in long declines. Risk targeting then
chooses HOW MUCH risk to run. Profit ~ Sharpe x volatility, so a higher Sharpe can be
turned into more profit at the same risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baselines import apply_band


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return np.log(panel).diff()


def rolling_vol(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised volatility from the last `window` daily returns (NaN until enough history)."""
    return log_returns(panel).rolling(window, min_periods=window).std() * np.sqrt(252)


def excess_momentum(panel: pd.DataFrame, rf_daily: pd.Series, lookback: int) -> pd.DataFrame:
    """Return over `lookback` days minus what T-bills paid over the same days."""
    mom = panel / panel.shift(lookback) - 1.0
    rf_sum = rf_daily.reindex(panel.index).fillna(0.0).rolling(lookback, min_periods=lookback).sum()
    return mom.sub(rf_sum, axis=0)


def inverse_vol_weights(vol: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weights proportional to 1/vol over eligible assets, summing to 1 (0 if none eligible)."""
    inv = 1.0 / vol.replace(0, np.nan)
    if mask is not None:
        inv = inv.where(mask)
    tot = inv.sum(axis=1)
    return inv.div(tot.replace(0, np.nan), axis=0).fillna(0.0)


def rebalance_days(index: pd.DatetimeIndex, how: str = "monthly") -> np.ndarray:
    """True on the first trading day of each month (decide at that close)."""
    if how == "daily":
        return np.ones(len(index), dtype=bool)
    if how != "monthly":
        raise ValueError(f"unknown rebalance '{how}'")
    m = index.to_period("M")
    out = np.ones(len(index), dtype=bool)
    out[1:] = m[1:] != m[:-1]
    return out


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def equal_weight(panel, rf, window: int = 60, **_):
    live = rolling_vol(panel, window).notna()
    n = live.sum(axis=1).replace(0, np.nan)
    return live.astype(float).div(n, axis=0).fillna(0.0)


def inverse_vol(panel, rf, window: int = 60, **_):
    return inverse_vol_weights(rolling_vol(panel, window))


def trend(panel, rf, lookback: int = 252, window: int = 60, **_):
    """Inverse-vol weights, but an asset is held only while its excess momentum is positive;
    its share is otherwise left in cash (that is the crash protection)."""
    w = inverse_vol_weights(rolling_vol(panel, window))
    up = excess_momentum(panel, rf, lookback) > 0
    return w.where(up, 0.0)


def xs_momentum(panel, rf, lookback: int = 252, skip: int = 21, top_frac: float = 0.34, window: int = 60, **_):
    """Top fraction of assets by 12-1 month return, inverse-vol weighted, each also needing
    positive excess momentum (dual momentum); otherwise its share stays in cash."""
    vol = rolling_vol(panel, window)
    rank_sig = panel.shift(skip) / panel.shift(lookback) - 1.0
    abs_ok = excess_momentum(panel, rf, lookback) > 0
    sig = rank_sig.where(rank_sig.notna() & vol.notna())
    n_valid = sig.notna().sum(axis=1)
    k = np.maximum(1, np.ceil(top_frac * n_valid.values))           # how many to hold each day
    rank = sig.rank(axis=1, ascending=False, method="first")         # 1 = strongest
    chosen = rank.le(pd.Series(k, index=panel.index), axis=0) & sig.notna()
    w = inverse_vol_weights(vol, chosen)
    return w.where(abs_ok, 0.0)


def per_asset_vol_target(panel, rf, target: float = 0.12, window: int = 20, band: float = 0.02, **_):
    """Round 3's incumbent in portfolio form: each live asset gets 1/N of equity, scaled by
    target / its own volatility (capped at its 1/N share)."""
    vol = rolling_vol(panel, window)
    live = vol.notna()
    n = live.sum(axis=1).replace(0, np.nan)
    scale = (target / vol).clip(upper=1.0).where(live, 0.0)
    w = scale.div(n, axis=0).fillna(0.0)
    return w.apply(lambda c: pd.Series(apply_band(c.values, band), index=c.index))


def scale_to_vol(w: pd.DataFrame, panel, target: float, max_gross: float, window: int = 60) -> pd.DataFrame:
    """Scale a weight frame so its PREDICTED volatility (rolling covariance) equals target,
    never exceeding max_gross total exposure. >1 means leverage (financed by the engine)."""
    r = log_returns(panel).fillna(0.0)
    k = len(panel.columns)
    cov = (r.rolling(window, min_periods=window).cov() * 252).values.reshape(len(panel), k, k)
    out = np.zeros_like(w.values)
    for i in range(len(panel)):
        wi = w.values[i]
        g = wi.sum()
        if g <= 0 or not np.all(np.isfinite(cov[i])):
            continue
        pv = float(np.sqrt(max(wi @ cov[i] @ wi, 1e-12)))
        out[i] = wi * min(target / pv, max_gross / g)
    return pd.DataFrame(out, index=w.index, columns=w.columns)


BASE_RULES = {"equal_weight": equal_weight, "inverse_vol": inverse_vol, "trend": trend,
              "xs_momentum": xs_momentum, "per_asset_vol_target": per_asset_vol_target}


def build_weights(name: str, spec: dict, all_specs: dict, panel: pd.DataFrame, rf: pd.Series,
                  override: dict | None = None, _cache: dict | None = None) -> pd.DataFrame:
    """Target weights for hypothesis `name` (resolving ensembles / vol targeting recursively)."""
    params = {**(spec.get("params") or {}), **(override or {})}
    rule = spec["rule"]
    key = (name, repr(sorted(params.items())))
    if _cache is not None and key in _cache:
        return _cache[key]
    if rule in BASE_RULES:
        w = BASE_RULES[rule](panel, rf, **params)
    elif rule == "ensemble":
        parts = [build_weights(m, all_specs[m], all_specs, panel, rf, None, _cache) for m in params["members"]]
        w = sum(parts) / len(parts)
    elif rule == "vol_target_portfolio":
        base = build_weights(params["base"], all_specs[params["base"]], all_specs, panel, rf, None, _cache)
        w = scale_to_vol(base, panel, float(params["target"]), float(params["max_gross"]), int(params.get("window", 60)))
    else:
        raise ValueError(f"unknown rule '{rule}'")
    w = w.reindex(columns=panel.columns).fillna(0.0)
    if _cache is not None:
        _cache[key] = w
    return w


def rebalance_rule(spec: dict, all_specs: dict) -> str:
    """Daily for rules that say so (the incumbent), monthly otherwise (ensembles follow members)."""
    return (spec.get("params") or {}).get("rebalance", "monthly")


def benchmark_weights(kind: str, panel: pd.DataFrame, equity: str, bonds: str) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    if kind == "buy_and_hold":
        w[equity] = 1.0
    elif kind == "sixty_forty":
        w[equity], w[bonds] = 0.6, 0.4
    else:
        raise ValueError(kind)
    return w.where(panel.notna(), 0.0)
