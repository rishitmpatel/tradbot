"""Backtest engine: the single source of truth for P&L.

Timing convention (no lookahead):
    pos[i]      decided at the CLOSE of day i, using information up to day i
    fwd_ret[i]  close(i) -> close(i+1)
    trading cost is paid at close i on |pos[i] - pos[i-1]|
    net return realised on day i+1 = (1 - cost_i) * (1 + pos[i] * fwd_ret[i]) - 1

Portfolio: capital is split equally into one sleeve per asset at the start of a
run; sleeves are NOT rebalanced (so no hidden rebalancing costs). Portfolio
equity = sum of sleeve equities on the union calendar of all exchanges.

Survival ("agent dies"): if portfolio equity drops more than
survival.max_drawdown below its running peak at any close, every sleeve is
liquidated (paying costs) and the bot stays flat for the rest of the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def cost_rate(cfg: dict, multiplier: float = 1.0) -> float:
    return (float(cfg["costs"]["commission"]) + float(cfg["costs"]["slippage"])) * multiplier


def sleeve_returns(pos: np.ndarray, fwd_ret: np.ndarray, rate: float, start_pos: float = 0.0):
    """Net daily returns and traded units for one asset."""
    pos = np.asarray(pos, dtype=float)
    fwd_ret = np.asarray(fwd_ret, dtype=float)
    if len(pos) != len(fwd_ret):
        raise ValueError("pos and fwd_ret length mismatch")
    if not np.all(np.isfinite(fwd_ret)):
        raise ValueError("non-finite forward return inside the evaluation window")
    prev = np.concatenate([[start_pos], pos[:-1]])
    traded = np.abs(pos - prev)
    net = (1.0 - traded * rate) * (1.0 + pos * fwd_ret) - 1.0
    return net, traded


@dataclass
class Sleeve:
    """One asset inside a run window.
    dates      : decision days d0..d_{n-1}  (positions decided at these closes)
    real_dates : d1..d_n (day each net return is realised); d_n = last day of window
    """
    ticker: str
    dates: pd.DatetimeIndex
    real_dates: pd.DatetimeIndex
    pos: np.ndarray
    fwd_ret: np.ndarray


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    dead: bool
    death_date: pd.Timestamp | None
    metrics: dict = field(default_factory=dict)
    positions: pd.DataFrame | None = None


def portfolio_backtest(sleeves: list[Sleeve], rate: float, initial: float, max_dd: float | None,
                       start_date: pd.Timestamp | None = None) -> BacktestResult:
    n = len(sleeves)
    if n == 0:
        raise ValueError("no sleeves")
    share = initial / n
    union = sorted(set().union(*[set(s.dates) | set(s.real_dates) for s in sleeves]))
    union = pd.DatetimeIndex(union)
    if start_date is not None:
        union = union[union >= start_date]
    eq_cols, held_cols, trades = {}, {}, []
    for s in sleeves:
        net, traded = sleeve_returns(s.pos, s.fwd_ret, rate)
        trades.append(pd.Series(traded / n, index=s.dates))
        eq = pd.Series(share * np.cumprod(1.0 + net), index=s.real_dates)
        if len(s.dates):
            eq = pd.concat([pd.Series([share], index=[s.dates[0]]), eq])
            eq = eq[~eq.index.duplicated(keep="last")]
        eq_cols[s.ticker] = eq.reindex(union).ffill().fillna(share)
        # position on the books going INTO the close of each union day
        p = pd.Series(s.pos, index=s.dates).reindex(union).ffill().shift(1).fillna(0.0)
        held_cols[s.ticker] = p
    eq_df = pd.DataFrame(eq_cols)
    held_df = pd.DataFrame(held_cols)
    equity = eq_df.sum(axis=1)

    dead, death_date = False, None
    if max_dd is not None:
        dd = 1.0 - equity / equity.cummax()
        breach = np.where(dd.values > max_dd)[0]
        if len(breach):
            k = breach[0]
            dead, death_date = True, equity.index[k]
            liq_cost = float((eq_df.iloc[k] * held_df.iloc[k].abs()).sum() * rate)
            frozen = equity.iloc[k] - liq_cost
            equity = equity.copy()
            equity.iloc[k:] = frozen
            held_df.iloc[k + 1:] = 0.0

    all_trades = pd.concat(trades)
    if dead:  # trades after death never happened
        all_trades = all_trades[all_trades.index < death_date]
    trade_total = float(all_trades.sum())
    rets = equity.pct_change().fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    res = BacktestResult(equity, rets, dead, death_date, positions=held_df)
    res.metrics = compute_metrics(rets, equity, annual_turnover=trade_total / years,
                                  exposure=float(held_df.abs().mean(axis=1).mean()))
    res.metrics.update({"dead": dead, "death_date": str(death_date.date()) if death_date is not None else None,
                        "final_equity": float(equity.iloc[-1])})
    return res


def periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    years = (index[-1] - index[0]).days / 365.25
    return (len(index) - 1) / years if years > 0 else 252.0


def compute_metrics(rets: pd.Series, equity: pd.Series | None = None, annual_turnover: float | None = None,
                    exposure: float | None = None) -> dict:
    """Return, Sharpe, Sortino, max drawdown, turnover (rf = 0)."""
    r = rets.astype(float)
    if equity is None:
        equity = (1 + r).cumprod()
    ppy = periods_per_year(r.index)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9) if len(r) > 1 else 1e-9
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    mu, sd = float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2))) if len(r) else 0.0
    sharpe = mu / sd * np.sqrt(ppy) if sd > 1e-12 else 0.0
    sortino = mu / downside * np.sqrt(ppy) if downside > 1e-12 else 0.0
    mdd = float((1 - equity / equity.cummax()).max())
    out = {"total_return": total, "cagr": cagr, "ann_vol": sd * np.sqrt(ppy), "sharpe": float(sharpe),
           "sortino": float(sortino), "max_drawdown": mdd, "calmar": cagr / mdd if mdd > 1e-9 else 0.0,
           "n_days": int(len(r)), "years": round(years, 2)}
    if annual_turnover is not None:
        out["annual_turnover"] = float(annual_turnover)
    if exposure is not None:
        out["avg_exposure"] = exposure
    return out


def window_sleeve(asset, start: pd.Timestamp, end: pd.Timestamp, pos_fn) -> Sleeve | None:
    """Build a Sleeve for days in [start, end]. pos_fn(dates) -> positions for decision days."""
    idx = asset.features.index
    days = idx[(idx >= start) & (idx <= end)]
    if len(days) < 3:
        return None
    dec, real = days[:-1], days[1:]
    fwd = asset.fwd_ret.loc[dec].values
    # fwd_ret is computed on the asset's own calendar, so fwd[i] is exactly close(dec[i]) -> close(real[i])
    pos = np.asarray(pos_fn(dec), dtype=float)
    return Sleeve(asset.ticker, dec, real, pos, fwd)
