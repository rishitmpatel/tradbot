"""Synthetic prices for TESTING THE CODE ONLY.

Results on synthetic data say nothing about real markets. The generator plants
a regime-switching process with weak autocorrelation so that the pipeline has
something to (maybe) learn; any "edge" found here is an artefact of this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _calendar(start: str, end: str, rng: np.random.Generator, n_holidays_per_year: int) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    keep = np.ones(len(days), dtype=bool)
    n_drop = int(len(days) / 252 * n_holidays_per_year)
    keep[rng.choice(len(days), size=n_drop, replace=False)] = False
    return days[keep]


def generate_prices(tickers: list[str], start: str = "2010-01-01", end: str | None = None,
                    seed: int = 7) -> dict[str, pd.DataFrame]:
    end = end or "2026-10-16"
    rng = np.random.default_rng(seed)
    # Two calendars so the multi-calendar code path is exercised (US vs India).
    cal_us = _calendar(start, end, rng, 9)
    cal_in = _calendar(start, end, rng, 14)
    full = pd.bdate_range(start, end)

    # Shared market regime: 0 = calm bull, 1 = stressed bear.
    regime = np.zeros(len(full), dtype=int)
    for i in range(1, len(full)):
        stay = 0.995 if regime[i - 1] == 0 else 0.97
        regime[i] = regime[i - 1] if rng.random() < stay else 1 - regime[i - 1]
    market = np.where(regime == 0, rng.normal(0.0004, 0.008, len(full)),
                      rng.normal(-0.0010, 0.020, len(full)))

    out = {}
    for j, t in enumerate(tickers):
        beta = rng.uniform(0.3, 1.3)
        idio_vol = rng.uniform(0.006, 0.015)
        phi = rng.uniform(-0.03, 0.08)  # weak return autocorrelation
        eps = rng.normal(0, idio_vol, len(full))
        r = np.zeros(len(full))
        for i in range(len(full)):
            prev = r[i - 1] if i else 0.0
            r[i] = beta * market[i] + eps[i] + phi * prev
        close = 100 * np.exp(np.cumsum(r))
        s = pd.Series(close, index=full)
        cal = cal_in if t.upper().endswith(".NS") else cal_us
        s = s.reindex(cal)
        df = pd.DataFrame({"open": s, "high": s * 1.005, "low": s * 0.995, "close": s,
                           "volume": 1_000_000.0})
        df.index.name = "date"
        out[t] = df
    return out
