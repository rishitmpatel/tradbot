"""Round 3: rule-based strategies as first-class candidates (no learning, no training).

Two kinds of rule:
  * per-asset rules (any function in baselines.py, e.g. vol_target): each asset gets
    its own sleeve of capital, exactly as in the baseline tables of rounds 1-2.
  * portfolio rules (below): decide WEIGHTS of total equity across all assets at once,
    so they can take correlations into account. Evaluated with backtest.weights_backtest.

A rule config (configs/*.yaml) looks like:
    kind: rule
    rule:
      name: vol_target                       # or portfolio_vol_target
      params: {target: 0.12, window: 20, band: 0.1}
      neighborhood: [{target: 0.09}, ...]    # robustness only - NEVER used to pick params

Everything uses prices up to and including day t only (selfcheck.py verifies).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestResult, cost_rate, portfolio_backtest, weights_backtest, window_sleeve
from baselines import BASELINES, apply_band, baseline_positions


def is_rule(cfg: dict) -> bool:
    return cfg.get("kind") == "rule"


# --------------------------------------------------------------------------
# Portfolio rule: inverse-volatility weights + portfolio-level volatility target
# --------------------------------------------------------------------------
def close_panel(assets: dict) -> pd.DataFrame:
    """Closes of all assets on the union calendar (a closed market keeps its last price)."""
    return pd.DataFrame({t: a.close for t, a in assets.items()}).sort_index().ffill()


def portfolio_vol_target_weights(panel: pd.DataFrame, target: float = 0.08, window: int = 60,
                                 max_gross: float = 1.0, band: float = 0.02, **_) -> pd.DataFrame:
    """1) each asset gets weight proportional to 1 / its volatility (risk parity, ignoring correlation)
       2) the whole portfolio is scaled so its PREDICTED volatility - computed with the rolling
          covariance matrix, i.e. INCLUDING correlations - equals `target`
       3) never more than 100% of equity invested in total (no leverage), long only.
    Why: per-asset vol targeting (round 2's best rule) ignores that assets can fall together
    (stocks and bonds in 2022). This rule cuts total risk when correlations rise."""
    r = np.log(panel).diff()
    vol = r.rolling(window, min_periods=window).std() * np.sqrt(252)
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)
    cols = list(panel.columns)
    k = len(cols)
    # rolling covariance matrix for every day: shape (days, k, k), annualised
    cov = (r.rolling(window, min_periods=window).cov() * 252).values.reshape(len(panel), k, k)
    out = np.zeros((len(panel), k))
    wv = w.values
    for i in range(len(panel)):
        wi = wv[i]
        if not np.all(np.isfinite(wi)):
            continue
        c = cov[i]
        if not np.all(np.isfinite(c)):
            continue
        pvol = float(np.sqrt(max(wi @ c @ wi, 1e-12)))
        out[i] = wi * min(max_gross, target / pvol)
    W = pd.DataFrame(out, index=panel.index, columns=cols)
    W = W.apply(lambda col: pd.Series(apply_band(col.values, band), index=col.index))
    # the band keeps some old weights while others move, which can push the total above
    # max_gross: scale those days back down (selfcheck.py caught this)
    gross = W.sum(axis=1)
    return W.div(np.maximum(gross / max_gross, 1.0), axis=0)


PORTFOLIO_RULES = {"portfolio_vol_target": portfolio_vol_target_weights}


def rule_params(cfg: dict, override: dict | None = None) -> dict:
    return {**(cfg["rule"].get("params") or {}), **(override or {})}


def evaluate_window(cfg: dict, assets: dict, start: pd.Timestamp, end: pd.Timestamp, cost_mult: float = 1.0,
                    override: dict | None = None, _cache: dict | None = None) -> BacktestResult:
    """Trade the rule over [start, end] with fresh capital and the survival rule."""
    name = cfg["rule"]["name"]
    params = rule_params(cfg, override)
    rate = cost_rate(cfg, cost_mult)
    initial, max_dd = float(cfg["capital"]["initial"]), float(cfg["survival"]["max_drawdown"])
    if name in BASELINES:
        sleeves = []
        for a in assets.values():
            full = baseline_positions(name, a.close, params)
            sl = window_sleeve(a, start, end, lambda d, fp=full: fp.reindex(d).fillna(0.0).values)
            if sl is not None:
                sleeves.append(sl)
        return portfolio_backtest(sleeves, rate, initial, max_dd)
    if name not in PORTFOLIO_RULES:
        raise ValueError(f"unknown rule '{name}'")
    key = json.dumps(params, sort_keys=True)
    if _cache is not None and key in _cache:
        panel, W = _cache[key]
    else:
        panel = close_panel(assets)
        W = PORTFOLIO_RULES[name](panel, **params)
        if _cache is not None:
            _cache[key] = (panel, W)
    days = panel.index[(panel.index >= start) & (panel.index <= end)]
    dec, real = days[:-1], days[1:]
    fwd = (panel.shift(-1) / panel - 1.0).loc[dec].values
    tradable = np.column_stack([dec.isin(a.close.index) for a in assets.values()])
    return weights_backtest(dec, real, W.loc[dec].values, fwd, tradable, rate, initial, max_dd,
                            tickers=list(panel.columns))


def run_rule_strategy(cfg: dict, assets: dict, folds, runs_dir: Path, cost_mult: float = 1.0,
                      override: dict | None = None) -> list[dict]:
    """One deterministic 'run' per walk-forward validation window (seed 0)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    recs = []
    for f in folds:
        t0 = time.time()
        bt = evaluate_window(cfg, assets, pd.Timestamp(f.val_start), pd.Timestamp(f.val_end), cost_mult,
                             override, cache)
        csv = runs_dir / f"fold{f.k:02d}_seed0.csv"
        expo = bt.positions.abs().sum(axis=1) if cfg["rule"]["name"] in PORTFOLIO_RULES \
            else bt.positions.abs().mean(axis=1)
        pd.DataFrame({"equity": bt.equity, "ret": bt.returns, "exposure": expo}).to_csv(csv)
        recs.append({"status": "ok", "fold": asdict(f), "seed": 0, "metrics": bt.metrics, "csv": csv.name,
                     "_dir": str(runs_dir), "seconds": round(time.time() - t0, 2)})
    return recs
