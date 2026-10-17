"""Daily market data from Yahoo Finance (via yfinance), cached to CSV.

Run directly to download/refresh the cache and print a data-quality summary:
    python data.py             # download missing tickers only
    python data.py --refresh   # re-download everything

The TEST SPLIT is locked the first time data is loaded: the last
`data.test_years` years become the test set and that start date is written to
data/raw/split_lock_<source>.json. Re-downloading later never moves it, so
new data can never leak old test days into development.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config_utils import ROOT, load_base

log = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_").replace("=", "_")
    return cache_dir / f"{safe}.csv"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].astype(float)


def download_ticker(ticker: str, start: str, end: str | None = None, retries: int = 4) -> pd.DataFrame:
    """Adjusted (splits + dividends) daily OHLCV from Yahoo. Retries with backoff."""
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
            if df is None or df.empty:
                # Fallback API
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            if df is None or df.empty:
                raise ValueError("empty dataframe returned")
            return _clean(df)
        except Exception as e:  # network hiccups, rate limits
            last_err = e
            wait = 2 ** (attempt + 1)
            log.warning("download %s failed (%s), retry in %ss", ticker, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Could not download {ticker}: {last_err}")


def load_prices(cfg: dict, tickers: list[str] | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """dict ticker -> OHLCV DataFrame. Uses CSV cache when available."""
    dcfg = cfg["data"]
    tickers = list(tickers or dcfg["tickers"])
    if dcfg.get("source", "yahoo") == "synthetic":
        from synthetic_data import generate_prices
        return generate_prices(tickers, dcfg["start"], None, int(dcfg.get("synthetic_seed", 7)))

    cache_dir = ROOT / dcfg["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in tickers:
        p = _cache_path(cache_dir, t)
        if p.exists() and not refresh:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            out[t] = _clean(df)
            continue
        log.info("downloading %s ...", t)
        df = download_ticker(t, dcfg["start"], dcfg.get("end"))
        df.to_csv(p)
        out[t] = df
        log.info("  %s: %d rows %s -> %s", t, len(df), df.index[0].date(), df.index[-1].date())
    return out


def quality_report(prices: dict[str, pd.DataFrame], jump_threshold: float = 0.25) -> list[dict]:
    """Per-ticker summary + suspicious jumps (possible bad split adjustment)."""
    rows = []
    for t, df in prices.items():
        lr = np.log(df["close"]).diff().dropna()
        big = lr[lr.abs() > np.log1p(jump_threshold)]
        gaps = df.index.to_series().diff().dt.days
        rows.append({
            "ticker": t, "rows": len(df),
            "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
            "years": round((df.index[-1] - df.index[0]).days / 365.25, 1),
            "big_jumps": [f"{d.date()}:{v:+.2f}" for d, v in big.items()][:10],
            "max_gap_days": int(gaps.max()) if len(gaps) > 1 else 0,
        })
    return rows


# --------------------------------------------------------------------------
# Locked train/test split
# --------------------------------------------------------------------------
@dataclass
class Split:
    test_start: pd.Timestamp   # first day of the locked test set
    data_end: pd.Timestamp     # last day of the test set


def get_split(cfg: dict, prices: dict[str, pd.DataFrame]) -> Split:
    source = cfg["data"].get("source", "yahoo")
    lock = ROOT / cfg["data"]["cache_dir"] / f"split_lock_{source}.json"
    if source != "synthetic" and lock.exists():
        d = json.loads(lock.read_text(encoding="utf-8"))
        return Split(pd.Timestamp(d["test_start"]), pd.Timestamp(d["data_end"]))
    data_end = min(df.index[-1] for df in prices.values())  # last day all assets have
    test_start = (data_end - pd.DateOffset(years=int(cfg["data"]["test_years"]))).normalize()
    split = Split(test_start, data_end)
    if source != "synthetic":
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"test_start": str(test_start.date()), "data_end": str(data_end.date()),
                                    "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                                    "note": "Do not edit. Days >= test_start are only for final_test.py"},
                                   indent=2), encoding="utf-8")
        log.info("Locked test split: test_start=%s data_end=%s", test_start.date(), data_end.date())
    return split


def dev_view(prices: dict[str, pd.DataFrame], split: Split) -> dict[str, pd.DataFrame]:
    """Development data = strictly before the test set. Walk-forward only ever sees this."""
    return {t: df[df.index < split.test_start] for t, df in prices.items()}


def test_view(prices: dict[str, pd.DataFrame], split: Split) -> dict[str, pd.DataFrame]:
    """Everything up to the locked data_end (history needed for feature warm-up)."""
    return {t: df[df.index <= split.data_end] for t, df in prices.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--extra", action="store_true", help="also download robustness.other_assets")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_base()
    prices = load_prices(cfg, refresh=args.refresh)
    if args.extra:
        prices.update(load_prices(cfg, cfg["robustness"]["other_assets"], refresh=args.refresh))
    for r in quality_report(prices):
        print(r)
    split = get_split(cfg, {t: prices[t] for t in cfg["data"]["tickers"]})
    print(f"Test set locked: {split.test_start.date()} -> {split.data_end.date()}")


if __name__ == "__main__":
    main()
