"""Ken French Data Library loader (round 5). Free, no login, monthly/daily since 1926.

Files are zipped CSVs with a few text lines on top, then a block of MONTHLY (YYYYMM) or DAILY
(YYYYMMDD) rows in percent, then other blocks (annual, equal-weighted, ...). We read only
the FIRST date block: for portfolio files that is "Average Value Weighted Returns -- Monthly".
Missing values are coded -99.99 or -999 and become NaN.

Cache: data/french/<file>.csv (the extracted CSV). If a download fails, the error says
exactly which file to fetch by hand and where to put it.
"""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from config_utils import ROOT

log = logging.getLogger(__name__)
CACHE = ROOT / "data" / "french"
LIBRARY_PAGE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"


class DataUnavailable(RuntimeError):
    pass


def parse_french_csv(text: str) -> pd.DataFrame:
    """First block of dated rows -> DataFrame (index = month-end or day), values as decimals."""
    lines = text.replace("\r", "").split("\n")
    start, header, width = None, None, None
    for i, line in enumerate(lines[:-1]):
        parts = [p.strip() for p in line.split(",")]
        nxt = lines[i + 1].split(",")[0].strip()
        if len(parts) >= 2 and parts[0] == "" and any(parts[1:]) and nxt.isdigit() and len(nxt) in (6, 8):
            start, header, width = i + 1, parts[1:], len(nxt)
            break
    if start is None:
        raise ValueError("no dated data block found (file format changed?)")
    rows = []
    for line in lines[start:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit() or len(parts[0]) != width:
            break
        rows.append(parts[: len(header) + 1])
    df = pd.DataFrame([r[1:] for r in rows], columns=header).apply(pd.to_numeric, errors="coerce")
    df = df.mask((df <= -99.99 + 1e-9) | (df == -999)) / 100.0
    keys = [r[0] for r in rows]
    if width == 6:
        idx = pd.PeriodIndex([f"{k[:4]}-{k[4:]}" for k in keys], freq="M").to_timestamp(how="end").normalize()
    else:
        idx = pd.to_datetime(keys, format="%Y%m%d")
    df.index = idx
    df.columns = [c.strip() for c in df.columns]
    return df


def _norm(c: str) -> str:
    return c.upper().replace(" ", "").replace("_", "")


def fetch(base_url: str, filename: str) -> pd.DataFrame:
    """Parsed first block of one French file (download once, then cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / filename.replace(".zip", ".csv")
    if not cached.exists():
        url = base_url + filename
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
                cached.write_bytes(z.read(name))
        except Exception as e:
            raise DataUnavailable(
                f"Could not download {url} ({type(e).__name__}: {e}). Fix: open {LIBRARY_PAGE}, download "
                f"'{filename}', unzip it and save the CSV as {cached}") from e
    return parse_french_csv(cached.read_text(encoding="latin-1"))


def load_market(spec: dict, base_url: str, cutoff: pd.Timestamp) -> dict:
    """{'monthly': mkt, rf, big_momentum, big_value; 'daily': mkt, rf} strictly before `cutoff`."""
    fm = fetch(base_url, spec["factors_monthly"])
    cols = {_norm(c): c for c in fm.columns}
    monthly = pd.DataFrame({"market": fm[cols["MKT-RF"]] + fm[cols["RF"]], "rf": fm[cols["RF"]]})
    for key, want in (("size_momentum_monthly", "BIGHIPRIOR"), ("size_value_monthly", "BIGHIBM")):
        p = fetch(base_url, spec[key])
        match = [c for c in p.columns if _norm(c) == want]
        if not match:
            raise DataUnavailable(f"{spec[key]}: column like '{want}' not found (columns: {list(p.columns)})")
        monthly["big_momentum" if "PRIOR" in want else "big_value"] = p[match[0]]
    fd = fetch(base_url, spec["factors_daily"])
    dcols = {_norm(c): c for c in fd.columns}
    daily = pd.DataFrame({"mkt": fd[dcols["MKT-RF"]] + fd[dcols["RF"]], "rf": fd[dcols["RF"]]})
    monthly = monthly[monthly.index < cutoff]
    daily = daily[daily.index < cutoff]
    assert monthly.index.max() < cutoff and daily.index.max() < cutoff, "data beyond the cut-off"
    return {"monthly": monthly, "daily": daily}


def synthetic_market(seed: int, start: str = "1926-07-01", end: str = "2024-09-30", tom_bump: float = 0.0004) -> dict:
    """Fake data with the same shape (code testing only)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    rf_d = np.full(len(days), 0.03 / 252)
    mkt_d = rf_d + rng.normal(0.0003, 0.011, len(days))
    m = days.to_period("M")
    first = np.r_[True, m[1:] != m[:-1]]
    last = np.r_[m[1:] != m[:-1], True]
    tom = last | first | np.r_[False, first[:-1]] | np.r_[False, False, first[:-2]]
    mkt_d = mkt_d + tom * tom_bump
    daily = pd.DataFrame({"mkt": mkt_d, "rf": rf_d}, index=days)
    g = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    g.index = g.index.to_timestamp(how="end").normalize()
    n = len(g)
    monthly = pd.DataFrame({"market": g["mkt"].values, "rf": g["rf"].values}, index=g.index)
    monthly["big_momentum"] = monthly["market"] + rng.normal(0.002, 0.02, n)
    monthly["big_value"] = monthly["market"] + rng.normal(0.0015, 0.025, n)
    return {"monthly": monthly, "daily": daily}
