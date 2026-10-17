"""Round 6 research stage: overnight effect, momentum crash guard, Emerging-Markets replication,
India tax harvesting (pre-registered in research/plan_round6.yaml).

    python overnight_research.py            (also called automatically by run_all.py)
    python overnight_research.py --synthetic

Overnight (daily, Yahoo open/high/low/close, split+dividend adjusted):
  * strategy return on day t = open_t / close_(t-1) - 1 - round-trip cost; the money is idle during
    the day (a retail account earns nothing on it). Benchmark = buy and hold, close to close.
  * open prices from free sources can be fake (= previous close) or impossible (outside the
    day's high/low). Every data set is checked; an UNRELIABLE one cannot confirm anything.
  * stress fills: on days the gap is <= -2% you sell at the day's LOW instead of the open
    (24 Aug 2015-type dislocations).
  * costs and Indian tax: US via LRS (assumed 2 bps per round trip, slab tax) and NIFTYBEES on NSE
    (every statutory charge, DP charge per sell day, assumed slippage, STCG 20% or slab 30%).
Momentum (monthly, Ken French): round 5's machinery (10% vol target, 15 bps switches, fund drag,
  15% kill switch per calendar year), plus a Daniel-Moskowitz bear-market guard.
Tax harvesting: buy and hold NIFTYBEES with vs without realising Rs 1.25 lakh of tax-free
  long-term gain each financial year (sell at the close, re-buy at the next day's close).

Exploration: US only. Confirmation (once, finalists only): NIFTYBEES (India) and Emerging Markets,
data this project never loaded before. The locked test set is never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config_utils import ROOT, load_base
from data import _cache_path, get_split, load_prices
from evaluation import base_checks, bootstrap_sharpe_diff, deflated_sharpe_prob, penalize
from factor_data import DataUnavailable, _norm, fetch, synthetic_market
from factor_research import _clean, compare, positions, run_monthly, spec_hash, stats
from reporting import save_json

log = logging.getLogger("overnight_research")
PLAN6 = ROOT / "research" / "plan_round6.yaml"
CAPITALS = (100_000, 1_000_000, 10_000_000)
OVERNIGHT_SUBPERIODS = [("1993", "2000"), ("2001", "2010"), ("2011", "2020"), ("2021", "2100")]


# --------------------------------------------------------------------------
# Plan, registration, ledger
# --------------------------------------------------------------------------
def load_plan(path: Path = PLAN6) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_trials(plan: dict) -> dict:
    return {**plan["hypotheses"], **(plan.get("diagnostics") or {})}


def register_round6(plan: dict, res_root: Path) -> tuple[dict, int]:
    """Freeze the registered list on the first run; add every trial to the ledger ('r6:' keys)."""
    rdir = res_root / "research6"
    rdir.mkdir(parents=True, exist_ok=True)
    reg_path = rdir / "registration.json"
    trials = all_trials(plan)
    hashes = {n: spec_hash(s) for n, s in trials.items()}
    if not reg_path.exists():
        save_json(reg_path, {"registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "plan_sha": hashlib.sha1(PLAN6.read_bytes()).hexdigest()[:12], "trials": hashes})
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["trials"]
    status = {}
    for n, h in hashes.items():
        if reg.get(n) != h:
            status[n] = "exploratory"
        else:
            status[n] = "diagnostic" if n in (plan.get("diagnostics") or {}) else "registered"
    lp = res_root / "trial_ledger.json"
    ledger = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {"trials": {}}
    for n, h in hashes.items():
        ledger["trials"].setdefault(f"r6:{n}@{h}", {"round": 6, "kind": status[n],
                                                    "first_seen": time.strftime("%Y-%m-%d %H:%M")})
    save_json(lp, ledger)
    return status, sum(1 for k in ledger["trials"] if k.startswith("r6:"))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def synthetic_ohlc(seed: int, start: str, end: str, overnight_mu: float = 0.0003, intraday_mu: float = 0.0,
                   fake_open: bool = False) -> pd.DataFrame:
    """Fake daily OHLC (code testing only). fake_open=True mimics a broken free feed (open = previous close)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    gap = rng.normal(overnight_mu, 0.006, len(days))
    intra = rng.normal(intraday_mu, 0.008, len(days))
    close = 100 * np.cumprod((1 + gap) * (1 + intra))
    prev = np.r_[100.0, close[:-1]]
    opn = prev * (1 + gap)
    hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.003, len(days))))
    lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.003, len(days))))
    if fake_open:
        opn = prev.copy()
        hi, lo = np.maximum(hi, opn), np.minimum(lo, opn)
    return pd.DataFrame({"open": opn, "high": hi, "low": lo, "close": close}, index=days)


def load_ohlc(plan: dict, tickers: list[str], cutoff: pd.Timestamp, synthetic: bool) -> dict:
    """{ticker: daily OHLC strictly before `cutoff`} (Yahoo, cached in data/ohlc)."""
    out = {}
    for i, t in enumerate(tickers):
        if synthetic:
            start = "1993-01-29" if not t.endswith(".NS") and not t.startswith("^") else "2009-01-01"
            df = synthetic_ohlc(60 + i, start, str((cutoff - pd.Timedelta(days=1)).date()), fake_open=t.startswith("^"))
        else:
            cfg = {"data": {"source": "yahoo", "cache_dir": plan["data"]["ohlc_cache_dir"],
                            "start": plan["data"]["ohlc_start"], "end": None, "tickers": [t]}}
            try:
                df = load_prices(cfg, [t])[t]
            except Exception as e:
                path = _cache_path(ROOT / plan["data"]["ohlc_cache_dir"], t)
                raise DataUnavailable(
                    f"Could not load daily prices for {t} from Yahoo ({type(e).__name__}: {e}). Fix: check the internet "
                    f"connection, wait 10-15 minutes (Yahoo rate limit) and re-run python run_all.py (it resumes). If it "
                    f"keeps failing, save a CSV with columns date,open,high,low,close (split- and dividend-adjusted) as "
                    f"{path}") from e
            missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
            if missing:
                raise DataUnavailable(f"{t}: Yahoo data has no {missing} column(s); cannot test open prices")
        df = df[df.index < cutoff]
        if len(df) < 500:
            raise DataUnavailable(f"{t}: only {len(df)} days before {cutoff.date()} - too little data")
        assert df.index.max() < cutoff, "data beyond the cut-off"
        out[t] = df
    return out


def us_daily_rf(plan: dict, cutoff: pd.Timestamp, synthetic: bool, index: pd.DatetimeIndex) -> pd.Series:
    if synthetic:
        return pd.Series(0.03 / 252, index=index)
    fd = fetch(plan["data"]["french_base_url"], plan["data"]["us_rf_daily"])
    rf = fd[{_norm(c): c for c in fd.columns}["RF"]]
    rf = rf[rf.index < cutoff]
    return rf.reindex(index, method="ffill").fillna(0.0)


def india_daily_rf(plan: dict, index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((1 + float(plan["data"]["india_rf_annual"])) ** (1 / 252) - 1, index=index)


def load_momentum(spec: dict, base_url: str, cutoff: pd.Timestamp, synthetic: bool, seed: int = 6) -> dict:
    """{'monthly': market, rf, big_momentum} strictly before `cutoff`."""
    if synthetic:
        start = "1926-07-01" if seed == 6 else "1989-07-01"
        d = synthetic_market(seed, start, str((cutoff - pd.Timedelta(days=1)).date()))
        return {"monthly": d["monthly"][["market", "rf", "big_momentum"]]}
    fm = fetch(base_url, spec["factors_monthly"])
    cols = {_norm(c): c for c in fm.columns}
    m = pd.DataFrame({"market": fm[cols["MKT-RF"]] + fm[cols["RF"]], "rf": fm[cols["RF"]]})
    p = fetch(base_url, spec["size_momentum_monthly"])
    match = [c for c in p.columns if _norm(c) == "BIGHIPRIOR"]
    if not match:
        raise DataUnavailable(f"{spec['size_momentum_monthly']}: column like 'BIG HiPRIOR' not found (columns: {list(p.columns)})")
    m["big_momentum"] = p[match[0]]
    m = m[m.index < cutoff].dropna()
    assert m.index.max() < cutoff, "data beyond the cut-off"
    return {"monthly": m}


# --------------------------------------------------------------------------
# Open-price reliability
# --------------------------------------------------------------------------
def open_quality(df: pd.DataFrame, th: dict) -> dict:
    prev = df["close"].shift(1)
    ok = prev.notna()
    o = df["open"]
    missing = float((o.isna() | (o <= 0))[ok].mean())
    eq = float(((o / prev - 1).abs() < 1e-6)[ok].mean())
    outside = float(((o > df["high"] * (1 + 1e-6)) | (o < df["low"] * (1 - 1e-6)))[ok].mean())
    gap = (o / prev - 1)[ok].dropna()
    top = gap.reindex(gap.abs().sort_values(ascending=False).index[:5])
    cc = (df["close"] / prev - 1)[ok].dropna()
    jumps = cc[cc.abs() > 0.25]                       # a broad index fund never moves 25% in a day: bad split adjustment
    reliable = (eq <= th["max_share_open_eq_prev_close"] and outside <= th["max_share_open_outside_range"]
                and missing <= th["max_share_missing"] and len(jumps) == 0)
    return {"days": int(ok.sum()), "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
            "share_open_eq_prev_close": eq, "share_open_outside_range": outside, "share_missing": missing,
            "largest_gaps": [f"{d.date()}:{v:+.1%}" for d, v in top.items()],
            "jumps": [f"{d.date()}:{v:+.1%}" for d, v in jumps.items()][:5], "reliable": bool(reliable)}


# --------------------------------------------------------------------------
# Daily returns, costs, engine
# --------------------------------------------------------------------------
def india_round_trip(c: dict, capital: float) -> float:
    """Fraction of the traded amount lost per buy + sell on NSE (charges + DP + assumed slippage)."""
    per_side = (float(c["both_pct"]) + float(c["sebi_pct"])) * (1 + float(c["gst"]))
    return (float(c["buy_pct"]) + float(c["sell_pct"]) + 2 * per_side + 2 * float(c["slippage_bps_per_side"]) / 1e4
            + float(c["dp_per_sell_rs"]) / capital)


def day_returns(df: pd.DataFrame, kind: str, cost: float = 0.0, stress_gap: float | None = None) -> pd.Series:
    """Daily return of a position decided BEFORE the period starts (no signal: pure calendar timing).
    overnight: close(t-1) -> open(t); intraday: open(t) -> close(t); buy_and_hold: close -> close."""
    prev = df["close"].shift(1)
    if kind == "overnight":
        r = df["open"] / prev - 1
        if stress_gap is not None:
            r = r.where(~(r <= stress_gap), df["low"] / prev - 1)    # sell at the day's low on big gap-downs
    elif kind == "intraday":
        r = df["close"] / df["open"] - 1
    elif kind == "buy_and_hold":
        return (df["close"] / prev - 1).dropna()
    else:
        raise ValueError(kind)
    return (r - cost).dropna()


def run_daily(rets: pd.Series, kill: float | None, start, end, single_window: bool = False) -> tuple[pd.Series, list[dict]]:
    """Calendar-year windows with fresh 100,000; dead (flat for the rest of the year) below (1-kill) x peak."""
    rets = rets[(rets.index >= pd.Timestamp(start)) & (rets.index <= pd.Timestamp(end))]
    out, wins = [], []
    groups = [(0, rets)] if single_window else rets.groupby(rets.index.year)
    for year, r in groups:
        if len(r) < 60 and not single_window:
            continue
        eq, peak, dead = 1e5, 1e5, False
        vals = []
        for x in r.values:
            if dead:
                vals.append(0.0)
                continue
            new = eq * (1 + x)
            peak = max(peak, new)
            vals.append(new / eq - 1)
            eq = new
            if kill is not None and 1 - eq / peak > kill:
                dead = True
        out.append(pd.Series(vals, index=r.index))
        wins.append({"year": int(year), "return": eq / 1e5 - 1, "dead": dead})
    return (pd.concat(out) if out else pd.Series(dtype=float)), wins


def stats_daily(rets: pd.Series, rf: pd.Series, wins: list[dict]) -> dict:
    if len(rets) < 250:
        return {"n_ok": 0, "survived": 0, "complete": False, "survival_rate": 0.0}
    ex = rets - rf.reindex(rets.index).fillna(0.0)
    years = len(rets) / 252
    eq = (1 + rets).cumprod()
    sd = ex.std(ddof=1)
    x = ex.values
    alive = sum(1 for w in wins if not w["dead"])
    sd = sd if sd > 1e-10 else 0.0                    # a constant return (e.g. fake opens) has no Sharpe
    return {"mean_sharpe": float(ex.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            "cagr": float(eq.iloc[-1] ** (1 / years) - 1), "excess_cagr": float((1 + ex).prod() ** (1 / years) - 1),
            "mean_return": float(eq.iloc[-1] - 1), "ann_vol": float(rets.std(ddof=1) * np.sqrt(252)),
            "mean_stitched_mdd": float((1 - eq / eq.cummax()).max()),
            "survived": alive, "n_ok": len(wins), "n_runs": len(wins), "complete": True,
            "survival_rate": alive / len(wins) if wins else 0.0, "n_days": len(rets), "years": years,
            "skew": float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0,
            "kurt": float(((x - x.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0,
            "first": str(rets.index[0].date()), "last": str(rets.index[-1].date()), "_rets": rets, "_ex": ex}


def fy_of(idx: pd.DatetimeIndex) -> np.ndarray:
    """Indian financial year (April-March), labelled by the calendar year it starts in."""
    return np.where(idx.month >= 4, idx.year, idx.year - 1)


def after_tax_cagr(rets: pd.Series, capital: float, rate: float, cess: float) -> dict:
    """A strategy that realises everything daily: tax each financial year on the net gain, losses carried
    forward. One continuous run (no kill switch). Returns after-tax CAGR and total tax / start capital."""
    eq, carry, tax_total = float(capital), 0.0, 0.0
    for _, r in rets.groupby(fy_of(rets.index)):
        start = eq
        eq *= float(np.prod(1 + r.values))
        taxable = eq - start - carry
        if taxable > 0:
            tax = taxable * rate * (1 + cess)
            eq -= tax
            tax_total += tax
            carry = 0.0
        else:
            carry = -taxable
    years = len(rets) / 252
    return {"cagr": (eq / capital) ** (1 / years) - 1 if eq > 0 else -1.0, "tax_share": tax_total / capital}


def buy_hold_after_tax(rets: pd.Series, capital: float, rate: float, cess: float, exempt: float) -> dict:
    """Buy once, sell once at the end: long-term tax above the yearly exemption."""
    final = capital * float(np.prod(1 + rets.values))
    tax = max(0.0, final - capital - exempt) * rate * (1 + cess)
    years = len(rets) / 252
    return {"cagr": ((final - tax) / capital) ** (1 / years) - 1, "tax_share": tax / capital}


def compare_daily(s: dict, b: dict) -> tuple[float, float]:
    a, c = s["_ex"].align(b["_ex"], join="inner")
    return s["mean_sharpe"] - b["mean_sharpe"], bootstrap_sharpe_diff(a.values, c.values, block=21)


def decomposition(df: pd.DataFrame, periods=OVERNIGHT_SUBPERIODS) -> list[dict]:
    """Gross average yearly return from close->open, open->close and close->close, by sub-period."""
    rows = []
    on, intra, cc = day_returns(df, "overnight"), day_returns(df, "intraday"), day_returns(df, "buy_and_hold")
    for lo, hi in periods + [("all", "all")]:
        sel = (lambda s: s) if lo == "all" else (lambda s, lo=lo, hi=hi: s[(s.index >= lo) & (s.index < str(int(hi) + 1))])
        o, i, c = sel(on), sel(intra), sel(cc)
        if len(o) < 120:
            continue
        rows.append({"period": "all" if lo == "all" else f"{lo}-{hi if hi < '2100' else 'now'}",
                     "overnight": float(o.mean() * 252), "intraday": float(i.mean() * 252), "close_to_close": float(c.mean() * 252),
                     "overnight_bps_per_day": float(o.mean() * 1e4), "days": len(o)})
    return rows


def eval_overnight(df: pd.DataFrame, rf: pd.Series, kind: str, cost: float, kill: float, start, end,
                   stress_gap: float | None = None) -> dict:
    rets, wins = run_daily(day_returns(df, kind, cost, stress_gap), kill, start, end)
    return stats_daily(rets, rf, wins)


# --------------------------------------------------------------------------
# Momentum (monthly)
# --------------------------------------------------------------------------
def mom_sleeve(spec: dict, common: dict, m: pd.DataFrame, cost_mult: float = 1.0) -> pd.Series:
    """Monthly sleeve return at full exposure. With a guard: market instead of momentum whenever the
    market's trailing `lookback_months` return was below T-bills at the PREVIOUS month end."""
    drag = common["sleeve_drag_bps_per_year"]
    mkt = m["market"] - drag["market"] / 1e4 / 12 * cost_mult
    if spec["sleeve"] == "market":
        return mkt
    mom = m["big_momentum"] - drag["big_momentum"] / 1e4 / 12 * cost_mult
    if not spec.get("guard"):
        return mom
    bear = guard_state(spec, m)
    s = pd.Series(np.where(bear == 1, mkt, np.where(bear == 0, mom, np.nan)), index=m.index)
    switch = bear.diff().abs().fillna(0.0)
    return s - switch * 2 * float(common["switch_cost_bps"]) / 1e4 * cost_mult     # sell one sleeve, buy the other


def guard_state(spec: dict, m: pd.DataFrame) -> pd.Series:
    """1 = hold the market this month (its trailing return was below T-bills at the PREVIOUS month end),
    0 = hold momentum, NaN = not enough history yet."""
    L = int(spec["guard"]["lookback_months"])
    mk = (1 + m["market"]).rolling(L, min_periods=L).apply(np.prod, raw=True)
    rf = (1 + m["rf"]).rolling(L, min_periods=L).apply(np.prod, raw=True)
    return (mk < rf).astype(float).where(mk.notna()).shift(1)


def guard_share(spec: dict, m: pd.DataFrame, start, end) -> float:
    if not spec.get("guard"):
        return 0.0
    bear = guard_state(spec, m)
    b = bear[(bear.index >= pd.Timestamp(start)) & (bear.index <= end)].dropna()
    return float(b.mean()) if len(b) else float("nan")


def eval_mom(spec: dict, common: dict, data: dict, start, end, kill: bool = True, cost_mult: float = 1.0) -> dict:
    m = data["monthly"]
    sl = mom_sleeve(spec, common, m, cost_mult)
    pos = positions({}, common, data, sl)
    rets, wins = run_monthly(sl, m["rf"], pos, float(common["switch_cost_bps"]) / 1e4 * cost_mult,
                             float(common["kill_switch"]) if kill else None, str(start), end)
    out = stats(rets, m["rf"], wins)
    out["avg_exposure"] = float(pos.reindex(rets.index).mean()) if len(rets) else float("nan")
    return out


MOM_SUBPERIODS = [("1963-07", "1990-12"), ("1991-01", "2006-12"), ("2007-01", "2100-01")]
CROWDING_EPISODES = [("Aug 2007 quant quake", "2007-08", "2007-08"), ("2009 rebound (Mar-May)", "2009-03", "2009-05"),
                     ("Nov 2016 value rotation", "2016-11", "2016-11"), ("Nov 2020 vaccine rally", "2020-11", "2020-11"),
                     ("2022 rotation (Jan-Feb)", "2022-01", "2022-02")]


def crowding_checks(m: pd.DataFrame, end) -> dict:
    """Momentum minus market (gross, full exposure) in known crowding/crash episodes + its worst months."""
    rel = (m["big_momentum"] - m["market"])[m.index <= end]
    ep = {}
    for label, lo, hi in CROWDING_EPISODES:
        s = rel[(rel.index >= pd.Timestamp(lo)) & (rel.index <= pd.Timestamp(hi) + pd.offsets.MonthEnd(0))]
        ep[label] = float((1 + s).prod() - 1) if len(s) else float("nan")
    worst = rel[rel.index >= "1963-07"].nsmallest(5)
    return {"episodes": ep, "worst_months": [f"{d.strftime('%Y-%m')}:{v:+.1%}" for d, v in worst.items()]}


# --------------------------------------------------------------------------
# India tax harvesting
# --------------------------------------------------------------------------
def tax_harvest_sim(close: pd.Series, capital: float, harvest: bool, c: dict, exempt: float = 125_000.0,
                    lt_rate: float = 0.125, st_rate: float = 0.20, cess: float = 0.04) -> dict:
    """Buy and hold an Indian equity ETF (FIFO lots), optionally realising up to `exempt` of long-term gain
    each financial year in Jan-Mar (sell at the close, re-buy at the next day's close). Everything is sold
    on the last day and taxed (so both versions pay their full tax). Rules: current (FY2025-26) rates."""
    side = (float(c["both_pct"]) + float(c["sebi_pct"])) * (1 + float(c["gst"])) + float(c["slippage_bps_per_side"]) / 1e4
    buy_f, sell_f, dp = float(c["buy_pct"]) + side, float(c["sell_pct"]) + side, float(c["dp_per_sell_rs"])
    px = close.dropna()
    u0 = capital * (1 - buy_f) / px.iloc[0]
    lots = [[u0, capital / u0, px.index[0]]]          # [units, cost per unit incl. buy charges, purchase day]
    charges, tax_paid, pending, harvests = capital * buy_f, 0.0, 0.0, 0
    realised = {}                                       # fy -> [short-term, long-term]
    fys = [int(x) for x in fy_of(px.index)]

    def sell(units_wanted, p, t, fy):
        nonlocal charges
        got, units = 0.0, units_wanted
        while units > 1e-12 and lots:
            lot = lots[0]
            take = min(units, lot[0])
            net = take * p * (1 - sell_f)
            r = realised.setdefault(fy, [0.0, 0.0])
            r[1 if (t - lot[2]).days > 365 else 0] += net - take * lot[1]
            charges += take * p * sell_f
            got += net
            lot[0] -= take
            units -= take
            if lot[0] <= 1e-12:
                lots.pop(0)
        return got

    def settle(fy):
        st, lt = realised.pop(fy, [0.0, 0.0])
        if st < 0:
            lt, st = lt + st, 0.0
        return (max(0.0, st) * st_rate + max(0.0, lt - exempt) * lt_rate) * (1 + cess)

    for i, (t, p) in enumerate(px.items()):
        if pending > 0:                                 # re-buy the day after a harvest
            u = pending * (1 - buy_f) / p
            lots.append([u, pending / u, t])
            charges += pending * buy_f
            pending = 0.0
        fy = fys[i]
        last = i == len(px) - 1
        if harvest and not last and t.month in (1, 2, 3):
            room = exempt - realised.get(fy, [0.0, 0.0])[1]
            units = 0.0
            for lot in lots:                            # FIFO: only the oldest lots can be sold first
                if (t - lot[2]).days <= 365 or room <= 1000:
                    break
                g = p * (1 - sell_f) - lot[1]
                if g <= 0:
                    break
                take = min(lot[0], room / g)
                units += take
                room -= take * g
                if take < lot[0]:
                    break
            if units > 0:
                pending = sell(units, p, t, fy) - dp
                charges += dp
                harvests += 1
        if last:
            final = sell(sum(lot[0] for lot in lots), p, t, fy) - dp
            charges += dp
            tax = settle(fy)
            tax_paid += tax
            final -= tax
        elif fys[i + 1] != fy and fy in realised:
            tax_paid += settle(fy)                      # paid from your pocket (stays ~0 by construction)
    years = len(px) / 252
    return {"final": final, "cagr": (final / capital) ** (1 / years) - 1, "tax_share": tax_paid / capital,
            "charges_share": charges / capital, "harvests": harvests, "years": years}


# --------------------------------------------------------------------------
# Main stage
# --------------------------------------------------------------------------
def _cutoffs(base: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    split = get_split(base, load_prices(base))
    return split.test_start, pd.Timestamp(split.test_start.year, split.test_start.month, 1)


def run_round6(base: dict, res_root: Path, synthetic: bool, other_trials: int, run_confirmation: bool) -> dict:
    """other_trials = every trial counted outside round 6 (rounds 1-5)."""
    plan = load_plan()
    status, n_r6 = register_round6(plan, res_root)
    n_total = other_trials + n_r6
    rdir = res_root / "research6"
    day_cut, month_cut = _cutoffs(base)
    day_end, month_end = day_cut - pd.Timedelta(days=1), month_cut - pd.Timedelta(days=1)
    ov, hyps, common = plan["overnight"], plan["hypotheses"], plan["momentum_common"]
    kill = float(ov["kill_switch"])
    pc = base["pass_criteria"]
    log.info("round 6: %d trials (%s); total trials = %d + %d = %d", n_r6, status, other_trials, n_r6, n_total)

    # ---- exploration: overnight (US) ----
    ex_ticker = plan["data"]["overnight_exploration"][0]
    errors, quality = {}, {}
    us = load_ohlc(plan, [ex_ticker], day_cut, synthetic)[ex_ticker]
    quality[ex_ticker] = open_quality(us, ov["reliability"])
    rf_us = us_daily_rf(plan, day_cut, synthetic, us.index)
    c_us = float(ov["costs"]["us_lrs"]["round_trip_bps"]) / 1e4
    tax_us = ov["costs"]["us_lrs"]["tax"]
    st0 = plan["periods"]["overnight_exploration"]["start"]
    summ = {}
    summ["bench:buy_and_hold_spy"] = eval_overnight(us, rf_us, "buy_and_hold", 0.0, kill, st0, day_end)
    b_on = summ["bench:buy_and_hold_spy"]
    for n, spec in all_trials(plan).items():
        if spec.get("family") not in ("overnight", "intraday"):
            continue
        s = eval_overnight(us, rf_us, spec["family"], c_us, kill, st0, day_end)
        s["family"] = spec["family"]
        s2 = eval_overnight(us, rf_us, spec["family"], 2 * c_us, kill, st0, day_end)
        s["costs_2x_sharpe"], s["costs_2x_excess_cagr"] = s2.get("mean_sharpe"), s2.get("excess_cagr")
        s["sharpe_minus_bench"], s["p_vs_bench"] = compare_daily(s, b_on)
        if spec["family"] == "overnight":
            st = eval_overnight(us, rf_us, "overnight", c_us, kill, st0, day_end, float(ov["stress_gap"]))
            s["stress_sharpe"], s["stress_cagr"] = st.get("mean_sharpe"), st.get("cagr")
            nk = eval_overnight(us, rf_us, "overnight", c_us, None, st0, day_end)
            s["nokill_sharpe"], s["nokill_cagr"] = nk.get("mean_sharpe"), nk.get("cagr")
            if "_rets" in nk:
                s["after_tax_cagr"] = after_tax_cagr(nk["_rets"], 1e6, float(tax_us["rate"]), float(tax_us["cess"]))["cagr"]
        g = day_returns(us[us.index >= st0], spec["family"])
        cc = day_returns(us[us.index >= st0], "buy_and_hold")
        rfm = rf_us.reindex(g.index).fillna(0.0)
        s["breakeven_bps_vs_tbills"] = float((g - rfm).mean() * 1e4)
        s["breakeven_bps_vs_bh"] = float((g.mean() - cc.reindex(g.index).mean()) * 1e4)
        summ[n] = s
        log.info("exploration %-28s excess Sharpe %.2f (B&H %.2f) CAGR %+.1f%% survived %d/%d", n,
                 s.get("mean_sharpe", float("nan")), b_on.get("mean_sharpe", float("nan")),
                 100 * s.get("cagr", float("nan")), s["survived"], s["n_ok"])
    nkb = eval_overnight(us, rf_us, "buy_and_hold", 0.0, None, st0, day_end)
    b_on["nokill_sharpe"], b_on["nokill_cagr"] = nkb.get("mean_sharpe"), nkb.get("cagr")
    if "_rets" in nkb:
        b_on["after_tax_cagr"] = buy_hold_after_tax(nkb["_rets"], 1e6, 0.125, 0.04, 0.0)["cagr"]
    decomp = {ex_ticker: decomposition(us[us.index >= st0])}

    # ---- exploration: momentum (US) ----
    fb = plan["data"]["french_base_url"]
    mus = load_momentum(plan["data"]["momentum_exploration"], fb, month_cut, synthetic)
    mst = plan["periods"]["momentum_exploration"]["start"]
    bench_m = {"sleeve": "market"}
    summ["bench:market_10vol"] = eval_mom(bench_m, common, mus, mst, month_end)
    b_m = summ["bench:market_10vol"]
    b_m["nokill_sharpe"] = eval_mom(bench_m, common, mus, mst, month_end, kill=False).get("mean_sharpe")
    for n, spec in hyps.items():
        if spec.get("family") != "momentum":
            continue
        s = eval_mom(spec, common, mus, mst, month_end)
        s["family"] = "momentum"
        s["sharpe_minus_bench"], s["p_vs_bench"] = compare(s, b_m)
        c2 = eval_mom(spec, common, mus, mst, month_end, cost_mult=2.0)
        s["costs_2x_sharpe"], s["costs_2x_excess_cagr"] = c2.get("mean_sharpe"), c2.get("excess_cagr")
        s["nokill_sharpe"] = eval_mom(spec, common, mus, mst, month_end, kill=False).get("mean_sharpe")
        s["guard_share"] = guard_share(spec, mus["monthly"], mst, month_end)
        s["subperiods"] = {}
        for lo, hi in MOM_SUBPERIODS:
            e = min(pd.Timestamp(hi), month_end)
            a, bb = eval_mom(spec, common, mus, max(lo, mst), e), eval_mom(bench_m, common, mus, max(lo, mst), e)
            s["subperiods"][f"{lo[:4]}-{hi[:4] if hi < '2100' else 'now'}"] = a.get("mean_sharpe", np.nan) - bb.get("mean_sharpe", np.nan)
        summ[n] = s
        log.info("exploration %-28s excess Sharpe %.2f (market %.2f) p %.3f", n, s.get("mean_sharpe", np.nan),
                 b_m.get("mean_sharpe", np.nan), s["p_vs_bench"])
    crowd = crowding_checks(mus["monthly"], month_end)

    # ---- penalty + pre-registered finalist rule ----
    cands = [n for n in hyps if hyps[n].get("family") in ("overnight", "momentum")]
    for n in cands:
        # cross-sectional spread only within a family: daily overnight and monthly momentum Sharpes are not comparable
        cross = [summ[k]["mean_sharpe"] for k in cands if "mean_sharpe" in summ[k] and hyps[k]["family"] == hyps[n]["family"]]
        penalize(summ[n], n_total, cross)
        summ[n]["status"] = status[n]
    fs = plan["finalist_selection"]
    eligible = [n for n in cands if summ[n].get("survival_rate", 0) >= 0.8 and (summ[n].get("costs_2x_excess_cagr") or -1) > 0
                and (summ[n].get("sharpe_minus_bench") or -1) > 0]
    finalists = sorted(eligible, key=lambda n: -summ[n]["sharpe_minus_bench"])[: int(fs["max_finalists"])]
    lock = rdir / "confirmation_LOCK.json"
    changed = None
    if lock.exists():
        locked = json.loads(lock.read_text(encoding="utf-8"))["finalists"]
        if locked != finalists:
            changed = f"FINALISTS CHANGED after confirmation ({locked} -> {finalists}). Using the locked ones."
            log.error(changed)
        finalists = locked
    save_json(rdir / "finalists.json", {"finalists": finalists, "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": fs})
    log.info("round 6 finalists: %s", finalists)

    # ---- India open-price data check (data quality only - no returns are computed unless H1 is a finalist) ----
    conf_tickers = list(plan["data"]["overnight_confirmation"])
    india = {}
    for t in conf_tickers:
        try:
            india[t] = load_ohlc(plan, [t], day_cut, synthetic)[t]
            quality[t] = open_quality(india[t], ov["reliability"])
        except Exception as e:
            errors[t] = str(e)
            log.error("data %s not loaded: %s", t, e)

    # ---- confirmation (once) ----
    conf = {"status": "blocked (research6.run_confirmation is false in base_config.yaml, or this is a smoke run)"}
    if run_confirmation and finalists:
        if not lock.exists():      # freeze the finalists BEFORE any confirmation return is computed
            save_json(lock, {"finalists": finalists, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        conf = {"status": "ok", "overnight": {}, "momentum": {}, "verdicts": {}}
        cr = plan["confirmation_criteria"]
        ci = ov["costs"]["india"]
        for n in finalists:
            spec, why_all = hyps[n], []
            if spec["family"] == "overnight":
                for t, tspec in plan["data"]["overnight_confirmation"].items():
                    req = bool(tspec.get("required"))
                    if t not in india:
                        conf["overnight"][t] = {"required": req, "loaded": False, "error": errors.get(t)}
                        if req:
                            why_all.append(f"{t}: NOT LOADED (required)")
                        continue
                    df = india[t]
                    st_c = max(pd.Timestamp(plan["periods"]["overnight_confirmation"]["start"]), df.index[0])
                    assert df.index.max() < day_cut, "confirmation overlaps the locked test set"
                    rf_in = india_daily_rf(plan, df.index)
                    cost = india_round_trip(ci, float(ci["capital_rs"]))
                    s = eval_overnight(df, rf_in, "overnight", cost, kill, st_c, day_end)
                    bh = eval_overnight(df, rf_in, "buy_and_hold", 0.0, kill, st_c, day_end)
                    row = {"required": req, "loaded": True, "summ": s, "bench": bh, "quality": quality[t],
                           "cost_bps_round_trip": {int(k): india_round_trip(ci, k) * 1e4 for k in CAPITALS},
                           "decomposition": decomposition(df[df.index >= st_c], [("2009", "2014"), ("2015", "2019"), ("2020", "2100")])}
                    why = []
                    if not quality[t]["reliable"]:
                        why.append("open prices UNRELIABLE")
                    if "mean_sharpe" in s and "mean_sharpe" in bh:
                        s["sharpe_minus_bench"], s["p_vs_bench"] = compare_daily(s, bh)
                        if s["mean_sharpe"] <= bh["mean_sharpe"]:
                            why.append(f"excess Sharpe {s['mean_sharpe']:.2f} <= buy & hold {bh['mean_sharpe']:.2f}")
                        if not s["p_vs_bench"] < float(cr["overnight"]["p_value_lt"]):
                            why.append(f"advantage not significant (p {s['p_vs_bench']:.2f})")
                        if s["survival_rate"] < float(cr["overnight"]["survival_rate_ge"]):
                            why.append(f"survival {s['survived']}/{s['n_ok']}")
                    else:
                        why.append("too little data")
                    taxes = {}
                    for cap in CAPITALS:
                        r = day_returns(df, "overnight", india_round_trip(ci, cap))
                        r = r[(r.index >= st_c) & (r.index <= day_end)]
                        taxes[int(cap)] = {k: after_tax_cagr(r, cap, float(ci[k]["rate"]), float(ci[k]["cess"]))["cagr"]
                                           for k in ("tax_main", "tax_alt")}
                    bhr = day_returns(df, "buy_and_hold")
                    bhr = bhr[(bhr.index >= st_c) & (bhr.index <= day_end)]
                    row["after_tax"] = taxes
                    row["bh_after_tax"] = {int(cap): buy_hold_after_tax(bhr, cap, 0.125, 0.04, 125_000.0)["cagr"] for cap in CAPITALS}
                    rf_ann = float(plan["data"]["india_rf_annual"])
                    if taxes[int(ci["capital_rs"])]["tax_main"] <= rf_ann:
                        why.append(f"after costs and STCG {taxes[int(ci['capital_rs'])]['tax_main']:+.1%}/yr <= liquid fund {rf_ann:.1%}")
                    row["why"] = why
                    conf["overnight"][t] = row
                    if req and why:
                        why_all.append(f"{t}: " + "; ".join(why))
            else:
                for reg, rspec in plan["data"]["momentum_confirmation"].items():
                    req = bool(rspec.get("required"))
                    try:
                        d = load_momentum(rspec, fb, month_cut, synthetic, seed=66)
                    except Exception as e:
                        conf["momentum"].setdefault(reg, {"required": req, "loaded": False, "error": str(e), "summ": {}})
                        if req:
                            why_all.append(f"{reg}: NOT LOADED ({e})")
                        continue
                    cst = plan["periods"]["momentum_confirmation"]["start"]
                    box = conf["momentum"].setdefault(reg, {"required": req, "loaded": True, "summ": {}, "why": {}})
                    if "bench:market_10vol" not in box["summ"]:
                        box["summ"]["bench:market_10vol"] = eval_mom(bench_m, common, d, cst, month_end)
                    bm = box["summ"]["bench:market_10vol"]
                    s = eval_mom(spec, common, d, cst, month_end)
                    box["summ"][n] = s
                    why = []
                    if "mean_sharpe" in s and "mean_sharpe" in bm:
                        s["sharpe_minus_bench"], s["p_vs_bench"] = compare(s, bm)
                        if s["mean_sharpe"] <= bm["mean_sharpe"]:
                            why.append(f"excess Sharpe {s['mean_sharpe']:.2f} <= market {bm['mean_sharpe']:.2f}")
                        if s["excess_cagr"] <= float(cr["momentum"]["excess_return_gt"]):
                            why.append(f"excess return {s['excess_cagr']:+.1%}/yr")
                        if not s["p_vs_bench"] < float(cr["momentum"]["p_value_lt"]):
                            why.append(f"advantage not significant (p {s['p_vs_bench']:.2f})")
                        if s["survival_rate"] < float(cr["momentum"]["survival_rate_ge"]):
                            why.append(f"survival {s['survived']}/{s['n_ok']}")
                    else:
                        why.append("too little data")
                    box["why"][n] = why
                    if req and why:
                        why_all.append(f"{reg}: " + "; ".join(why))
            conf["verdicts"][n] = {"pass": not why_all, "reasons": why_all}
    elif run_confirmation:
        conf = {"status": "no finalists"}

    # ---- verdicts under the ACTIVE bar, plus P9 (NOT active) ----
    for n in cands:
        s = summ[n]
        ok, why = base_checks(s, pc)
        if (s.get("sharpe_minus_bench") or -1) <= 0:
            ok, why = False, why + ["excess Sharpe not above its benchmark's"]
        if (s.get("costs_2x_excess_cagr") or -1) <= 0:
            ok, why = False, why + ["not profitable over T-bills at 2x costs"]
        if n not in finalists:
            ok, why = False, why + ["not a finalist"]
        elif conf.get("status") != "ok":
            ok, why = False, why + ["confirmation not run"]
        elif not conf["verdicts"][n]["pass"]:
            ok, why = False, why + ["confirmation failed: " + " | ".join(conf["verdicts"][n]["reasons"])]
        s["verdict"], s["reasons"] = ("PASS" if ok else "FAIL"), why
        dsr = deflated_sharpe_prob(s["mean_sharpe"], s["sharpe_haircut"], s["n_days"], s["years"], s["skew"], s["kurt"]) \
            if "mean_sharpe" in s else 0.0
        s["p9"] = bool(s.get("p_vs_bench", 1) < 0.05 / len(cands) and dsr > 0.95 and s.get("survival_rate", 0) >= 0.8
                       and (s.get("costs_2x_excess_cagr") or -1) > 0 and n in finalists
                       and conf.get("status") == "ok" and conf["verdicts"][n]["pass"])

    # ---- India tax harvesting (accounting rule, NIFTYBEES closes) ----
    harvest = {}
    tick = next(iter(plan["data"]["overnight_confirmation"]))
    if tick in india:
        close = india[tick]["close"]
        close = close[(close.index >= "2010-01-01") & (close.index <= day_end)]
        for cap in CAPITALS:
            harvest[int(cap)] = {"hold": tax_harvest_sim(close, cap, False, ov["costs"]["india"]),
                                 "harvest": tax_harvest_sim(close, cap, True, ov["costs"]["india"])}
        harvest_period = f"{close.index[0].date()} -> {close.index[-1].date()}"
    else:
        harvest_period = f"not run: {tick} not loaded ({errors.get(tick)})"

    res = {"plan": plan, "status": status, "n_total": n_total, "n_r6": n_r6, "summ": summ, "finalists": finalists,
           "confirmation": conf, "quality": quality, "errors": errors, "decomposition": decomp, "crowding": crowd,
           "harvest": harvest, "harvest_period": harvest_period, "changed": changed, "synthetic": synthetic,
           "explore_end": str(day_end.date()), "plan_sha": hashlib.sha1(PLAN6.read_bytes()).hexdigest()[:12]}
    save_json(rdir / "summary.json", _clean(res))
    return res


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(res: dict, path: Path) -> str:
    from reporting import _num, _pct, _plain, _signed
    summ, plan, st = res["summ"], res["plan"], res["status"]
    L = ["## Round 6 research: overnight effect, momentum crash guard, Emerging Markets, India tax harvesting\n"]
    if res["synthetic"]:
        L.append("> **SYNTHETIC DATA - NOT REAL RESULTS.**\n")
    if res.get("changed"):
        L.append(f"**{res['changed']}**\n")
    L.append(f"- Plan `research/plan_round6.yaml` sha `{res['plan_sha']}`; trial status: "
             + ", ".join(f"{n}={v}" for n, v in st.items()) + "  \n"
             f"- **Trials counted for the Sharpe penalty: {res['n_total']}** (all rounds, incl. {res['n_r6']} round-6)  \n"
             f"- Exploration ends {res['explore_end']} (the day before the locked test set). Literature, sources and "
             "the ideas NOT tested: `research/ROUND6_NOTES.md`.\n")

    L.append("### Are the free open prices usable?\n")
    L.append("| data | days | first | last | open = previous close | open outside day's range | missing | largest gaps | "
             "close-to-close jumps > 25% (bad split?) | verdict |\n|---|---|---|---|---|---|---|---|---|---|")
    for t, q in res["quality"].items():
        L.append(f"| {t} | {q['days']} | {q['first']} | {q['last']} | {q['share_open_eq_prev_close']:.1%} | "
                 f"{q['share_open_outside_range']:.1%} | {q['share_missing']:.1%} | {', '.join(q['largest_gaps'])} | "
                 f"{', '.join(q['jumps']) or 'none'} | {'reliable' if q['reliable'] else '**UNRELIABLE**'} |")
    for t, e in res["errors"].items():
        L.append(f"| {t} | NOT LOADED: {e} | | | | | | | | |")

    L.append("\n### Where does the return come from? (gross, average per year, no costs)\n")
    L.append("| data | period | close -> open (overnight) | open -> close (intraday) | close -> close | overnight bps/day |\n"
             "|---|---|---|---|---|---|")
    for t, rows in res["decomposition"].items():
        for r in rows:
            L.append(f"| {t} | {r['period']} | {_pct(r['overnight'])} | {_pct(r['intraday'])} | {_pct(r['close_to_close'])} | "
                     f"{r['overnight_bps_per_day']:.1f} |")

    L.append(f"\n### Overnight exploration (US, {plan['overnight']['costs']['us_lrs']['round_trip_bps']} bps per round trip via LRS)\n")
    L.append("| strategy | trial | period | excess Sharpe | vs buy & hold | p (not better) | CAGR | max DD | survived | "
             "break-even cost bps/round trip (vs T-bills / vs B&H) | excess Sharpe 2x costs | stress fills: Sharpe / CAGR | "
             "w/o kill switch: Sharpe / CAGR | after-tax CAGR (slab 30%) | penalized Sharpe | DSR prob | finalist | verdict | P9 (NOT active) |")
    L.append("|" + "---|" * 19)
    names_on = [n for n in summ if summ[n].get("family") in ("overnight", "intraday")]
    for n in names_on + ["bench:buy_and_hold_spy"]:
        s = summ[n]
        cand = n in plan["hypotheses"]
        L.append(f"| {n} | {st.get(n, 'benchmark')} | {s.get('first', '?')} -> {s.get('last', '?')} | {_num(s.get('mean_sharpe'))} | "
                 f"{_signed(s.get('sharpe_minus_bench'))} | {_num(s.get('p_vs_bench'), 3)} | {_pct(s.get('cagr'))} | "
                 f"{_plain(s.get('mean_stitched_mdd'), 1)} | {s.get('survived', 0)}/{s.get('n_ok', 0)} | "
                 f"{_num(s.get('breakeven_bps_vs_tbills'), 1)} / {_num(s.get('breakeven_bps_vs_bh'), 1)} | "
                 f"{_num(s.get('costs_2x_sharpe'))} | {_num(s.get('stress_sharpe'))} / {_pct(s.get('stress_cagr'))} | "
                 f"{_num(s.get('nokill_sharpe'))} / {_pct(s.get('nokill_cagr'))} | {_pct(s.get('after_tax_cagr'))} | "
                 f"{_num(s.get('penalized_sharpe'))} | {_num(s.get('dsr_prob'))} | "
                 f"{('yes' if n in res['finalists'] else 'no') if cand else '-'} | {s.get('verdict', 'diagnostic' if n in st else 'benchmark')} | "
                 f"{('yes' if s.get('p9') else 'no') if cand else '-'} |")
    L.append("\n*Break-even cost*: average overnight return per day minus T-bills (or minus buy & hold's daily return), in bps. "
             "A round trip that costs more than this loses money (or loses to buy & hold). *Stress fills*: on days the "
             "gap is <= -2% you sell at the day's low. Buy & hold's after-tax CAGR: one sale at the end, LTCG 12.5%. "
             "*Excess Sharpe* uses T-bills; overnight money earns nothing during the day.\n")

    L.append("### Momentum exploration (US monthly, Ken French; everything at 10% volatility)\n")
    L.append("| strategy | trial | period | excess Sharpe | vs market | p (not better) | CAGR | max DD | survived | "
             "months guarded | excess Sharpe 2x costs | w/o kill switch | " + " | ".join(f"vs market {p[0][:4]}-{p[1][:4] if p[1] < '2100' else 'now'}" for p in MOM_SUBPERIODS)
             + " | penalized Sharpe | DSR prob | finalist | verdict | P9 (NOT active) |")
    L.append("|" + "---|" * (17 + len(MOM_SUBPERIODS)))
    names_m = [n for n in summ if summ[n].get("family") == "momentum"]
    for n in names_m + ["bench:market_10vol"]:
        s = summ[n]
        cand = n in plan["hypotheses"]
        subs = " | ".join(_signed((s.get("subperiods") or {}).get(f"{p[0][:4]}-{p[1][:4] if p[1] < '2100' else 'now'}"))
                          for p in MOM_SUBPERIODS)
        L.append(f"| {n} | {st.get(n, 'benchmark')} | {s.get('first', '?')} -> {s.get('last', '?')} | {_num(s.get('mean_sharpe'))} | "
                 f"{_signed(s.get('sharpe_minus_bench'))} | {_num(s.get('p_vs_bench'), 3)} | {_pct(s.get('cagr'))} | "
                 f"{_plain(s.get('mean_stitched_mdd'), 1)} | {s.get('survived', 0)}/{s.get('n_ok', 0)} | "
                 f"{_pct(s.get('guard_share'), 0) if cand else '-'} | {_num(s.get('costs_2x_sharpe'))} | {_num(s.get('nokill_sharpe'))} | "
                 f"{subs} | {_num(s.get('penalized_sharpe'))} | {_num(s.get('dsr_prob'))} | "
                 f"{('yes' if n in res['finalists'] else 'no') if cand else '-'} | {s.get('verdict', 'benchmark')} | "
                 f"{('yes' if s.get('p9') else 'no') if cand else '-'} |")
    cw = res["crowding"]
    L.append("\n**Crowding / crash risk check** (large-cap winners minus the market, gross, full exposure): "
             + "; ".join(f"{k}: {_pct(v)}" for k, v in cw["episodes"].items())
             + ". Worst months since 1963: " + ", ".join(cw["worst_months"]) + ".\n")

    L.append("### Why each hypothesis passed or failed\n")
    for n in [n for n in plan["hypotheses"] if n in summ]:
        L.append(f"- **{n}** ({st[n]}): {summ[n]['verdict']} - " + ("; ".join(summ[n]["reasons"]) or "all checks met"))
    L.append("- **R6_H4_india_tax_harvest**: accounting rule, no pass/fail - see the table below.")
    L.append(f"\n### Finalists (pre-registered: survival >= 80%, beats T-bills at 2x costs, beats its benchmark; "
             f"top 3 by excess Sharpe minus benchmark's)\n\n{', '.join(res['finalists']) or 'none'}\n")

    conf = res["confirmation"]
    L.append("### Confirmation (data never used before: NIFTYBEES / Nifty opens; Emerging Markets 1991-2024)\n")
    if conf.get("status") != "ok":
        L.append(f"Not run: {conf.get('status')}.\n")
    else:
        for t, v in conf["overnight"].items():
            lab = "required" if v["required"] else "informational"
            if not v["loaded"]:
                L.append(f"- **{t}** ({lab}): NOT LOADED - {v.get('error')}")
                continue
            s, b = v["summ"], v["bench"]
            L.append(f"\n**{t} overnight** ({lab}; India costs per round trip: "
                     + ", ".join(f"Rs {k:,}: {x:.1f} bps" for k, x in v["cost_bps_round_trip"].items()) + ")\n")
            L.append("| strategy | period | excess Sharpe | p (not better) | CAGR | max DD | survived |\n|---|---|---|---|---|---|---|")
            for lab2, x in (("overnight (Rs 10 lakh costs)", s), ("buy & hold", b)):
                L.append(f"| {lab2} | {x.get('first', '?')} -> {x.get('last', '?')} | {_num(x.get('mean_sharpe'))} | "
                         f"{_num(x.get('p_vs_bench'), 3)} | {_pct(x.get('cagr'))} | {_plain(x.get('mean_stitched_mdd'), 1)} | "
                         f"{x.get('survived', 0)}/{x.get('n_ok', 0)} |")
            L.append("\n| capital | overnight after costs + STCG 20% | overnight if taxed at slab 30% | buy & hold after LTCG |\n|---|---|---|---|")
            for cap, x in v["after_tax"].items():
                L.append(f"| Rs {cap:,} | {_pct(x['tax_main'])} | {_pct(x['tax_alt'])} | {_pct(v['bh_after_tax'][cap])} |")
            L.append("\n| period | overnight | intraday | close -> close |\n|---|---|---|---|")
            for r in v["decomposition"]:
                L.append(f"| {r['period']} | {_pct(r['overnight'])} | {_pct(r['intraday'])} | {_pct(r['close_to_close'])} |")
            L.append(f"\nVerdict on {t}: " + ("PASS" if not v["why"] else "FAIL - " + "; ".join(v["why"])) + "\n")
        for reg, v in conf["momentum"].items():
            lab = "required" if v["required"] else "informational"
            if not v["loaded"]:
                L.append(f"- **{reg}** ({lab}): NOT LOADED - {v.get('error')}")
                continue
            L.append(f"\n**{reg} momentum** ({lab})\n")
            L.append("| strategy | period | excess Sharpe | p (not better) | CAGR | excess CAGR | max DD | survived | verdict |\n"
                     "|---|---|---|---|---|---|---|---|---|")
            for n, s in v["summ"].items():
                w = v["why"].get(n)
                L.append(f"| {n} | {s.get('first', '?')} -> {s.get('last', '?')} | {_num(s.get('mean_sharpe'))} | "
                         f"{_num(s.get('p_vs_bench'), 3)} | {_pct(s.get('cagr'))} | {_pct(s.get('excess_cagr'))} | "
                         f"{_plain(s.get('mean_stitched_mdd'), 1)} | {s.get('survived', 0)}/{s.get('n_ok', 0)} | "
                         f"{('PASS' if not w else 'FAIL: ' + '; '.join(w)) if w is not None else 'benchmark'} |")
        L.append("\nOverall: " + "; ".join(f"{n}: {'CONFIRMED' if v['pass'] else 'NOT confirmed'}"
                                           for n, v in conf["verdicts"].items()) + "\n")

    L.append(f"### India tax harvesting (NIFTYBEES {res['harvest_period']})\n")
    if res["harvest"]:
        L.append("Buy and hold vs buy and hold + selling up to Rs 1.25 lakh of long-term gain each Jan-Mar and re-buying "
                 "the next day. Both sold and fully taxed on the last day. Current tax rules applied to the whole period.\n")
        L.append("| capital | buy & hold: after-tax CAGR (tax, charges) | with harvesting: after-tax CAGR (tax, charges) | "
                 "harvest trades | gain per year | extra rupees at the end |\n|---|---|---|---|---|---|")
        for cap, v in res["harvest"].items():
            h, w = v["hold"], v["harvest"]
            L.append(f"| Rs {cap:,} | {_pct(h['cagr'], 2)} ({h['tax_share']:.1%}, {h['charges_share']:.2%}) | "
                     f"{_pct(w['cagr'], 2)} ({w['tax_share']:.1%}, {w['charges_share']:.2%}) | {w['harvests']} | "
                     f"{_signed(100 * (w['cagr'] - h['cagr']))} pp | Rs {w['final'] - h['final']:,.0f} |")
        L.append("\nTax and charges are shown as a share of the starting capital.\n")
    else:
        L.append("Not run.\n")
    md = "\n".join(L) + "\n"
    path.write_text("# RESEARCH - round 6\n\n" + md, encoding="utf-8")
    return md


def final_test_round6(name: str, base: dict, synthetic: bool) -> dict:
    """ONE-TIME evaluation of a round-6 finalist on the locked test period (called only by final_test.py)."""
    plan = load_plan()
    spec = plan["hypotheses"][name]
    split = get_split(base, load_prices(base))
    end = split.data_end + pd.Timedelta(days=1)
    out = {}
    if spec["family"] == "overnight":
        ci = plan["overnight"]["costs"]["india"]
        for t, cost in ((spec["ticker"], float(plan["overnight"]["costs"]["us_lrs"]["round_trip_bps"]) / 1e4),
                        (next(iter(plan["data"]["overnight_confirmation"])), india_round_trip(ci, float(ci["capital_rs"])))):
            df = load_ohlc(plan, [t], end, synthetic)[t]
            rf = india_daily_rf(plan, df.index) if t.endswith(".NS") else us_daily_rf(plan, end, synthetic, df.index)
            for kind, c in (("overnight", cost), ("buy_and_hold", 0.0)):
                r = day_returns(df, kind, c)
                r = r[(r.index >= split.test_start) & (r.index <= split.data_end)]
                rets, wins = run_daily(r, float(plan["overnight"]["kill_switch"]), split.test_start, split.data_end,
                                       single_window=True)
                out[f"{t} {kind}"] = stats_daily(rets, rf, wins)
    else:
        d = load_momentum(plan["data"]["momentum_exploration"], plan["data"]["french_base_url"], end, synthetic)
        start = pd.Timestamp(split.test_start.year, split.test_start.month, 1) + pd.offsets.MonthEnd(0)
        common = plan["momentum_common"]
        for n, sp in ((name, spec), ("bench:market_10vol", {"sleeve": "market"})):
            sl = mom_sleeve(sp, common, d["monthly"])
            pos = positions({}, common, d, sl)
            rets, wins = run_monthly(sl, d["monthly"]["rf"], pos, float(common["switch_cost_bps"]) / 1e4,
                                     float(common["kill_switch"]), str(start.date()), split.data_end, single_window=True)
            out[n] = stats(rets, d["monthly"]["rf"], wins) if len(rets) >= 24 else {
                "months": len(rets), "total_return": float((1 + rets).prod() - 1) if len(rets) else None,
                "dead": bool(wins and wins[0]["dead"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    from run_all import legacy_trial_count, mode_dirs, prepare_cfg, setup_logging
    from factor_research import count_prefix
    base_raw = load_base()
    base = prepare_cfg(base_raw, args.synthetic, smoke=False)
    setup_logging(base, "research6_" + ("synthetic" if args.synthetic else "real"))
    res_root, _ = mode_dirs(base, args.synthetic, False)
    run_conf = bool((base_raw.get("research6") or {}).get("run_confirmation", False))
    other = legacy_trial_count(res_root, base_raw) + count_prefix(res_root, "r4:") + count_prefix(res_root, "r5:")
    res = run_round6(base, res_root, args.synthetic, other, run_conf)
    md = ROOT / ("RESEARCH6.md" if not args.synthetic else "RESEARCH6_SYNTHETIC.md")
    write_report(res, md)
    log.info("wrote %s", md.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
