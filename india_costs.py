"""What an Indian retail investor would actually keep (round 5 reality check).

Simulates a strategy's monthly trades with FIFO tax lots and real-world frictions, for
three capital sizes, in two regimes:
  india_listed : Indian ETFs / index funds on NSE (e.g. NIFTYBEES, a Nifty200 Momentum 30 fund),
                 idle money in a liquid fund
  us_lrs       : US ETFs bought through the Liberalised Remittance Scheme (Vested/INDmoney-type
                 platforms), idle money in a US T-bill ETF

Rules as of FY2025-26 (they change - verify before using real money):
  Equity ETFs/funds held < 12 months: STCG 20%; >= 12 months: LTCG 12.5% above Rs 1.25 lakh
  per financial year. Liquid/debt funds bought after Apr 2023: gains taxed at your slab rate.
  Foreign ETFs: LTCG 12.5% if held > 24 months, otherwise slab rate; no Rs 1.25 lakh exemption.
  4% cess on all tax. Losses offset gains (short-term against anything, long-term against
  long-term) and carry forward. Surcharge, TCS (refundable), dividend withholding (creditable)
  and the Rs 87A rebate are ignored.
NSE charges per trade (discount broker, delivery): brokerage 0, STT 0.001% on ETF sells,
exchange 0.00297%, SEBI 0.0001%, stamp duty 0.015% on buys, 18% GST on exchange+SEBI+brokerage,
DP charge ~Rs 18 per scheme per day you sell, plus half the bid-ask spread.

IMPORTANT: the input returns are US data (Ken French) used as a stand-in for the PATTERN of
trading (how often you buy and sell, how long you hold). Indian factor funds have different
returns. What this measures is the DRAG that taxes and costs put on that pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIMES = {
    "india_listed": {"st_days": 365, "st_rate": 0.20, "lt_rate": 0.125, "lt_exempt": 125_000.0, "cash_rate": "slab",
                     "buy_pct": 0.00015 + 0.0000297 * 1.18 + 0.000001 * 1.18, "sell_pct": 0.00001 + 0.0000297 * 1.18 + 0.000001 * 1.18,
                     "dp_per_sell": 18.0, "fx_pct": 0.0},
    "us_lrs": {"st_days": 730, "st_rate": "slab", "lt_rate": 0.125, "lt_exempt": 0.0, "cash_rate": "slab",
               "buy_pct": 0.0, "sell_pct": 0.0, "dp_per_sell": 0.0, "fx_pct": 0.005},
}
SPREAD_HALF = {"market": 0.0002, "big_value": 0.001, "big_momentum": 0.0015}   # factor ETFs trade wider


@dataclass
class Lot:
    units: float
    cost: float          # per unit
    day: pd.Timestamp


@dataclass
class Book:
    lots: dict = field(default_factory=dict)       # fund -> [Lot]
    st_gain: float = 0.0
    lt_gain: float = 0.0
    cash_gain: float = 0.0
    carry_st: float = 0.0                           # carried-forward losses (positive numbers)
    carry_lt: float = 0.0
    tax_paid: float = 0.0
    charges: float = 0.0


def _sell(book: Book, fund: str, value: float, nav: float, day: pd.Timestamp, st_days: int, is_cash: bool):
    """Sell `value` of a fund FIFO; record realised gains by type."""
    units = value / nav
    lots = book.lots.get(fund, [])
    while units > 1e-12 and lots:
        lot = lots[0]
        take = min(units, lot.units)
        gain = take * (nav - lot.cost)
        if is_cash:
            book.cash_gain += gain
        elif (day - lot.day).days < st_days:
            book.st_gain += gain
        else:
            book.lt_gain += gain
        lot.units -= take
        units -= take
        if lot.units <= 1e-12:
            lots.pop(0)


def _settle_year(book: Book, reg: dict, slab: float) -> float:
    """Tax for one financial year (with set-off and carry-forward). Returns tax incl. 4% cess."""
    st_rate = slab if reg["st_rate"] == "slab" else reg["st_rate"]
    st = book.st_gain - book.carry_st
    lt = book.lt_gain - book.carry_lt
    book.carry_st = book.carry_lt = 0.0
    if st < 0:                                   # short-term losses can offset long-term gains
        lt += st
        st = 0.0
        if lt < 0:
            book.carry_st, lt = -lt, 0.0
    elif lt < 0:
        book.carry_lt, lt = -lt, 0.0
    tax = st * st_rate + max(0.0, lt - reg["lt_exempt"]) * reg["lt_rate"] + max(0.0, book.cash_gain) * slab
    book.st_gain = book.lt_gain = book.cash_gain = 0.0
    return tax * 1.04


def simulate(weights: pd.DataFrame, fund_returns: pd.DataFrame, capital: float, regime: str, slab: float = 0.30,
             round_trip_each_month: bool = False) -> dict:
    """weights: target fraction per fund at the START of each month (columns incl. 'cash').
    fund_returns: monthly returns per fund over that month. Returns final value and drag numbers."""
    reg = REGIMES[regime]
    funds = list(weights.columns)
    nav = {f: 100.0 for f in funds}
    book = Book(lots={f: [] for f in funds})
    hold = {f: 0.0 for f in funds}            # units
    value = capital * (1 - reg["fx_pct"])      # FX conversion on the way in (LRS)
    book.charges += capital * reg["fx_pct"]
    hold["cash"] = value / nav["cash"]
    book.lots["cash"].append(Lot(hold["cash"], nav["cash"], weights.index[0]))
    for i, t in enumerate(weights.index):
        day = t
        total = sum(hold[f] * nav[f] for f in funds)
        target = {f: float(weights.at[t, f]) * total for f in funds}
        if round_trip_each_month:                # e.g. turn-of-month: everything risky is sold and re-bought monthly
            sells = {f: hold[f] * nav[f] for f in funds if f != "cash"}
        else:   # cash is not "sold" here: it only funds buys and receives sale proceeds
            sells = {f: max(0.0, hold[f] * nav[f] - target[f]) for f in funds if f != "cash"}
        any_sell = False
        for f, v in sells.items():
            if v <= 1e-6:
                continue
            _sell(book, f, v, nav[f], day, reg["st_days"], f == "cash")
            hold[f] -= v / nav[f]
            if f != "cash":
                c = v * (reg["sell_pct"] + SPREAD_HALF.get(f, 0.0))
                any_sell = True
                book.charges += c
                hold["cash"] += (v - c) / nav["cash"]
                book.lots["cash"].append(Lot((v - c) / nav["cash"], nav["cash"], day))
        if any_sell:
            dp = reg["dp_per_sell"] * sum(1 for f, v in sells.items() if v > 1e-6 and f != "cash")
            book.charges += dp
            _sell(book, "cash", dp, nav["cash"], day, reg["st_days"], True)
            hold["cash"] -= dp / nav["cash"]
        total = sum(hold[f] * nav[f] for f in funds)
        for f in funds:
            if f == "cash":
                continue
            want = float(weights.at[t, f]) * total
            buy = want - hold[f] * nav[f]
            if buy > 1e-6:
                c = buy * (reg["buy_pct"] + SPREAD_HALF.get(f, 0.0))
                book.charges += c
                _sell(book, "cash", buy, nav["cash"], day, reg["st_days"], True)
                hold["cash"] -= buy / nav["cash"]
                units = (buy - c) / nav[f]
                hold[f] += units
                book.lots[f].append(Lot(units, nav[f], day))
        for f in funds:                                   # the month passes
            nav[f] *= 1 + float(fund_returns.at[t, f])
        if t.month == 3 or i == len(weights.index) - 1:    # Indian financial year ends in March
            if i == len(weights.index) - 1:                # liquidate at the end (so buy & hold pays its tax too)
                for f in funds:
                    if hold[f] > 0:
                        _sell(book, f, hold[f] * nav[f], nav[f], day + pd.Timedelta(days=31), reg["st_days"], f == "cash")
            tax = _settle_year(book, reg, slab)
            book.tax_paid += tax
            hold["cash"] -= tax / nav["cash"]
    final = sum(hold[f] * nav[f] for f in funds) * (1 - reg["fx_pct"])
    book.charges += sum(hold[f] * nav[f] for f in funds) * reg["fx_pct"]
    years = len(weights.index) / 12
    return {"final": final, "cagr": (final / capital) ** (1 / years) - 1 if final > 0 else -1.0,
            "tax_paid": book.tax_paid, "charges": book.charges}


def india_reality(spec: dict, common: dict, data: dict, end: pd.Timestamp, years: int = 20,
                  capitals=(1e5, 1e6, 1e7)) -> dict:
    """Pre-tax vs after-cost-and-tax CAGR for one strategy over the last `years` of exploration."""
    from factor_research import positions, sleeve_returns
    m = data["monthly"]
    sl = sleeve_returns(spec, common, data)
    # "passive" = plain buy & hold of the sleeve (no vol targeting, never sells until the end)
    pos = pd.Series(1.0, index=m.index) if spec.get("passive") else positions(spec, common, data, sl)
    idx = m.index[(m.index > end - pd.DateOffset(years=years)) & (m.index <= end)]
    idx = idx[pos.reindex(idx).notna().values & sl.reindex(idx).notna().values]
    tom = bool(spec.get("turn_of_month"))
    comps = list(spec["sleeve"])
    W = pd.DataFrame({k: pos.reindex(idx) * w for k, w in spec["sleeve"].items()}, index=idx)
    W["cash"] = 1.0 - W.sum(axis=1)
    R = pd.DataFrame(index=idx)
    for k in comps:   # the fund's own return (drag = fund fee/turnover); TOM's switching is in its sleeve
        R[k] = (sl.reindex(idx) if tom else m[k].reindex(idx) - common["sleeve_drag_bps_per_year"][k] / 1e4 / 12)
    R["cash"] = m["rf"].reindex(idx)
    gross = float(np.prod(1 + (W[comps].values * R[comps].values).sum(axis=1) + W["cash"].values * R["cash"].values))
    out = {"years": len(idx) / 12, "pre_tax_cagr": gross ** (12 / len(idx)) - 1, "regimes": {}}
    for regime in REGIMES:
        out["regimes"][regime] = {int(c): simulate(W, R, c, regime, round_trip_each_month=tom) for c in capitals}
    return out
