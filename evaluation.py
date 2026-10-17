"""Pass/fail logic and multiple-testing penalty.

Why penalise? If you try N strategy variants and keep the best, its Sharpe is
biased upward even when none has any real edge. Under the null (all true
Sharpes = 0) the expected best of N noisy Sharpes is roughly

    E[max SR] ~= sd(SR) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))   (g = 0.5772)

(Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", 2014). We subtract
that from the claimed validation Sharpe ("penalized Sharpe") and also report
the Deflated Sharpe probability: P(true Sharpe > that bar).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    if n_trials <= 1:
        return 0.0
    return float(sr_std * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def null_sharpe_std(n_days: int, years: float) -> float:
    """Std of an annualised Sharpe estimate when the true Sharpe is 0."""
    if n_days < 2 or years <= 0:
        return 1.0
    ppy = n_days / years
    return float(np.sqrt(ppy / n_days))


def deflated_sharpe_prob(sr_ann: float, sr0_ann: float, n_days: int, years: float, skew: float, kurt: float) -> float:
    if n_days < 3 or years <= 0:
        return 0.0
    ppy = n_days / years
    sr, sr0 = sr_ann / np.sqrt(ppy), sr0_ann / np.sqrt(ppy)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    denom = max(denom, 1e-6)
    return float(norm.cdf((sr - sr0) * np.sqrt(n_days - 1) / np.sqrt(denom)))


def penalize(summary: dict, n_trials: int, cross_sharpes: list[float]) -> dict:
    """Adds penalized Sharpe + DSR to a summary (in place) and returns it."""
    if "mean_sharpe" not in summary:
        return summary
    sd_null = null_sharpe_std(summary["n_days"], summary["years"])
    sd_cross = float(np.std(cross_sharpes, ddof=1)) if len(cross_sharpes) >= 3 else 0.0
    sr_std = max(sd_null, sd_cross)   # conservative
    haircut = expected_max_sharpe(n_trials, sr_std)
    summary["n_trials"] = n_trials
    summary["sharpe_haircut"] = haircut
    summary["penalized_sharpe"] = summary["mean_sharpe"] - haircut
    summary["dsr_prob"] = deflated_sharpe_prob(summary["mean_sharpe"], haircut, summary["n_days"],
                                               summary["years"], summary.get("skew", 0.0), summary.get("kurt", 3.0))
    return summary


def base_checks(summary: dict, pc: dict, sharpe_test: str | None = None) -> tuple[bool, list[str]]:
    """Validation-stage criteria. Returns (passed, list of failure reasons).

    sharpe_test (pass_criteria.sharpe_test):
      "penalized" (ACTIVE default): Sharpe minus the best-of-N luck haircut must exceed min_penalized_sharpe.
      "dsr" (round-3 PROPOSAL, needs your approval): raw Sharpe > min_raw_sharpe AND deflated-Sharpe
            probability > min_dsr_prob, i.e. "good enough, and very unlikely to be best-of-N luck"."""
    sharpe_test = sharpe_test or pc.get("sharpe_test", "penalized")
    reasons = []
    if not summary.get("complete"):
        reasons.append(f"incomplete ({summary.get('n_ok', 0)}/{summary.get('n_runs', 0)} runs ok)")
    if "mean_sharpe" not in summary:
        return False, reasons + ["no results"]
    if summary["mean_sharpe"] > pc["suspicious_sharpe"]:
        reasons.append(f"SUSPICIOUS Sharpe {summary['mean_sharpe']:.2f} > {pc['suspicious_sharpe']} - assume bug/leakage")
    if summary["mean_return"] <= pc["min_oos_return"]:
        reasons.append(f"OOS return {summary['mean_return']:+.1%} not > {pc['min_oos_return']:+.1%}")
    if sharpe_test == "penalized":
        if summary.get("penalized_sharpe", -9) <= pc["min_penalized_sharpe"]:
            reasons.append(f"penalized Sharpe {summary.get('penalized_sharpe', float('nan')):.2f} "
                           f"(raw {summary['mean_sharpe']:.2f} - haircut {summary.get('sharpe_haircut', 0):.2f}) "
                           f"not > {pc['min_penalized_sharpe']}")
    elif sharpe_test == "dsr":
        if summary["mean_sharpe"] <= pc.get("min_raw_sharpe", 1.0):
            reasons.append(f"Sharpe {summary['mean_sharpe']:.2f} not > {pc.get('min_raw_sharpe', 1.0)}")
        if summary.get("dsr_prob", 0.0) <= pc.get("min_dsr_prob", 0.95):
            reasons.append(f"deflated-Sharpe probability {summary.get('dsr_prob', 0):.2f} "
                           f"not > {pc.get('min_dsr_prob', 0.95)}")
    else:
        raise ValueError(f"pass_criteria.sharpe_test must be penalized|dsr, got {sharpe_test}")
    if summary["survival_rate"] < pc["min_survival_rate"]:
        reasons.append(f"survival {summary['survived']}/{summary['n_ok']} = {summary['survival_rate']:.0%} "
                       f"< {pc['min_survival_rate']:.0%}")
    return not reasons, reasons


def robust_check(summary: dict, pc: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not summary or "mean_sharpe" not in summary:
        return False, ["no results"]
    if not summary.get("complete"):
        reasons.append("incomplete")
    if summary["mean_return"] <= pc["robust_min_return"]:
        reasons.append(f"return {summary['mean_return']:+.1%}")
    if summary["mean_sharpe"] <= pc["robust_min_sharpe"]:
        reasons.append(f"Sharpe {summary['mean_sharpe']:.2f} <= {pc['robust_min_sharpe']}")
    if summary["survival_rate"] < pc["robust_min_survival_rate"]:
        reasons.append(f"survival {summary['survival_rate']:.0%}")
    return not reasons, reasons


def relative_checks(summary: dict, bench: dict | None, own_base: dict | None, pc: dict) -> tuple[bool, list[str]]:
    """Round 2: compare with what you could get WITHOUT RL on the same windows.
    These are added ON TOP of the absolute checks (never instead of them):
    loosening the bar after seeing round-1 results would be moving the goalposts."""
    reasons = []
    if "mean_sharpe" not in summary:
        return False, reasons
    if pc.get("must_beat_benchmark_sharpe", True):
        if not bench or "mean_sharpe" not in bench:
            reasons.append("benchmark missing")
        elif summary["mean_sharpe"] <= bench["mean_sharpe"]:
            reasons.append(f"Sharpe {summary['mean_sharpe']:.2f} not above buy-and-hold {bench['mean_sharpe']:.2f} "
                           "(same windows)")
    if pc.get("must_beat_own_base_rule", True) and own_base and "mean_sharpe" in own_base:
        if summary["mean_sharpe"] <= own_base["mean_sharpe"]:
            reasons.append(f"Sharpe {summary['mean_sharpe']:.2f} not above its own rule without RL "
                           f"{own_base['mean_sharpe']:.2f} - the RL part added nothing")
    return not reasons, reasons


def low_risk_match(summary: dict, bench: dict | None, pc: dict) -> bool | None:
    """Informational: 'about the same Sharpe as buy-and-hold with much smaller drawdown'.
    Does NOT make a strategy pass."""
    if not bench or "mean_sharpe" not in summary or "mean_sharpe" not in bench:
        return None
    tol = float(pc.get("low_risk_sharpe_tolerance", 0.1))
    ratio = float(pc.get("low_risk_dd_ratio", 0.6))
    return bool(summary["mean_sharpe"] >= bench["mean_sharpe"] - tol
                and summary["mean_stitched_mdd"] <= ratio * bench["mean_stitched_mdd"]
                and summary["survival_rate"] >= bench["survival_rate"])


def final_verdict(summary: dict, robustness: dict, required: list[str], pc: dict,
                  bench: dict | None = None, own_base: dict | None = None, sharpe_test: str | None = None) -> dict:
    ok, reasons = base_checks(summary, pc, sharpe_test)
    rel_ok, rel_reasons = relative_checks(summary, bench, own_base, pc)
    ok = ok and rel_ok
    reasons += rel_reasons
    rob = {}
    for test in required:
        r = robustness.get(test)
        if r is None or r.get("status") != "ok":
            ok = False
            reasons.append(f"robustness '{test}' not run" + (f" ({r.get('status')})" if r else ""))
            continue
        if "pass" in r:   # the test's own verdict (param_neighborhood: ALL neighbours must pass)
            p, why = bool(r["pass"]), list(r.get("reasons") or [])
        else:
            p, why = robust_check(r["summary"], pc)
        rob[test] = p
        if not p:
            ok = False
            reasons.append(f"robustness '{test}' failed: {', '.join(why)}")
    return {"pass": ok, "reasons": reasons, "robust": rob}


# --------------------------------------------------------------------------
# Round 4: measuring PROFIT at equal risk, and whether an edge is real
# --------------------------------------------------------------------------
def bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, block: int = 21, n_boot: int = 2000, seed: int = 0) -> float:
    """p-value that strategy a's Sharpe is NOT above b's (paired circular block bootstrap).
    Both series are resampled with the SAME blocks of days, so their correlation is kept:
    two strategies that move together need a smaller gap to be told apart.
    Small p (e.g. < 0.05) = the Sharpe advantage is unlikely to be luck of this sample."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2 * block or len(b) != n:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))

    def sharpe(x):
        sd = x.std(ddof=1)
        return x.mean() / sd if sd > 0 else 0.0
    worse = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n] % n
        if sharpe(a[idx]) - sharpe(b[idx]) <= 0:
            worse += 1
    return worse / n_boot


def cagr_at_vol(rets: np.ndarray, rf_daily: np.ndarray, target_vol: float = 0.10, borrow_spread: float = 0.01,
                periods_per_year: float = 252.0) -> tuple[float, float]:
    """CAGR the strategy would have earned if run at `target_vol` annual volatility.
    Scale k = target / realised vol. k > 1 means borrowing: the extra (k - 1) pays T-bill +
    spread every day. k < 1 leaves cash earning nothing (same as the active design).
    Returns (cagr, k). This compares strategies at EQUAL RISK - the honest way to ask
    'which makes more money?' (a 2x riskier strategy should make more, that is not skill)."""
    r = np.asarray(rets, float)
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    if vol <= 0:
        return float("nan"), float("nan")
    k = target_vol / vol
    fin = (k - 1.0) * (np.asarray(rf_daily, float) + borrow_spread / periods_per_year) if k > 1 else 0.0
    scaled = k * r - fin
    growth = np.prod(1.0 + scaled)
    years = len(r) / periods_per_year
    return float(growth ** (1 / years) - 1) if growth > 0 else -1.0, float(k)
