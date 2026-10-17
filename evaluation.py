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


def base_checks(summary: dict, pc: dict) -> tuple[bool, list[str]]:
    """Validation-stage criteria. Returns (passed, list of failure reasons)."""
    reasons = []
    if not summary.get("complete"):
        reasons.append(f"incomplete ({summary.get('n_ok', 0)}/{summary.get('n_runs', 0)} runs ok)")
    if "mean_sharpe" not in summary:
        return False, reasons + ["no results"]
    if summary["mean_sharpe"] > pc["suspicious_sharpe"]:
        reasons.append(f"SUSPICIOUS Sharpe {summary['mean_sharpe']:.2f} > {pc['suspicious_sharpe']} - assume bug/leakage")
    if summary["mean_return"] <= pc["min_oos_return"]:
        reasons.append(f"OOS return {summary['mean_return']:+.1%} not > {pc['min_oos_return']:+.1%}")
    if summary.get("penalized_sharpe", -9) <= pc["min_penalized_sharpe"]:
        reasons.append(f"penalized Sharpe {summary.get('penalized_sharpe', float('nan')):.2f} "
                       f"(raw {summary['mean_sharpe']:.2f} - haircut {summary.get('sharpe_haircut', 0):.2f}) "
                       f"not > {pc['min_penalized_sharpe']}")
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


def final_verdict(summary: dict, robustness: dict, required: list[str], pc: dict) -> dict:
    ok, reasons = base_checks(summary, pc)
    rob = {}
    for test in required:
        r = robustness.get(test)
        if r is None or r.get("status") != "ok":
            ok = False
            reasons.append(f"robustness '{test}' not run" + (f" ({r.get('status')})" if r else ""))
            continue
        p, why = robust_check(r["summary"], pc)
        rob[test] = p
        if not p:
            ok = False
            reasons.append(f"robustness '{test}' failed: {', '.join(why)}")
    return {"pass": ok, "reasons": reasons, "robust": rob}
