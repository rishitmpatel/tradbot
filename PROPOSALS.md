# Round 3 proposals - for the owner to approve or reject

Nothing in this file is active. Each proposal says what would change, why, the evidence,
and the risk. To approve one, YOU edit the named line in `base_config.yaml` yourself.
The automated agent never changes these settings.

**Decide before you look at round-3 results.** The round-2 numbers below are the only
results you should base the decision on. Approving a bar after seeing which strategy
it lets through is the "moving goalposts" mistake the whole project is built to avoid.

---

## P1 - Replace "penalized Sharpe > 1" with "Sharpe > 1 AND deflated-Sharpe probability > 0.95"

Switch: `pass_criteria.sharpe_test: dsr` (currently `penalized`).

**What changes.**
- *Current test:* subtract the best-of-N luck haircut from the Sharpe, then require what is
  left to exceed 1.0.
- *Proposed test:*
  - (a) the Sharpe itself must exceed 1.0, and
  - (b) the deflated Sharpe ratio test (Bailey & López de Prado) must say the result is
    better than best-of-N luck with probability > 95%. It uses the same N, counted the same way.

All other checks stay as they are: return after costs > 0, survival >= 80%, beat buy-and-hold,
robustness under 2x costs and neighbouring parameters or new seeds, suspicious above 2.5,
and the one-time test set.

**Why the current bar is the wrong yardstick (evidence).**
- *How noisy the number is:* the validation record is 8.8 years long. Over that span a
  measured Sharpe has a standard error of about **0.34** (1 / sqrt(years)).
- *What the current bar demands:* with N = 27 trials the luck haircut is **0.68**. So
  "penalized > 1" means a measured Sharpe above **1.68**, while anything above 2.5 is failed as
  suspicious. The pass window is 1.68 to 2.5.
- *How often an honest strategy could clear it* (computed with this repo's `evaluation.py`,
  normal returns assumed):

| true Sharpe of the strategy | chance it passes the current bar | chance it passes P1 |
|---|---|---|
| 0 (pure luck, best of 27) | 0.00% | 0.32% |
| 0.5 | 0.0% | 1.4% |
| 1.0 (already excellent) | 2.1% | 23.8% |
| 1.25 | 9.9% | 51.2% |
| 1.5 | 29.2% | 77.9% |

Even a truly excellent strategy fails the current bar about 98% of the time. P1 still lets
through only about 0.3% of zero-skill best-of-27 results. Under P1, the measured Sharpe
needed is about 1.24.

**The honest catch.** Round 2's `vol_target` rule (Sharpe 1.46, DSR probability 0.99)
would clear P1's Sharpe test. I am proposing P1 after seeing that, which is exactly when
you should be most suspicious. Three things limit the risk:
- P1 is the standard published test, not a threshold I invented to fit 1.46.
- The rule must still pass robustness (2x costs, every neighbouring parameter set),
  ideally P2 below, and then the untouched test set once.
- If you approve P1, approve P2 together with it. P2 adds evidence nobody has looked at.

**Recommendation: approve P1 together with P2.**

---

## P2 - Rules must also pass on 2007-2015 (history never used before)

Switch: `pass_criteria.require_long_history: true` (currently `false`). The check runs
and is reported either way, in the robustness table as `long_history_2007_2015`.

**Why.**
- 2016-2024 was an unusually good period for this basket. Buy-and-hold Sharpe was 1.06, far
  above multi-decade norms for diversified portfolios, and there was no 2008-style crash.
- Rules need no training window, so they can be checked on 2007-2015. That period includes
  2008 and 2011.
- No rule has ever been evaluated on 2007-2015, and the rule parameters were fixed in round 2,
  so it is a second, fresh hold-out.
- The rule must make money after costs there, with Sharpe > 0.5 and survival >= 80%.

**Risk.** Yahoo's pre-2010 data for the Indian stocks may contain bad prices. Check the
data-quality table in `LONG_HISTORY.md` (jumps > 25%) before trusting it. Treat it as a
hold-out from now on: never tune on it.

---

## P3 - For rules, "robust to other seeds" becomes "robust to neighbouring parameters" (ACTIVE)

Already active, because without it a rule could never be evaluated at all:
`robustness.rule_required_tests: [costs_2x, param_neighborhood]`.

- **Why seeds don't apply:** a fixed rule has no random seed.
- **What replaces them:** the rule is re-run with each nearby parameter set listed in its
  config (for example vol target 9% / 15%, window 10 / 40 days, band 0.05 / 0.2).
- **How strict it is:** every neighbour must pass (return > 0, Sharpe > 0.5, survival >= 80%).
  This is stricter than the seed test, which looks at the mean over seeds.
- **What neighbours are not:** they are never used to pick parameters.

If you reject P3, tell me what should replace it. For example, add `shifted_dates` to the list.

---

## Note - "27 folds" is really 9 folds

The tables show "27/27 runs" because there are 9 one-year validation folds × 3 seeds.

- *Seeds are not new data:* they repeat the same 9 years with different random training.
- *The fold length (1 year) is fine.* The real limit is the total of 8.8 out-of-sample
  years, which is what sets the 0.34 standard error above.
- *What would help:* more years, which is P2's purpose. More seeds or shorter folds would not.

## Note - pre-registered final-test criteria (already written, reporting only)

`final_test_criteria` in `base_config.yaml` sets the bar for the one-time test: return > 0,
Sharpe > 0.5, survives. It was written before anyone looked at the test set.
`final_test.py` reports MET or NOT MET against it.

Two years of test data have a Sharpe standard error of about 0.7. So the test can expose a
broken strategy, but it cannot prove a good one. Only forward paper trading over more years
can do that.
