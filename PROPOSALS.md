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

---

# Round 4 proposals (P1 stays OFF as you decided; P2 stays ON)

## P4 - Measure PROFIT AT EQUAL RISK, not raw return and not Sharpe alone (NOT active)

**The quant logic.** Every strategy has two dials:
- **Sharpe (efficiency):** how much return you get per unit of risk.
- **Volatility (throttle):** how much risk you run.

Profit ≈ Sharpe × volatility. So "it made less money" tells you nothing until you know how much risk it ran.

Your own numbers (2007-2024, all 18 windows) show this:

| | Sharpe | CAGR as run | max DD |
|---|---|---|---|
| buy & hold (6-asset basket) | 1.10 | 11.6% | 26.2% |
| vol_target | 1.26 | 7.8% | 13.4% |

- `vol_target` ran at roughly half the risk, so it made less money.
- **At equal risk it is the more profitable engine.** At equal volatility it would return about 1.26 / 1.10 ≈ 15% more per unit of risk. This is the rough Sharpe arithmetic; the pipeline computes the exact "CAGR at 10% vol" with financing costs.
- **It is not "just a risk reducer"; it's a better engine running at low throttle.**
- **It is also not a big edge:** 1.26 vs 1.10 over 18 years is well within noise for strategies this similar. That's why the round-4 report shows a bootstrap p-value.

**The kill switch caps the throttle.** A 15% drawdown kill switch inside one-year windows makes much more than about 10% volatility risky; buy & hold at ~11% volatility died in 3 of 18 windows. So the most this project can expect is roughly **CAGR ≈ Sharpe × 10%**:
- Sharpe 0.7 → about 7% a year.
- Sharpe 1.0 → about 10% a year.

More profit requires a higher Sharpe (more independent return sources) or a looser kill switch. Clever timing can't buy it.

**Proposed success test (all must hold; the report shows it as the "P4 (NOT active)" column):**
1. **Beats 60/40 at equal risk.** CAGR at 10% volatility, after costs and leverage financing (T-bill + 1%), above 60/40's CAGR at 10% volatility. 60/40 is the portfolio a sensible investor holds anyway.
2. **The edge is unlikely to be luck.** Paired block-bootstrap p < 0.05 for "Sharpe not above 60/40", and deflated-Sharpe probability > 0.95 with the full trial count (34+).
3. **Survives.** Survival >= 80%, and still profitable at 2x costs.
4. **Confirms.** Passes the pre-registered confirmation on the untouched hold-out (P6).
5. Then the one-time locked test, then **12+ months of forward paper trading** before any real money.

**Why replace "penalized Sharpe > 1".** With 34 trials and an 18-year record, that bar needs a measured Sharpe around 1.6-1.75. Long-only multi-asset strategies have historically delivered roughly 0.5-1.0 (my estimate from the literature, not measured here). Under the current bar the honest expectation for round 4 is again **NO STRATEGY PASSED**, however good a strategy is. P4 asks the question you actually care about ("more profit than the sensible default, at the same risk, and real?") and still controls for luck. It is not a loosening in disguise: points 1, 2 and 4 are new hurdles the current bar does not have.

## P5 - Idle cash earns the T-bill rate (NOT active)

The engine charges T-bill + 1% on borrowed money but pays 0% on idle cash.
- Real brokers pay interest on cash: T-bills paid about 5% in 2007 and in 2023-24.
- Trend rules hold 30-60% cash in bad years, so the current accounting understates them. Buy & hold holds no cash and is unaffected.
- The round-4 table shows "Sharpe if cash earned T-bill" next to the active number, so you can see how much it matters.
- **Switch:** `common.credit_cash_interest: true` in `research/plan_round4.yaml`. That would be a plan change, so it must be decided before the first round-4 run. Otherwise it counts as a new exploratory variant.

## P6 - Approve the confirmation hold-out: mutual funds, 1991-2006 (needs your approval)

Switch: `research.confirmation_approved: true` in `base_config.yaml`.

**What it is.** The ≤ 3 finalists (picked by a pre-registered rule from exploration only) and the benchmarks run once on:
- 10 Vanguard/Fidelity mutual funds, covering US large and small caps, international, emerging markets, long and intermediate Treasuries, corporate bonds, high yield, precious-metals miners and REITs;
- the years 1991-2006, which include the 1994 bond crash, 1998 (LTCM) and the 2000-02 bear market.

No instrument or year of it has been used anywhere in this project.

**Recommended:** approve it BEFORE the first round-4 run. The pipeline then goes exploration → finalists → confirmation mechanically, and nobody sees the exploration results before the finalists are fixed. After it runs once, the finalists are locked (`confirmation_LOCK.json`).

**What I rejected:**
- **Other countries' ETFs:** most start in 1996 or later and are highly correlated with SPY and EFA. Their 2007-2024 years are already "seen" by us.
- **Sector ETFs:** equities only, so they don't fit multi-asset rules.
- **A new walk-forward on 2007-2024:** those years are burned.
- **Forward paper trading alone:** honest but slow. It needs years to say anything, so it runs in parallel, not instead.

**Risks to accept if you approve:**
- Yahoo's mutual-fund history before 2000 may have gaps or bad prices. The confirmation report includes a data-quality table and lists funds that failed to load.
- VGPMX holds gold MINERS, not gold. There's no commodity or dollar fund, so trend/momentum have fewer assets to work with.
- The trend-following literature covers 1991-2006, so the hold-out is fresh for OUR selection, not for the idea itself.
