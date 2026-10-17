# RESEARCH - round 5

## Round 5 research: century-long premia (Ken French data, pre-registered plan)

- Plan `research/plan_round5.yaml` sha `36352d850604`; trial status: R5_H1_market_trend=registered, R5_H2_big_momentum=registered, R5_H3_big_value=registered, R5_H4_value_momentum=registered, R5_H5_value_momentum_trend=registered, R5_H6_turn_of_month=registered, R5_D1_P5_recheck=diagnostic  
- **Trials counted for the Sharpe penalty: 41** (all rounds, incl. 7 round-5)  
- Exploration: US 1963-07-31 -> 2024-09-30 (61 years, monthly). Every strategy AND the market are scaled to ~10% volatility (rest in T-bills, no leverage), 15 bps per position change, fund/turnover drag per sleeve, 15% kill switch per calendar year.  
- *Excess Sharpe* = Sharpe of returns ABOVE T-bills (correct accounting; idle cash earns T-bills here).

| strategy | trial | excess Sharpe | vs market | p (not better) | CAGR | excess CAGR | vol | max DD | survived | avg exposure | Sharpe w/o kill switch | CAGR w/o kill switch | excess Sharpe 2x costs | penalized Sharpe | DSR prob | finalist | verdict | P9 (NOT active) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R5_H1_market_trend | registered | 0.44 | +0.04 | 0.306 | +8.1% | +3.5% | 8.6% | 20.4% | 61/62 | 0.56 | 0.44 | +8.1% | 0.41 | 0.06 | 0.68 | no | FAIL | no |
| R5_H2_big_momentum | registered | 0.50 | +0.10 | 0.052 | +9.6% | +5.0% | 10.9% | 29.6% | 57/62 | 0.69 | 0.52 | +10.0% | 0.43 | 0.12 | 0.82 | yes | FAIL | no |
| R5_H3_big_value | registered | 0.38 | -0.01 | 0.570 | +8.2% | +3.6% | 11.0% | 35.8% | 54/62 | 0.69 | 0.42 | +8.8% | 0.34 | 0.01 | 0.52 | no | FAIL | no |
| R5_H4_value_momentum | registered | 0.47 | +0.07 | 0.123 | +9.2% | +4.6% | 10.9% | 33.3% | 57/62 | 0.71 | 0.49 | +9.5% | 0.41 | 0.09 | 0.75 | yes | FAIL | no |
| R5_H5_value_momentum_trend | registered | 0.52 | +0.12 | 0.112 | +9.0% | +4.4% | 9.1% | 26.1% | 60/62 | 0.54 | 0.53 | +9.2% | 0.46 | 0.14 | 0.85 | yes | FAIL | no |
| R5_H6_turn_of_month | registered | 0.06 | -0.33 | 0.992 | +4.6% | +0.2% | 7.0% | 26.1% | 61/62 | 0.99 | 0.06 | +4.6% | -0.49 | -0.31 | 0.01 | no | FAIL | no |
| bench:market | benchmark | 0.39 | n/a | n/a | +8.2% | +3.6% | 10.3% | 28.4% | 55/62 | 0.73 | 0.38 | +8.2% | n/a | n/a | n/a | - | benchmark | - |

*p*: paired block-bootstrap (12-month blocks) probability that the strategy's excess Sharpe is NOT above the market's. *P9* (proposal, not active): p < 0.05 / number of round-5 hypotheses, deflated-Sharpe prob > 0.95 with ALL trials, survives, profitable at 2x costs, finalist, confirmed.

### Does the edge decay? Excess Sharpe minus the market's, by sub-period

| strategy | 1963-1990 | 1991-2006 | 2007-now |
|---|---|---|---|
| R5_H1_market_trend | +0.10 | +0.07 | -0.10 |
| R5_H2_big_momentum | +0.22 | +0.12 | -0.10 |
| R5_H3_big_value | +0.20 | +0.14 | -0.45 |
| R5_H4_value_momentum | +0.22 | +0.17 | -0.25 |
| R5_H5_value_momentum_trend | +0.25 | +0.26 | -0.23 |
| R5_H6_turn_of_month | +0.09 | -0.51 | -0.84 |

### Why each hypothesis passed or failed

- **R5_H1_market_trend** (registered): FAIL - penalized Sharpe 0.06 (raw 0.44 - haircut 0.37) not > 1.0; not a finalist
- **R5_H2_big_momentum** (registered): FAIL - penalized Sharpe 0.12 (raw 0.50 - haircut 0.37) not > 1.0
- **R5_H3_big_value** (registered): FAIL - penalized Sharpe 0.01 (raw 0.38 - haircut 0.37) not > 1.0; excess Sharpe 0.38 not above the market's 0.39; not a finalist
- **R5_H4_value_momentum** (registered): FAIL - penalized Sharpe 0.09 (raw 0.47 - haircut 0.37) not > 1.0; confirmation failed: US 1927-1963: advantage not significant (p 0.20 >= 0.1) | Developed_ex_US: ok
- **R5_H5_value_momentum_trend** (registered): FAIL - penalized Sharpe 0.14 (raw 0.52 - haircut 0.37) not > 1.0; confirmation failed: US 1927-1963: advantage not significant (p 0.21 >= 0.1) | Developed_ex_US: ok
- **R5_H6_turn_of_month** (registered): FAIL - penalized Sharpe -0.31 (raw 0.06 - haircut 0.37) not > 1.0; excess Sharpe 0.06 not above the market's 0.39; not profitable over T-bills at 2x costs; not a finalist

### Finalists (pre-registered: survival >= 80%, beats T-bills at 2x costs, top 3 by excess Sharpe)

R5_H5_value_momentum_trend, R5_H2_big_momentum, R5_H4_value_momentum

### Confirmation (US 1927-1963 + other developed markets; never used before)


**US 1927-1963** (required)

| strategy | period | excess Sharpe | p (not better) | CAGR | max DD | survived | verdict |
|---|---|---|---|---|---|---|---|
| bench:market | 1927-07-31 -> 1963-06-30 | 0.70 | n/a | +8.7% | 38.2% | 32/37 | benchmark |
| R5_H5_value_momentum_trend | 1928-01-31 -> 1963-06-30 | 0.76 | 0.214 | +8.2% | 25.4% | 35/36 | FAIL: advantage not significant (p 0.21 >= 0.1) |
| R5_H2_big_momentum | 1928-01-31 -> 1963-06-30 | 0.75 | 0.051 | +9.5% | 35.9% | 33/36 | PASS |
| R5_H4_value_momentum | 1928-01-31 -> 1963-06-30 | 0.71 | 0.200 | +9.0% | 36.7% | 31/36 | FAIL: advantage not significant (p 0.20 >= 0.1) |

**Developed_ex_US** (required)

| strategy | period | excess Sharpe | p (not better) | CAGR | max DD | survived | verdict |
|---|---|---|---|---|---|---|---|
| bench:market | 1991-07-31 -> 2024-09-30 | 0.22 | n/a | +4.2% | 33.8% | 29/34 | benchmark |
| R5_H5_value_momentum_trend | 1992-01-31 -> 2024-09-30 | 0.39 | n/a | +5.4% | 23.7% | 33/33 | PASS |
| R5_H2_big_momentum | 1992-01-31 -> 2024-09-30 | 0.36 | n/a | +5.8% | 29.7% | 29/33 | PASS |
| R5_H4_value_momentum | 1992-01-31 -> 2024-09-30 | 0.29 | n/a | +5.0% | 30.4% | 29/33 | PASS |

**Japan** (informational)

| strategy | period | excess Sharpe | p (not better) | CAGR | max DD | survived | verdict |
|---|---|---|---|---|---|---|---|
| bench:market | 1991-07-31 -> 2024-09-30 | 0.03 | n/a | +2.2% | 34.6% | 29/34 | benchmark |
| R5_H5_value_momentum_trend | 1992-01-31 -> 2024-09-30 | 0.30 | n/a | +4.5% | 19.9% | 33/33 | PASS |
| R5_H2_big_momentum | 1992-01-31 -> 2024-09-30 | 0.10 | n/a | +3.0% | 35.6% | 29/33 | PASS |
| R5_H4_value_momentum | 1992-01-31 -> 2024-09-30 | 0.05 | n/a | +2.4% | 37.7% | 29/33 | FAIL: excess return -0.1%/yr |

**Asia_Pacific_ex_Japan** (informational)

| strategy | period | excess Sharpe | p (not better) | CAGR | max DD | survived | verdict |
|---|---|---|---|---|---|---|---|
| bench:market | 1991-07-31 -> 2024-09-30 | 0.36 | n/a | +5.9% | 38.3% | 28/34 | benchmark |
| R5_H5_value_momentum_trend | 1992-01-31 -> 2024-09-30 | 0.49 | n/a | +6.7% | 19.1% | 31/33 | PASS |
| R5_H2_big_momentum | 1992-01-31 -> 2024-09-30 | 0.46 | n/a | +7.3% | 35.9% | 28/33 | PASS |
| R5_H4_value_momentum | 1992-01-31 -> 2024-09-30 | 0.45 | n/a | +7.0% | 34.1% | 28/33 | PASS |

Overall: R5_H5_value_momentum_trend: NOT confirmed; R5_H2_big_momentum: CONFIRMED; R5_H4_value_momentum: NOT confirmed

### What an Indian retail investor would keep (last 20 years of the exploration pattern)

US returns used as a stand-in for the TRADING PATTERN; taxes/charges per india_costs.py (FY2025-26 rules, slab 30%, liquidated at the end so buy & hold also pays its tax).

| strategy | pre-tax CAGR | route | Rs 1 lakh | Rs 10 lakh | Rs 1 crore |
|---|---|---|---|---|---|
| R5_H5_value_momentum_trend | +6.4% | india_listed | +5.9% (tax 15%, costs 8.4%) | +5.6% (tax 27%, costs 5.1%) | +5.4% (tax 35%, costs 4.7%) |
| R5_H5_value_momentum_trend | +6.4% | us_lrs | +5.0% (tax 48%, costs 5.9%) | +5.0% (tax 48%, costs 5.9%) | +5.0% (tax 48%, costs 5.9%) |
| R5_H2_big_momentum | +7.9% | india_listed | +7.5% (tax 14%, costs 5.5%) | +7.2% (tax 33%, costs 3.4%) | +6.8% (tax 47%, costs 3.1%) |
| R5_H2_big_momentum | +7.9% | us_lrs | +6.5% (tax 60%, costs 5.0%) | +6.5% (tax 60%, costs 5.0%) | +6.5% (tax 60%, costs 5.0%) |
| R5_H4_value_momentum | +6.6% | india_listed | +6.2% (tax 11%, costs 6.9%) | +6.1% (tax 21%, costs 2.9%) | +5.7% (tax 35%, costs 2.4%) |
| R5_H4_value_momentum | +6.6% | us_lrs | +5.4% (tax 44%, costs 4.1%) | +5.4% (tax 44%, costs 4.1%) | +5.4% (tax 44%, costs 4.1%) |
| bench:market | +7.3% | india_listed | +7.1% (tax 9%, costs 2.4%) | +6.8% (tax 28%, costs 0.7%) | +6.5% (tax 40%, costs 0.5%) |
| bench:market | +7.3% | us_lrs | +6.2% (tax 49%, costs 2.5%) | +6.2% (tax 49%, costs 2.5%) | +6.2% (tax 49%, costs 2.5%) |
| passive:buy_and_hold_market | +10.8% | india_listed | +10.3% (tax 72%, costs 0.0%) | +10.1% (tax 86%, costs 0.0%) | +10.1% (tax 88%, costs 0.0%) |
| passive:buy_and_hold_market | +10.8% | us_lrs | +10.1% (tax 88%, costs 3.9%) | +10.1% (tax 88%, costs 3.9%) | +10.1% (tax 88%, costs 3.9%) |

Cells: after-cost, after-tax CAGR (total tax and charges as a share of starting capital). *bench:market* is the market scaled to 10% volatility (it trades monthly); *passive:buy_and_hold_market* is simply holding an index fund at full risk and selling once at the end - the 'do nothing' alternative (higher risk, so compare its CAGR with care).

