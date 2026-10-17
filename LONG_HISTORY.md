# LONG HISTORY (rules only) - informational, NOT part of pass/fail

Generated 2026-10-18 00:58. One-year windows from 2007-01-03 to 2024-10-15, fresh 100,000 each window, same costs and 15% kill switch as the main pipeline. Locked test set (2024-10-16 onwards) NOT used.

Rules need no training, so every year after the 2-year warm-up is out-of-sample for them. **2007-2015 was never used to evaluate or choose any rule** - treat it as a second hold-out and do not tune on it.

## 2007-2015 (never used before)

| strategy | Sharpe | CAGR | max DD (stitched) | survived windows |
|---|---|---|---|---|
| baseline:buy_and_hold | 1.13 | +13.6% | 26.2% | 8/9 |
| baseline:ma_crossover | 1.18 | +10.8% | 8.8% | 9/9 |
| baseline:mean_reversion | 0.29 | +1.9% | 20.0% | 8/9 |
| baseline:momentum | 1.01 | +9.0% | 13.9% | 9/9 |
| baseline:vol_target | 1.05 | +6.4% | 13.4% | 9/9 |
| baseline:trend_vol_target | 1.02 | +5.3% | 6.2% | 9/9 |
| r3_rule_portfolio_vt | 1.04 | +8.2% | 19.8% | 8/9 |
| r3_rule_vol_target | 1.05 | +6.4% | 13.4% | 9/9 |

## 2016-2024 (rounds 1-2 validation period)

| strategy | Sharpe | CAGR | max DD (stitched) | survived windows |
|---|---|---|---|---|
| baseline:buy_and_hold | 1.08 | +9.6% | 18.4% | 7/9 |
| baseline:ma_crossover | 0.96 | +7.6% | 17.1% | 8/9 |
| baseline:mean_reversion | 0.04 | +0.0% | 22.2% | 8/9 |
| baseline:momentum | 0.64 | +5.0% | 22.5% | 8/9 |
| baseline:vol_target | 1.47 | +9.4% | 13.0% | 9/9 |
| baseline:trend_vol_target | 1.06 | +5.7% | 11.5% | 9/9 |
| r3_rule_portfolio_vt | 1.43 | +11.6% | 14.9% | 9/9 |
| r3_rule_vol_target | 1.47 | +9.4% | 13.0% | 9/9 |

## all

| strategy | Sharpe | CAGR | max DD (stitched) | survived windows |
|---|---|---|---|---|
| baseline:buy_and_hold | 1.10 | +11.6% | 26.2% | 15/18 |
| baseline:ma_crossover | 1.08 | +9.2% | 17.1% | 17/18 |
| baseline:mean_reversion | 0.17 | +1.0% | 22.2% | 16/18 |
| baseline:momentum | 0.83 | +7.0% | 22.5% | 17/18 |
| baseline:vol_target | 1.26 | +7.8% | 13.4% | 18/18 |
| baseline:trend_vol_target | 1.04 | +5.5% | 11.5% | 18/18 |
| r3_rule_portfolio_vt | 1.23 | +9.9% | 19.8% | 17/18 |
| r3_rule_vol_target | 1.26 | +7.8% | 13.4% | 18/18 |

## Return per one-year window (DEAD = kill switch hit)

| window start | baseline:buy_and_hold | baseline:ma_crossover | baseline:mean_reversion | baseline:momentum | baseline:vol_target | baseline:trend_vol_target | r3_rule_portfolio_vt | r3_rule_vol_target |
|---|---|---|---|---|---|---|---|---|
| 2007-01-03 | +29.2% | +28.8% | +6.7% | +28.2% | +14.1% | +13.2% | +17.2% | +14.1% |
| 2008-01-03 | -15.9% DEAD | -1.3% | -17.2% DEAD | -8.2% | -9.3% | -3.1% | -15.9% DEAD | -9.3% |
| 2009-01-03 | +56.1% | +28.9% | +7.9% | +10.4% | +15.9% | +4.3% | +19.7% | +15.9% |
| 2010-01-03 | +20.7% | +13.7% | +4.1% | +16.4% | +12.9% | +10.2% | +18.8% | +12.9% |
| 2011-01-03 | +3.0% | -0.3% | +0.3% | +3.3% | +1.9% | +2.4% | +5.9% | +1.9% |
| 2012-01-03 | +11.5% | +7.7% | +1.9% | +2.3% | +7.7% | +1.8% | +10.2% | +7.7% |
| 2013-01-03 | +16.0% | +20.2% | +4.0% | +22.4% | +8.9% | +13.9% | +9.8% | +8.9% |
| 2014-01-03 | +13.5% | +9.9% | +3.1% | +11.8% | +8.9% | +7.7% | +13.8% | +8.9% |
| 2015-01-03 | +1.8% | -4.6% | +8.7% | -0.8% | -1.2% | -1.5% | -0.5% | -1.2% |
| 2016-01-03 | +6.8% | +0.3% | -0.8% | -3.5% | +3.9% | -3.0% | +8.3% | +3.9% |
| 2017-01-03 | +26.2% | +22.8% | +7.1% | +22.3% | +19.9% | +17.2% | +22.7% | +19.9% |
| 2018-01-03 | +9.5% | +9.9% | +2.6% | +7.3% | +5.9% | +3.8% | +4.9% | +5.9% |
| 2019-01-03 | +28.3% | +18.4% | +7.7% | +19.9% | +16.7% | +11.6% | +24.4% | +16.7% |
| 2020-01-03 | -12.9% DEAD | -12.9% DEAD | -16.5% DEAD | -12.9% DEAD | +12.2% | +9.2% | +11.7% | +12.2% |
| 2021-01-03 | +15.3% | +14.5% | +3.8% | +10.1% | +10.9% | +7.3% | +12.5% | +10.9% |
| 2022-01-03 | -14.9% DEAD | -7.9% | -7.6% | -13.2% | -9.5% | -8.7% | -11.0% | -9.5% |
| 2023-01-03 | +19.9% | +11.9% | +2.9% | +5.6% | +13.4% | +4.3% | +16.5% | +13.4% |
| 2024-01-03 | +15.7% | +15.7% | +3.4% | +15.3% | +11.6% | +11.0% | +16.3% | +11.6% |

## Data quality (long history)

| ticker | rows | start | end | jumps > 25% |
|---|---|---|---|---|
| SPY | 5482 | 2005-01-03 | 2026-10-16 | - |
| QQQ | 5482 | 2005-01-03 | 2026-10-16 | - |
| GLD | 5482 | 2005-01-03 | 2026-10-16 | - |
| TLT | 5482 | 2005-01-03 | 2026-10-16 | - |
| RELIANCE.NS | 5379 | 2005-01-03 | 2026-10-16 | 2005-07-28:+1.48, 2005-07-29:-1.47 |
| TCS.NS | 5379 | 2005-01-03 | 2026-10-16 | 2005-07-28:+0.69, 2005-07-29:-0.71 |
