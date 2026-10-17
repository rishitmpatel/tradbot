# Round 6 research notes: what I looked at, what I tested, what I rejected

Written before any round-6 code ran on real data (the plan, `research/plan_round6.yaml`, was committed first).
The numbers that decide anything come only from your laptop run (`RESEARCH6.md`).

**Labels.** **[V]** = verified: I opened or searched the source on 2026-10-17 and the claim is what it says.
**[M]** = from memory: I believe it is right but did not re-check it this round. **[A]** = an assumption I made
for the model (a number you can change in the plan only through a new, counted trial).

**The Gemini Deep Research document was not attached** (the message contained the placeholder
"[ATTACH OR PASTE THE GEMINI DOCUMENT HERE]"). Below I cover the topics you listed for it from my own
research, and say what is useful for you and what is not.

---

## 1. The overnight effect

**The claim.** In US stocks, almost all of the market's return since the 1990s came from close to the next open;
open to close earned about nothing.
- Cliff, Cooper and Gulen (2008), "Return differences between trading and non-trading hours: like night and day" [M: title and year].
- Kelly and Clark (2011), Journal of Asset Management, "Returns in trading versus non-trading hours: the difference is day and night" [V] https://ideas.repec.org/a/pal/assmgt/v12y2011i2d10.1057_jam.2011.2.html
- Lou, Polk and Skouras (2019), Journal of Financial Economics, "A tug of war: overnight versus intraday expected returns": momentum profits accrue overnight, most other factors intraday [V] https://personal.lse.ac.uk/polk/research/TugOfWar.pdf
- Boyarchenko, Larsen and Whelan (2023), Review of Financial Studies, "The overnight drift": the drift is concentrated around the European open and is linked to dealers' inventory [V] https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr917.pdf

**Why I still expected it to fail for you (written down before testing).**
- **Costs.** You pay a round trip every day, about 250 a year. Studies that include realistic costs find the profit disappears [V]: https://alphaarchitect.com/trading-costs-wipe-out-the-overnight-return-anomaly/ and JOIM "Night Moves" (2024) https://joim.com/wp-content/uploads/emember/downloads/P0753.pdf
- **It may be gone.** The NY Fed's Liberty Street Economics post "The disappearing overnight drift" (1 July 2026) reports that the drift has been about zero since 2021 [V] https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/
- **Real products failed.** The NightShares overnight ETFs (NSPY, NIWM) closed in 2022, after about 14 months [V: same NY Fed post].
- **Stress-day fills.** On 24 August 2015 many US ETFs opened far below the value of what they held, and trading was halted over and over. An overnight strategy *sells at the open*, so it fills at exactly the worst moment. SEC research note (Dec 2015) [V] https://www.sec.gov/marketstructure/research/equity_market_volatility.pdf and DERA white paper (Feb 2016) [V] https://www.sec.gov/about/divisions-offices/division-economic-risk-analysis/staff-papers-analyses/feb2016-dera-white-paper-etf-volatility . The code's "stress fills" column assumes you sell at the day's LOW whenever the gap is -2% or worse.

**Data check added after the plan commit, before any real-data run (stricter, never looser):** a close-to-close
move above 25% in a day also marks a data set UNRELIABLE. A broad index fund never moves that much, so such a move
means a split was not adjusted.

**India.** I found no rigorous study of Nifty overnight vs intraday returns, only blog-level analyses [V: search 2026-10-17] (e.g. https://www.multibagg.ai/market-pulse/articles/nifty-overnight-intraday-returns-study-cmquu8ofy9pn1nz0j1dln1abo). So this is genuinely new data for this project and a clean confirmation set.

**Are free open prices reliable?** Not always. For Indian *indices* on Yahoo, the "open" is often just the previous close [M]. That would create a fake overnight return of zero and push all of the return into "intraday". The code measures three things for every data set: the share of days where open = previous close, open outside the day's high-low range, and missing opens. If any is too high, the data set is marked UNRELIABLE and cannot confirm anything. NIFTYBEES (a traded ETF) should be better than ^NSEI, and the code checks that rather than assuming it.

**Could you actually do it in India?**
- *Selling at the open.* ETFs were **excluded** from the 9:00-9:08 pre-open call auction for years. An exchange circular states that all equities except ETFs joined pre-open from 1 April 2013 [V: MSEI circular 15278, 2024, https://msei.in/SX-Content/Circulars/2024/April/Circular-15278.pdf; that the same applied on NSE is M]. A search found an NSE circular, NSE/CMTR/76041 (28 Aug 2026), on "call auction in pre-open session ... for ETFs", effective 5 Sept 2026 [V: title and dates from search results only; I could not open the text. Look it up in the NSE circular archive]. So in the 2009-2024 data, NIFTYBEES's "open" is roughly the first trade at 9:15. A market order at 9:15 could get that price, minus the spread (the 1 bp-per-side slippage assumption).
- *Buying at the close.* NSE's official close is a 30-minute average, not one trade [M]. SEBI's circular of 16 Jan 2026 introduces a closing auction session [V] https://www.sebi.gov.in/sebi_data/attachdocs/jan-2026/1768576287344.pdf ; check which securities and dates it covers. Brokers describe a post-close session in which orders execute at the closing price [V: Zerodha support] https://support.zerodha.com/category/trading-and-markets/trading-faqs/market-sessions/articles/what-are-pre-market-and-post-market-sessions-and-orders . Check with your broker whether ETFs can use it.
- *BTST.* You sell units bought the day before, before they reach your demat. Brokers allow it, but if your seller fails to deliver, an auction penalty is possible [M]. The DP charge applies on every sell day [V: Zerodha charges page].
- *Tax.* Gains are probably short-term capital gains (STCG, 20%) because the trades settle by delivery and STT is paid. But a tax officer may treat daily trading as business income at your slab rate [V: practice varies, https://tradingqna.com/t/taxation-question-on-btst-trade/1286 , https://www.kotakneo.com/investing-guide/share-market/how-gains-from-intraday-trading-are-taxed/]. Both are reported.
- *US route via LRS.* SPY's open is 7:00 pm and its close 1:30 am IST (in US summer time). Doing that by hand every day is impractical, and broker APIs are off-limits in this project. Gains on foreign ETFs held under 24 months are taxed at your slab rate [V: Budget 2024 sources below]. TCS of 20% applies on remittances above Rs 10 lakh a year; it is refundable but locks up cash [V] https://tax2win.in/guide/section-206c-tax-collection-at-source-tcs , https://www.business-standard.com/amp/budget/news/proposed-tcs-threshold-hike-on-lrs-expected-to-boost-outbound-travel-125020300459_1.html . **Verdict: you cannot practically trade the US version.**

**Costs used (NIFTYBEES, discount broker, delivery)** [V: https://zerodha.com/charges, 2026-10-17]: brokerage 0; STT 0.001% on ETF sells [V: https://www.dspim.com/knowledge-hub/personal-finance-guide/what-is-securities-transaction-tax-stt-applicability-rates-and-rules-explained , https://www.kotakneo.com/investing-guide/articles/what-is-stt/]; NSE transaction charge 0.00307%; SEBI Rs 10 per crore; 18% GST on those charges; stamp duty 0.015% on buys; DP charge Rs 15.34 per scrip per sell day. Plus **[A]** 1 bp slippage per side (about 2 ticks on a Rs 250-300 unit). Total per round trip: about **5.9 bps at Rs 1 lakh, 4.5 bps at Rs 10 lakh, 4.4 bps at Rs 1 crore** (from `india_round_trip`). Over about 248 trading days that is roughly **11%+ a year before tax**. This is the hurdle the overnight return must beat.

**Minimum capital.** The DP charge (about Rs 3,800 a year) costs 3.8% a year on Rs 1 lakh and 0.4% on Rs 10 lakh. Below roughly Rs 5 lakh it alone ruins the idea. Above that, the percentage charges (about 11% a year) still dominate, and more capital does not reduce them.

## 2. Momentum: crash risk, crowding, and one more out-of-sample test

- **Momentum crashes.** Long-short momentum lost heavily in 1932 and in 2009, when beaten-down stocks rebounded after a bear market. Daniel and Moskowitz (2016, JFE, "Momentum crashes") show this happens in "panic states" [M: paper; V: summary https://alphaarchitect.com/avoiding-momentum-crashes/]. Barroso and Santa-Clara (2015, JFE, "Momentum has its moments") scale momentum by its own recent volatility [V] https://ciencia.ucp.pt/en/publications/momentum-has-its-moments/ . Round 5 already scaled every strategy to 10% volatility, which is the Barroso/Santa-Clara idea in long-only form. So round 6 tests the *other* idea: hold the market instead of momentum when the market's 24-month return is below T-bills. Daniel-Moskowitz use the 24-month bear indicator; the parameter is fixed, not tuned.
- **Crowding (the August 2007 "quant quake").** Many quant funds held the same long-short positions and were forced to sell together, losing heavily in a few days before rebounding. Khandani and Lo [V] https://www.nber.org/papers/w14465 , https://web.mit.edu/~alo/www/Papers/august07b.html . For you, as a long-only, monthly index-fund holder, the lesson is that factor ETFs (momentum, value) can lag the market sharply when everyone exits the same trade. The report lists momentum's relative return in Aug 2007, the 2009 rebound, Nov 2016, Nov 2020 and early 2022, plus its 5 worst months.
- **Emerging Markets.** Ken French publishes EM momentum portfolios from about 1991. This project never loaded them, and India is roughly 15-20% of EM [M]. File names verified on the library page [V] https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html : `Emerging_5_Factors_CSV.zip`, `Emerging_Markets_6_Portfolios_ME_Prior_12_2_CSV.zip`. There is no daily EM factors file [V], so the monthly engine is used.
- **Decay is the norm.** Published anomalies earn about 26% less out of sample and 58% less after publication (McLean and Pontiff, 2016, JF) [V in round 5].

## 3. India tax harvesting (an accounting rule, not a market edge)

- Listed equity funds/ETFs since 23 July 2024: STCG 20%; LTCG 12.5% above **Rs 1.25 lakh per financial year** [V] https://zerodha.com/z-connect/business-updates/what-changes-for-investors-after-budget-2024 , https://www.business-standard.com/amp/markets/news/budget-2024-hikes-ltcg-tax-rate-to-12-5-stcg-to-20-stt-on-f-o-also-up-124072300553_1.html . The new Income-tax Act (effective FY2026-27) keeps these capital-gains rules, per [V] https://www.businesstoday.in/amp/union-budget/story/budget-2026-what-deductions-exemptions-and-capital-gains-rules-under-the-new-income-tax-act-mean-for-taxpayers-513819-2026-02-01 , https://www.patronaccounting.com/blog/itr-capital-gains-notifications-2026-what-changed .
- India has no wash-sale rule [M], so selling and re-buying to reset your cost basis is allowed. But a same-day sell and buy of the same ETF is netted by the broker into an intraday trade and your holding never changes [M]. So the tested rule sells at the close and re-buys at the **next day's** close (with an index mutual fund: redeem, then buy again).
- The most it can save is 12.5% x Rs 1.25 lakh x 1.04 = **Rs 16,250 of tax per year**. That matters a lot at Rs 1-10 lakh and very little at Rs 1 crore. The code measures it on 2010-2024 NIFTYBEES prices, with every charge included.
- The simulation applies today's rules to the whole period (the exemption was Rs 1 lakh before FY2024-25, and LTCG on equity was exempt before 2018). It is a "what this rule does to that price path" test, not history.

## 4. The Gemini-document topics: useful for you or not?

| topic | useful to you? | why |
|---|---|---|
| 2007 quant crowding (Khandani-Lo) | **Yes, as a risk check** | Factor funds can underperform together and fast. Size factor tilts modestly, hold them for years, and never use leverage. Checked in the report. |
| Negative WTI oil (20 Apr 2020) | **Yes, as an avoid-list** | Futures-based commodity funds (USO-type) lose money to roll costs and can blow up near expiry [V] https://acfr.aut.ac.nz/__data/assets/pdf_file/0004/541903/WTI-crude-oil-prices_Adrian-2021DMC.pdf , https://finance.yahoo.com/markets/commodities/articles/uso-front-month-oil-strategy-173327031.html . Indian gold ETFs hold physical gold, so they have no roll risk [M]. Avoid commodity futures funds. |
| Latency arbitrage, quote stuffing | No | Microsecond games that need co-location and exchange feeds. A retail investor is the counterparty, not the winner. Lesson: use limit orders, and do not trade in the first and last minutes on volatile days. |
| VPIN (order-flow toxicity) | No | Needs tick-by-tick trade data you do not have, and its predictive value is disputed [M]. |
| Aug 2015 ETF dislocation | **Yes** | Directly tested ("stress fills"). Practical rule: never send a market order for an ETF at the open on a panic day; use a limit order near its value. |
| Deflated Sharpe ratio | **Yes, already used** | Every report shows the deflated-Sharpe probability with all 46 trials. |

## 5. Ideas considered and NOT tested (and why)

| idea | why not |
|---|---|
| Selling index options (F&O) | SEBI: 93% of individual F&O traders lost money in FY22-FY24 [V] https://www.outlookmoney.com/invest/93-of-individual-traders-suffered-losses-in-fo-in-last-3-years-sebi-study . Tail risk, margin calls, and business-income tax. Free daily data cannot price the risk honestly. |
| Intraday / scalping / HFT | Speculative-business tax [V Kotak link above], costs, and no edge in daily data (round 1: autocorrelation ~0). |
| Sell-in-May, pre-holiday, weekday effects | Calendar patterns with weak economic reasons and many tries behind them. Round 5's turn-of-month test lost to costs and tax. |
| Earnings drift (PEAD), news trading | Needs point-in-time earnings data and many single stocks. Not free, and too concentrated for small capital. |
| FX carry, global macro | Indian residents cannot easily hold foreign-currency carry positions. LRS forbids margin trading abroad [M]. |
| Crypto | Flat 30% tax, 1% TDS, and losses cannot be set off [M: Budget 2022 rules]. Any edge is taxed away. |
| Sovereign Gold Bonds bought on the exchange | The tax exemption on redemption no longer applies to secondary-market buyers [V] https://www.nism.ac.in/blog/how-budget-2026-changes-sovereign-gold-bond-sgb-taxation , https://taxguru.in/?p=1023039 . |
| Indian international ETFs (MON100 etc.) at a premium | They traded well above NAV when fund houses hit overseas limits [M]. You cannot short the premium; buying at a premium is a loss. |
| Indian factor indices (Nifty200 Momentum 30, Value, Low Vol) | Their pre-launch history is back-filled and the ETFs are young [M]. EM momentum is the cleaner test. A real momentum index fund is the way to *implement* momentum if it ever confirms. |
| RL | No specific, testable reason it would add anything here (rounds 1-3: it learned nothing at realistic costs). The code is kept. |
