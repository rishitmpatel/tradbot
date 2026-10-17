# quant-rl-bot

A reinforcement-learning trading **research** pipeline: PPO and DQN agents trade a
small basket of ETFs and Indian stocks on daily data. They are judged by
walk-forward out-of-sample results after costs, a survival ("agent dies") rule,
robustness tests and a penalty for trying many variants.

> **Paper trading and research only.** There is no broker connection and no real
> money. Nothing here is financial advice. The goal is to *find out honestly*
> whether any variant has a consistent, risk-adjusted edge after costs. The
> expected answer is often "no".

> **About the code's own testing:** it was developed and tested only on
> **synthetic (fake) prices**, because the build sandbox could not download market data.
> Those runs only showed that the code works. They are **not results**. Real numbers
> appear only after you run it on your machine.

---

## 1. Windows setup (exact steps)

You need Windows 10/11, about 3 GB of free disk space, and internet for the first
data download. Everything runs on the CPU, so no GPU is needed.

1. **Install Python 3.11** from <https://www.python.org/downloads/windows/>
   (the "Windows installer (64-bit)"). On the first installer screen, tick
   **"Add python.exe to PATH"**.
2. **Install Git** from <https://git-scm.com/download/win> (keep the defaults).
3. Open **PowerShell** (Start menu → type `PowerShell`) and run the commands below. The code
   currently lives on branch `claude/wonderful-fermi-fhjd5s` of `rishitmpatel/tradbot`.
   If you later move it to its own `quant-rl-bot` repo, clone that instead.

```powershell
cd $HOME\Documents
git clone -b claude/wonderful-fermi-fhjd5s https://github.com/rishitmpatel/tradbot.git quant-rl-bot
cd quant-rl-bot
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

   If PowerShell says *running scripts is disabled*, run this once and then repeat the
   last line:
   `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

4. Install the packages (takes a few minutes; torch is large):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Check that everything works (about 1 minute, uses fake data, no internet needed):

```powershell
python -m pytest -q
```

   You should see `36 passed`.

6. Download the real data and lock the test set:

```powershell
python data.py --extra
```

   This writes CSVs to `data\raw\` and creates `data\raw\split_lock_yahoo.json`.
   **Never edit or delete that lock file.** It fixes which final 2 years form the
   untouchable test set.

7. Run a quick **smoke test** of the full pipeline on real data (about 5–10 minutes):

```powershell
python run_all.py --smoke
```

   It writes `RESULTS_SMOKE_REAL.md`. The numbers mean nothing because of the tiny
   training budget. This step only checks that the pipeline works.

8. **Full run.** There is no RL training in the queue, so it takes **about 10-25 minutes**, mostly the
   first downloads (about 25 Yahoo tickers and 16 small Ken French files). RL configs in `configs/` would make
   it take hours.

```powershell
python run_all.py
```

   One command does everything: data, self-checks, round-4 research, the confirmation hold-out
   (only if you approved it in `base_config.yaml`), and the forward paper tracker. Run it again
   any day to extend the paper-trading log; finished work is reused.

   Alternatively, double-click `run_overnight.bat`.
   Before you leave it, stop Windows from sleeping: **Settings → System → Power →
   Sleep: Never** while plugged in, or run `powercfg /change standby-timeout-ac 0`.
   If the run is interrupted (reboot, sleep, Ctrl+C), **run the same command again**.
   It skips every run that already finished.

9. Read **`RESULTS.md`**.

10. **Only if `RESULTS.md` says a strategy PASSED**, run the one-time final test:

```powershell
python final_test.py
```

   It writes `FINAL_TEST.md`. It refuses to run a second time. If nothing passed, it
   does not touch the test set.

---

## 2. What the pipeline does

```
data.py          Yahoo Finance daily prices (2010 onwards) -> data/raw/*.csv, test split lock
features.py      log returns, momentum, volatility, RSI, MA ratio (causal, rolling z-score)
backtest.py      P&L engine: 0.10% commission + 0.05% slippage, no lookahead, survival rule, metrics
baselines.py     buy & hold, MA crossover (50/200), mean reversion (20d z), momentum (12-1)
trading_env.py   Gymnasium environment (state / action / reward explained in comments)
overlay.py       round 2: rule-based base position the agent scales (residual RL)
rules.py         round 3: rules as candidates (per-asset rules + correlation-aware portfolio rule)
long_history.py  round 3: rules over 2007-2024 incl. 2008 (informational unless P2 approved)
rl_signal_ladder.py  round 3: synthetic experiment - how strong must a signal be for PPO to find it?
PROPOSALS.md     changes to the pass bar / validation that need YOUR approval (P1-P6)
research/plan_round4.yaml  round 4: the PRE-REGISTERED hypotheses (committed before any real run)
allocation.py    round 4: multi-asset allocation rules (equal weight, risk parity, trend, momentum, ensemble)
research.py      round 4: exploration -> finalists -> confirmation hold-out -> report (called by run_all.py)
paper_trade.py   forward paper-trading tracker (round-4 + round-5 finalists), append-only CSV log in paper/
research/plan_round5.yaml  round 5: the PRE-REGISTERED hypotheses (committed before any real run)
factor_data.py   round 5: Ken French Data Library downloader/parser (free, 1926 onwards)
factor_research.py round 5: century-long premia: exploration -> finalists -> confirmation -> report
india_costs.py   round 5: what an Indian retail investor keeps after NSE charges, STT, stamp duty and capital-gains tax
agents.py        PPO + DQN from stable-baselines3, config-driven
walkforward.py   folds, per-run training/evaluation, checkpoint/resume, summaries
evaluation.py    multiple-testing (deflated Sharpe) penalty, pass/fail rules
robustness.py    2x costs, other seeds, other assets, shifted dates
run_all.py       the whole thing unattended -> RESULTS.md
final_test.py    one-time evaluation on the locked final 2 years -> FINAL_TEST.md
inspect_agent.py prints Q-values / policy probabilities day by day (for learning)
selfcheck.py     lookahead / accounting checks run before every pipeline run
synthetic_data.py fake prices for testing the code only
```

### Data and the locked test set
* Basket: SPY, QQQ, GLD, TLT, RELIANCE.NS, TCS.NS. Data uses dividend- and
  split-adjusted closes and starts on 2010-01-01.
* The **last 2 years** become the test set. Their start date is written once to
  `data/raw/split_lock_yahoo.json`. All development code (`run_all.py`,
  `robustness.py`) only receives data **before** that date. This is enforced in
  `data.dev_view` and asserted again on every fold.

### No lookahead
* Features for day *t* use closes up to *t* only. At startup, `selfcheck.py`
  recomputes features with the future cut off and checks that they are identical.
* A position decided at the close of day *t* earns the return from *t* to *t+1*.
  The self-check also feeds the engine a cheating "oracle". The engine must flag
  it, and a lagged copy of it must not look like an oracle.
* The RL environment and the backtest engine must produce identical P&L. This is
  checked as well.

### Costs
Every unit of position change pays `commission + slippage` = 0.15% of the traded
notional. Robustness testing doubles this.

### Portfolio and the survival rule ("agent dies")
Each run starts with **100,000 paper capital**, split equally across the assets.
Each asset gets its own sleeve, and sleeves are not rebalanced. If total equity
falls **more than 15% below its running peak** at any close, the bot liquidates
(paying costs) and halts for the rest of that run. The run is then marked **DEAD**.
Profits made before that point are kept. `RESULTS.md` reports how many runs
survived, for example `24/27`.

### Walk-forward validation
Development data is split into expanding training windows followed by 1-year
validation windows. Validation years run from roughly 2016 up to the test start,
with a 5-business-day embargo before each one. Each (fold, seed) pair is one run.
A fresh agent is trained on the training window and trades only the following
year. For each seed, the validation years are stitched into a single
out-of-sample equity curve.

### Pass/fail (all must hold)
| check | rule |
|---|---|
| OOS return | stitched walk-forward return after costs > 0 (mean over seeds) |
| Sharpe | **penalized** validation Sharpe > 1.0 |
| survival | ≥ 80% of walk-forward runs never die |
| robustness | 2x costs **and** new seeds: return > 0, Sharpe > 0.5, survival ≥ 80% |
| sanity | raw Sharpe > 2.5 ⇒ automatic FAIL as "suspicious – assume bug/leakage" |
| vs buy-and-hold (round 2) | OOS Sharpe above buy-and-hold's on the same windows |
| vs its own rule (round 2) | an overlay agent must beat its rule run without the agent |

*Penalized Sharpe* = validation Sharpe − the Sharpe that the best of N strategies
with zero skill would show by luck. Here N counts every variant ever tried, across
all rounds, as recorded in `results/real/trial_ledger.json`. The formula follows
Bailey & López de Prado's deflated Sharpe ratio. The deflated-Sharpe probability is
reported too. More variants mean a bigger haircut. This is why the pipeline caps
each round at **30 variants**.

Other assets (IWM, EFA, INFY.NS, HDFCBANK.NS) and shifted fold dates are reported
for information. You can make them mandatory in `robustness.required_tests`.

---

## 3. Round 2: what round 1 taught us and what changed

**Round 1 result:** no strategy passed. Buy-and-hold had Sharpe 1.06 and +120.9%
and beat every RL agent. `ppo_pure_return` made the most money of the RL agents
but died in 6 of 27 runs. The variants with drawdown penalties learned to hold cash.

**Why the agents held cash (the reward was the problem).** The default reward
subtracted `0.5 × (new drawdown)` every day. Near a peak, any down day creates new
drawdown, so the penalty is roughly `0.5 × the loss part of each day's return`. With
about 1% daily volatility, that loss part averages about 0.4% per day (counting up
days as 0), so the penalty is about 0.2% per day. The reward for being long is about +0.04% per day
(the equity premium). The penalty is about 5 times larger than the reward. Gains
earn nothing extra back because recovering from a drawdown is not rewarded. So
"always in cash" was the **correct** answer to the question the reward asked.
*Lesson: the agent optimises exactly what you write, not what you meant.*

**Why `ppo_pure_return` died.** Its reward was profit only, with no death penalty
and no early stop, so it never experienced the kill switch during training. It
learned to be mostly long, like buy-and-hold, but at full size. Market crashes
then pushed the equal-weight basket beyond -15%, and it paid trading costs for
timing noise on top. That is buy-and-hold with extra costs and no safety margin.

**Why buy-and-hold won.** Being long earns the equity premium for free. A from-scratch
agent has to discover that from noisy daily rewards in 60k steps, and every flip
costs 0.15%. It is much easier to start from a sensible rule and learn only when to
deviate from it.

**What changed (all config-driven; round-1 behaviour is still the default):**

| knob | what it does | why |
|---|---|---|
| `env.base_signal` + `env.vol_target` | position = rule × agent's multiplier (`overlay.py`) | the agent starts at buy-and-hold or trend and only learns to cut risk |
| `reward.relative_to` | reward = my return − benchmark return | cash is no longer free: cash in a rising market scores negative |
| `reward.drawdown_soft_limit` | drawdown below e.g. 8% is not penalised | stop punishing normal wiggles, punish only deep drawdowns |
| `reward.type: differential_sharpe` | reward = how much today improves a running Sharpe | optimise risk-adjusted return directly instead of hand-tuned penalties |
| `env.rebalance_band` | skip position changes smaller than 0.1 | vol-targeting otherwise trades tiny amounts every day |
| `env.decision_interval` | decide every N days | less noise to chase, fewer trades |
| new baselines `vol_target`, `trend_vol_target` | the same rules with no RL | if RL cannot beat these, RL added nothing |
| pass: beat buy-and-hold Sharpe, beat own rule | extra requirements | the honest question is "better than the simple thing?" |
| `low-risk match` column | Sharpe about B&H's with ≤60% of its drawdown | useful to know, but **not** a pass |

Nothing in the pass criteria was loosened. Changing the bar after seeing results
would be moving the goalposts. Round-1 configs moved to `configs/round1/` and still
count toward the multiple-testing penalty: N = 20 now, so the Sharpe haircut is
bigger than in round 1. Expect "NO STRATEGY PASSED" again. The useful information
is in the `Sharpe - B&H`, `DD / B&H DD` and `rule-only Sharpe` columns.

**The 8 round-2 variants (`configs/r2_*.yaml`, each file explains its reason):**

| variant | idea | tests the round-1 finding... |
|---|---|---|
| `r2_overlay_bh_ppo` | PPO scales vol-targeted buy-and-hold by 0/0.5/1; reward = beat the rule; DD penalty only beyond 8% | B&H won → build on it |
| `r2_overlay_bh_dqn` | same with DQN | does it depend on the algorithm? |
| `r2_overlay_trend_ppo` | same, but the rule is 12-1 momentum | a trend rule already avoids long bear markets |
| `r2_overlay_bh_dsr_ppo` | overlay + differential Sharpe reward, no DD penalty | penalties caused cash → reward Sharpe directly |
| `r2_dsr_ppo` | no overlay, Sharpe reward | isolates the reward change from the overlay change |
| `r2_active_vs_bh_ppo` | no overlay, reward = return − B&H return, DD penalty beyond 8% | target: match B&H with less drawdown |
| `r2_overlay_bh_weekly_ppo` | overlay, decides every 5 days | costs + daily noise |
| `r2_voltarget_pure_return_ppo` | round-1 pure-return reward with vol-targeted sizing | was the dying only a sizing problem? |

## 3c. Round 3: stop searching with RL, test the simple rule properly

**Round 2 facts** (from RESULTS.md, real data):
- **RL lost to its own rule every time.** Every overlay agent scored below its rule with the
  agent removed (rule 1.46; agents 0.48 to 1.14).
- **The agents traded more.** Turnover was 5.7 to 11.7 per year, against 3.0 for the rule.
- **Seeds disagreed.** For `r2_overlay_bh_ppo` the mean return was +24% but the worst seed
  only +13%. Agents trained with different seeds learned different things, which is what
  fitting noise looks like.
- **The plain `vol_target` rule did best:** Sharpe 1.46, survived 9/9, DSR probability 0.99.
  It still failed the bar, because its penalized Sharpe was 0.82.
- **The trend filter hurt.** `trend_vol_target` scored 1.04 against 1.46 without it.

**Diagnosis: why RL fails here** (evidence for each in the final report and below):
1. **Little left to learn.** The features are price-only, and the one thing they reliably
   predict is volatility. Vol targeting already uses that (+0.40 Sharpe over buy-and-hold).
   Every other signal tried (trend, momentum, mean reversion) did not beat buy-and-hold here.
2. **The signal is too weak for the trading costs. This is the main cause.** The signal-ladder
   experiment below shows PPO *can* learn: with zero costs it found planted signals as weak as
   φ = 0.05. With the real 0.15% costs, even a trader who knows the signal perfectly loses money
   at φ = 0.05 and barely breaks even at φ = 0.10. A daily timing signal has to be strong before
   trading it pays.
3. **Too few examples of "when to cut risk".** 8.8 validation years contain only a handful of
   real sell-offs. The seed disagreement and the 2-4x turnover are what fitting noise looks like.
4. **The reward was not the remaining bottleneck.** Round 2 changed it three ways (relative,
   Sharpe-based, pure return). All lost to the rule, and the one closest to the rule
   (Sharpe reward) lost least.
5. **The period flatters everything long-biased.** Buy-and-hold Sharpe 1.06 is unusually
   high, and there was no 2008. This doesn't explain RL's failure, but it means
   `vol_target`'s 1.46 is probably optimistic.

`rl_signal_ladder.py` tests point 2 directly on synthetic data with a planted signal. These
results come from my sandbox run on **synthetic data**: a learning experiment, not market
results. Your laptop run writes `RL_SIGNAL_LADDER.md`. φ is the planted lag-1 autocorrelation
of daily returns; out-of-sample Sharpe over 3 unseen years, PPO mean of 3 seeds:

| costs | φ | PPO | PPO worst seed | ideal rule (knows the signal) |
|---|---|---|---|---|
| 0 | 0.00 | 0.03 | -0.13 | 0.11 |
| 0 | 0.05 | 0.51 | 0.49 | 0.69 |
| 0 | 0.10 | 0.92 | 0.58 | 1.45 |
| 0 | 0.20 | 2.14 | 2.03 | 2.67 |
| 0.15% | 0.05 | -0.20 | -0.37 | -0.42 |
| 0.15% | 0.10 | -0.03 | -0.18 | 0.07 |
| 0.15% | 0.20 | 0.57 | 0.26 | 1.28 |

Compare these φ values with the real ones: RESULTS.md's data-quality table now shows the lag-1
autocorrelation of every asset, measured on development data only. If the real values are
small, the most anything can earn from daily timing on these features, after costs, is small.

**Decision:**
- *RL:* stop using it to search for strategies. No new RL variants this round.
- *Rules:* make them proper candidates and test them hard.
- *RL code:* stays as a learning tool. `inspect_agent.py` and the ladder show how it works and
  when it can work.

**The 2 round-3 candidates** (each new candidate costs every result a bigger luck haircut):

| config | what | why |
|---|---|---|
| `r3_rule_vol_target` | round 2's `vol_target`, parameters unchanged | best result so far; never formally robustness-tested or eligible for the final test |
| `r3_rule_portfolio_vt` | inverse-vol weights + whole-portfolio vol target (8%) using the covariance matrix | vol_target ignores that assets fall together (2020, stocks + bonds in 2022) |

**Rejected:**
- *More RL variants on daily bars:* the evidence above.
- *Trend-filtered rules:* round 2 showed the trend filter hurt.
- *Intraday data:* Yahoo keeps only about 60 days of 5-minute bars, far too short for
  walk-forward, and 0.15% costs swamp short-horizon edges.
- *Crypto or a new universe:* changing the data after seeing results is another way of
  fishing. It could be a separate project with its own locked test set.
- *RL for execution:* needs order-book data, a broker feed and a different problem.

**Honest accounting:**
- Every rule shown in rounds 1-2 now counts as a trial (`multiple_testing.extra_trials: 5`),
  so N = 27.
- Rules have no seed, so their robustness uses neighbouring parameters instead; every
  neighbour must pass.
- The pass bar is unchanged. PROPOSALS.md argues, with numbers, that it is too strict: a
  strategy with a true Sharpe of 1.0 would pass it about 2% of the time. Two switches in
  `base_config.yaml` are there for you to approve or not (P1, P2).

## 3d. Round 4: more return sources, honest measurement, a clean hold-out

### Where an edge can and cannot exist (what 3 rounds taught us)

**Not here:**
- **Daily timing of these assets from price features.** Real lag-1 autocorrelation is about 0
  (SPY -0.10, QQQ -0.09, others near 0). The signal ladder showed that even a perfect trader needs
  about 0.2 before 15 bps costs are covered. RL on daily bars is not the tool for this problem.
- **Picking the "best" rule on a 6-asset basket.** With 27-34 trials and 9-18 years of data, the
  luck haircut is as big as any plausible edge.
- **Intraday trading or single-stock picking:** no data or cost advantage for us.

**Possibly here (economic reasons + decades of published evidence):**
1. **More independent return sources.** Equities, bonds, credit, gold, commodities, real estate and
   the dollar earn different premia and crash at different times. Holding more of them raises
   return per unit of risk (Sharpe). Diversification is the one free lunch.
2. **Trend following across asset classes.** It steps out of assets in long declines (2008,
   2022), which is exactly where buy-and-hold dies. It works through diversification across many
   markets, not on 6.
3. **Combining rules whose returns are not correlated** (risk parity + trend + momentum).
4. **Then choosing the risk level deliberately.** See the next section.

### Profit = Sharpe x risk (the most important idea in this project)

- **Two dials.** Sharpe is how efficient the engine is; volatility is how hard you press the throttle.
- **vol_target only looked like "just a risk reducer".** It had a better engine than buy-and-hold
  (Sharpe 1.26 vs 1.10 over 18 years) running at half the throttle, so it made half the money.
- **Compare profit at equal risk.** That's the "CAGR at 10% vol" column.
- **The kill switch caps the throttle at about 10% volatility.** So realistic profit is roughly
  Sharpe x 10% a year (Sharpe 0.7 → ~7%, Sharpe 1.0 → ~10%).
- **The only honest way to more profit is a higher Sharpe**, then leverage up to the risk you accept,
  paying interest on the borrowed part. `H6` does exactly that, with leverage capped at 1.5x.

### The round-4 hypotheses (pre-registered in `research/plan_round4.yaml`, committed first)

| | rule | why |
|---|---|---|
| H0 | round 3's vol_target on the broad universe | control: did its edge depend on the 6-asset basket? |
| H1 | equal weight 1/N, monthly | diversification baseline, famously hard to beat |
| H2 | inverse-volatility (naive risk parity) | equal risk from each asset; low-risk assets earn more per unit of risk |
| H3 | trend: hold an asset only while its 12-month return beats T-bills | crisis protection, documented for a century |
| H4 | dual momentum: top third by 12-1 month return, if it beats T-bills | relative strength, a different return source from H3 |
| H5 | ensemble of H2 + H3 + H4 | diversification across RULES |
| H6 | H5 scaled to 10% predicted volatility, leverage <= 1.5x, financed at T-bill + 1% | turns Sharpe into profit at the kill switch's risk limit |

The rules are long only and rebalance monthly (first trading day; low turnover keeps costs small).
The exploration universe is 13 ETFs (US large/small, developed, emerging, long/intermediate
Treasuries, TIPS, IG and high-yield credit, gold, commodities, REITs, dollar), over 2007 → the day
before the locked test set. Benchmarks: SPY buy-and-hold and 60/40 (SPY/TLT, monthly).

**Rejected:**
- *More RL:* no specific, testable reason it would add value. Daily predictability is about 0, and
  rebalancing under proportional costs has a known simple solution (trade only outside a band).
  The RL code stays as a learning tool.
- *Carry and value:* need futures or fundamental data that Yahoo doesn't provide reliably.
- *Crypto:* too short a history to validate, and extreme regime changes.
- *Tuning lookbacks or targets:* the parameters are textbook values. Neighbouring values are run
  only as robustness checks.

### Validation plan (since 2007-2015 and 2016-2024 are both "seen")

1. **Exploration** (development data): all 7 hypotheses, every one counted as a trial (N = 27 + 7 = 34).
   Anything added later is marked EXPLORATORY in the report and also counted.
2. **Finalists** (at most 3, pre-registered rule): survival >= 80%, profitable at 2x costs,
   top 3 by Sharpe. Chosen from exploration only.
3. **Confirmation** (needs your approval, P6): finalists on 10 mutual funds over 1991-2006, a
   different set of instruments and years never used anywhere. Pre-registered pass criteria.
   Runs once; then the finalists are locked.
4. **Locked test set** (2024-10-16 → 2026-10-16): only when you decide, once.
5. **Forward paper trading** (`paper/`): finalists + 60/40 + buy-and-hold, logged every trading day
   from the first round-4 run onwards. It never logs anything inside the locked test period.
   Append-only, committed to git. **12+ months of this before any real money.**

## 3e. Round 5: is there a real edge a retail investor can get? (century-long data)

### Where an edge CANNOT exist for you (with free daily data and retail costs)
- **Daily or intraday timing from prices.** Measured autocorrelation is about 0, and the signal ladder showed costs swamp
  anything that small. Free intraday data on Yahoo goes back at most ~60 days (5-minute bars) or 2 years (hourly):
  too short to test anything.
- **Reinforcement learning on any of the above.** It optimises what is in the data; there is nothing there to find.
- **Option selling (the volatility risk premium).** The premium is real, but NSE lot sizes need lakhs of rupees in margin,
  crashes can wipe out years of premium, and SEBI's own studies found about 9 in 10 retail F&O traders lose money.
- **Currency, rates or commodity carry.** These need futures/forward data and market access a resident Indian
  doesn't have.
- **Post-earnings drift and other stock-level anomalies.** These need clean, survivorship-free stock and earnings data,
  which isn't free.

### Where a modest edge MIGHT exist (economic reason + decades of evidence)
- **Owning premia cheaply.** The equity premium, plus tilts toward momentum and value.
- **Trend following** as crash protection.
- **Diversification** across assets and premia.
- **Low costs and low taxes.** For an Indian investor these often matter more than any signal.

Round 5 tests the tilts and the timing on **Ken French's free data from 1926**, with real statistical power.

### The quant logic (learn these, they are the whole game)
1. **Noise vs. sample size.** A Sharpe ratio estimated from Y years has a standard error of about 1/sqrt(Y):
   18 years → ±0.24, 61 years → ±0.13. Two related strategies are compared more sharply, but you still need decades.
   Most "edges" people find are inside this noise.
2. **Excess returns.** Judge returns ABOVE T-bills; T-bill interest is not skill (this is why P5 looked better than it was).
3. **Equal risk.** Profit ≈ Sharpe × risk. Round 5 scales every strategy AND the market to 10% volatility, so
   "beats the market" really means "more profit for the same risk".
4. **Multiple testing.** 41 trials have been counted so far. The best of 41 random strategies looks good by luck;
   the haircut and the deflated-Sharpe test correct for that.
5. **Decay.** Published anomalies shrink: McLean-Pontiff (2016) found returns about 26% lower out of sample and about
   58% lower after publication. Round 5
   shows each tilt's edge by sub-period (1963-90, 1991-2006, 2007-now), so you can see it.
6. **Taxes and fixed charges.** In India, selling within 12 months costs 20% of the gain (STCG). Every sale costs
   about ₹18 in DP charges whatever its size: 0.65% a year on a ₹1 lakh account that sells monthly, but only 0.0065% on
   ₹1 crore. Low-turnover strategies keep far more.

### The round-5 hypotheses (pre-registered, `research/plan_round5.yaml`)

| | idea | why |
|---|---|---|
| H1 | market only while its 12-month return beats T-bills | trend: avoids long bear markets |
| H2 | large-cap momentum (past winners) | the most persistent anomaly, across markets and centuries |
| H3 | large-cap value (cheap stocks) | classic premium; honest test of its post-2007 decay |
| H4 | 50/50 value + momentum | they are negatively correlated → diversification |
| H5 | H4 + the H1 trend filter | tests whether the two effects add up |
| H6 | market only around the turn of the month | calendar effect with a mechanical cause (month-end money flows) |

All run at 10% volatility, without leverage, with 15 bps per trade plus fund/turnover drag and the same kill switch.

**Exploration:** US 1963-2024. **Confirmation (P10, needs your approval):** US 1927-1963 + Developed ex-US 1991-2024.
**After that:** the locked test, then forward paper trading using US ETFs (SPY, MTUM, IWD, SHV) as the tradeable
stand-ins.

### Rejected
- *More RL:* no testable reason it could add value here.
- *India stock-level factor backtests:* no clean, free, survivorship-free NSE data reachable from code.
  IIMA publishes Indian Fama-French and momentum factors (faculty.iima.ac.in/iffm) for manual study.
- *Crypto:* the history is too short and changes regime too often.
- *Intraday:* the free data is too short.

### Can you actually trade the result? (India reality, see `india_costs.py` and the report table)
- **On NSE:**
  - Nifty 50 ETFs or index funds are liquid and cheap.
  - Momentum and value index funds and ETFs exist: Nifty200 Momentum 30, Nifty200 Value 30, Nifty500 Value 50. They
    launched around 2020-23, so their history is short, and the indexes' earlier history is a back-calculated
    simulation. Their bid-ask spreads are wider.
  - Liquid funds can hold idle money.
  - Holding over 12 months is taxed at 12.5% above ₹1.25 lakh of gains a year; under 12 months at 20%.
- **US ETFs via LRS** (Vested/INDmoney-type platforms):
  - You pay roughly 0.5% FX each way.
  - TCS of 20% applies above ₹10 lakh a year (you get it back through your tax return).
  - US dividends are taxed at 25% (credit is claimable).
  - There's no ₹1.25 lakh exemption, and gains are long-term only after 24 months.
  - Paperwork: Schedule FA, Form 67.
- **What capital matters:** an edge of 1% a year is ₹1,000 a year on ₹1 lakh. For a student, saving more and
  keeping costs and taxes low beats any strategy edge until the account is much larger.

## 3b. How the RL part works (short version)

Read `trading_env.py` and `agents.py`. Their comments are written as a tutorial.

* **State** (what the agent sees each day): today's z-scored features, optionally
  the last `obs_window` days, plus the current position and the current drawdown.
* **Action**: choose one of `env.actions`. `[0, 1]` means flat or long;
  `[-1, 0, 1]` adds short. The chosen position is held until the next close.
* **Reward**: `log(1 + net return after costs)`, minus optional penalties for
  turnover, drawdown level, new drawdown, variance, and a one-time "death" penalty.
  This is all set in `base_config.yaml → reward`.
* **DQN** learns **Q(s, a)**, the expected future reward of taking action *a* in
  state *s*, and acts greedily on it.
* **PPO** learns a **policy π(a|s)** (action probabilities) and a value **V(s)**,
  and pushes probability toward actions that did better than expected.
* `python inspect_agent.py --exp dqn_base --fold 3 --seed 0 --ticker SPY` prints,
  day by day, the Q-values or π and V(s), the action taken, and the reward.

---

## 4. Changing experiments (no code changes)

* `base_config.yaml` holds all defaults: data, features, costs, survival, walk-forward,
  env, reward, PPO/DQN hyperparameters, baselines, robustness and pass criteria.
* `configs/*.yaml` is the **experiment queue**. Each file is one variant and only lists
  what differs from the base. Copy `configs/_TEMPLATE.yaml.example` to start a new
  one. Files starting with `_` are ignored.
* At most `experiment_cap: 30` variants run per round. Every variant counts toward
  the multiple-testing penalty forever, so add variants for a reason, not to fish.
* When you change a config, its hash changes and its runs are redone automatically.
  Changing `runtime.n_jobs` does not change the hash.

Useful flags:

```powershell
python run_all.py --only ppo_base dqn_base   # subset
python run_all.py --n-jobs 3                 # fewer worker processes (default: cores - 1)
python run_all.py --skip-robustness
python robustness.py --exp ppo_base          # robustness for one experiment
python run_all.py --synthetic                # fake data, code testing only
```

## 5. Output files

| path | content |
|---|---|
| `RESULTS.md` | full report: every experiment, failures, survival counts, robustness, plots |
| `results/real/experiments/<name>/runs/` | one JSON and CSV per (fold, seed) run |
| `results/real/plots/` | stitched out-of-sample equity curves |
| `results/real/selection.json` | the strategy chosen for `final_test.py` (or none) |
| `results/real/trial_ledger.json` | every variant ever tried (multiple-testing count) |
| `models/real/...` | trained agents (`.zip`) |
| `logs/` | full logs of every run |
| `FINAL_TEST.md` | one-time test result |

## 6. Honesty rules built in

* The test set is evaluated once, by `final_test.py`, and only for a strategy that
  passed every check. A lock file prevents a second run.
* `final_test.py` refuses to run if the selected config changed after validation.
* Results that look too good (Sharpe > 2.5) are failed automatically and flagged as
  a likely bug or leak.
* Every experiment appears in `RESULTS.md`, including failed and crashed ones.
* Seeds are fixed. Results should reproduce on the same machine and package
  versions, although tiny numeric differences across machines are normal.
* Known simplifications: daily close-to-close fills, proportional costs only, no
  short borrow fees, no taxes, and no FX (INR stocks are treated in local returns).

## 7. If something goes wrong

* `ModuleNotFoundError`: activate the venv (`.\.venv\Scripts\Activate.ps1`) and run
  `pip install -r requirements.txt` again.
* A download fails or Yahoo rate-limits you: wait a few minutes and run `python data.py`
  again. Already cached tickers are skipped.
* `SELF-CHECK FAILED`: do not trust any results. The message names the check that failed.
* The run is too slow: lower `agent.total_timesteps` or `agent.seeds`, or remove
  variants from `configs/`. Each of those changes the experiments, which is fine
  before you look at results.
* New research round: bump `round:` in `base_config.yaml` and edit `configs/`. Old
  variants stay in `results/real/trial_ledger.json` and keep counting toward the
  penalty, which is intended. Never delete `data/raw/split_lock_yahoo.json` or the
  ledger. Once `final_test.py` has run, the test set is used up, and later rounds
  should be judged on new data, such as forward paper trading.
