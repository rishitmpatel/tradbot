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

   You should see `18 passed`.

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

8. **Overnight run** (expect about 2–5 h on a 4-core laptop; it scales with CPU cores):

```powershell
python run_all.py
```

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
