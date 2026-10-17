# Instructions for an automated agent (e.g. Claude Code) running this repo

Your job: **run, report, commit**. You are an operator, not a researcher.

## Hard rules
1. **Do not edit** any `.py` file, `base_config.yaml`, anything in `configs/`, or
   `research/plan_round4.yaml`. Do not add or remove configs or hypotheses. Do not change
   seeds, costs, dates or thresholds. In particular, only the human owner may change
   `pass_criteria.sharpe_test`, `pass_criteria.require_long_history` and
   `research.confirmation_approved` (see PROPOSALS.md).
2. **Do not delete or modify** `data/raw/split_lock_*.json`, `results/*/trial_ledger.json`,
   `results/*/final_test/LOCK.json`, anything in `results/*/research/` (registration,
   finalists, confirmation lock), or anything in `paper/`. The paper tracker only ever
   APPENDS to `paper/`; you never edit those files.
3. Run `final_test.py` **at most once, ever**, and only if `RESULTS.md` says a strategy
   PASSED and the owner has asked you to. Never run it to "see how things look".
4. **Never invent, round up or reinterpret numbers.** Copy them from the reports. If
   something failed, say so plainly.
5. Paper trading only. Do not connect any broker or trading API.
6. If a command errors, **report the error and the last ~40 log lines** (`logs/`).
   Do not try to fix code. You may retry once if the cause is clearly transient (network,
   Yahoo rate limit, sleep/reboot). Re-running `run_all.py` resumes where it stopped.

## Commands (Windows PowerShell, from the repo folder)
```powershell
git pull                              # if it complains about base_config.yaml: report it, do not resolve it yourself
.\.venv\Scripts\Activate.ps1
python -m pytest -q                   # must show all tests passed; if not, stop and report
python run_all.py                     # EVERYTHING: data, self-checks, baselines, round-4 research,
                                      # (approved) confirmation, forward paper tracker. ~5-15 min.
```
Between full runs, the owner may ask you to update only the paper tracker:
`python paper_trade.py`, then commit as below.

## Commit after every run
```powershell
git add -f RESULTS.md RESEARCH.md paper logs results/real/plots results/real/trial_ledger.json results/real/research results/real/selection.json
git commit -m "Run results <date>: <one-line verdict>"
git push
```
Never commit `data/` or `models/`.

## What to report (from RESULTS.md)
* the data source line and the test-set dates
* the **Verdict** section, verbatim. "NO STRATEGY PASSED" is a valid and common result.
* any **WARNING / SUSPICIOUS / CRASHED / FINALISTS CHANGED** lines, verbatim
* **Round 4 table**, every row: Sharpe, CAGR, max DD, survived, Sharpe - 60/40, p,
  CAGR at 10% vol, penalized Sharpe, finalist, verdict, and the P4 column. State that
  P4 is NOT active.
* the "Why each hypothesis passed or failed" list, verbatim
* how many hypotheses are marked EXPLORATORY, and the total trial count
* the correlation table and the finalists line
* the Confirmation section: the table, or the reason it did not run
* the forward paper-trading table, plus the note that under 12 months it says almost nothing
* the trial-ledger table
* tickers or funds that failed to load, and any data-quality jumps
* total runtime (last lines of the newest `logs/run_all_real_*.log`)
* whether `final_test.py` was run and its result, or why it was not run

State clearly that these are **paper-trading research results**, and that validation
numbers are optimistic even after the multiple-testing penalty.
