# Instructions for an automated agent (e.g. Claude Code) running this repo

Your job is to **run the pipeline and report results**. You are an operator, not a
researcher. You must follow these rules.

## Hard rules
1. **Do not edit** any `.py` file, `base_config.yaml`, or anything in `configs/`.
   Do not add or remove configs. Do not change seeds, costs, dates or thresholds.
   This includes `pass_criteria.sharpe_test` and `pass_criteria.require_long_history`:
   only the human owner may switch those (see PROPOSALS.md).
2. **Do not delete or modify** `data/raw/split_lock_*.json`,
   `results/*/trial_ledger.json`, or `results/*/final_test/LOCK.json`.
3. Run `final_test.py` **at most once, ever**, and only if `RESULTS.md` says a strategy
   PASSED. Never run it to "see how things look".
4. **Never invent, round up or reinterpret numbers.** Copy them from `RESULTS.md` /
   `FINAL_TEST.md`. If something failed, say so plainly.
5. Paper trading only. Do not connect any broker or trading API.
6. If a command errors, **report the error and the last ~40 log lines** (`logs/`).
   Do not try to fix code. You may retry a command once if the cause is clearly
   transient (network, Yahoo rate limit, sleep/reboot). Re-running `run_all.py`
   resumes where it stopped.

## Commands (Windows PowerShell, from the repo folder)
```powershell
.\.venv\Scripts\Activate.ps1          # if the venv does not exist, follow README section 1
python -m pytest -q                   # must show all tests passed; if not, stop and report
python data.py --extra                # download/cache data (re-run once if it fails)
python run_all.py --smoke             # ~5-10 min sanity run; if it fails, stop and report
python run_all.py                     # full run; re-run the same command if interrupted
python long_history.py                # rules over 2007-2024 (downloads data/raw_long once); informational
python rl_signal_ladder.py            # ~10-20 min synthetic learning experiment; informational
# only if RESULTS.md shows "Strategies passing ALL checks":
python final_test.py
```

## What to report
Report from `RESULTS.md`, and from `FINAL_TEST.md` if it exists:
* the data source line and the test-set dates
* the **Verdict** section, verbatim. "NO STRATEGY PASSED" is a valid and common result.
* any **WARNING / SUSPICIOUS** lines, verbatim
* the RL experiments table: name, survived (x/y), OOS return, Sharpe, penalized Sharpe,
  Sharpe - B&H, DD / B&H DD, rule-only Sharpe, low-risk match, verdict
* any "Low-risk match" line, verbatim (it is informational, NOT a pass)
* the "Active Sharpe test" line and the "would pass PROPOSED bar P1+P2 (NOT active)" column
  for every row. State clearly that the proposed bar is NOT active.
* the robustness rows for each rule, including `param_neighborhood` and `long_history_2007_2015`
* the lag-1 autocorrelation column of the data-quality table
* from `LONG_HISTORY.md`: the three period tables, and any DEAD windows in the per-window table
* from `RL_SIGNAL_LADDER.md`: the table, verbatim. It is a SYNTHETIC experiment.
* the baselines table, briefly
* the robustness table for the candidates
* experiments with errors, plus their error messages
* total runtime (last lines of the newest `logs/run_all_real_*.log`)
* whether `final_test.py` was run and its result, or why it was not run

State clearly that these are **paper-trading research results**, and that validation
numbers are optimistic even after the multiple-testing penalty.
