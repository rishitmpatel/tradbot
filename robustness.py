"""Robustness tests for a candidate strategy (all on DEVELOPMENT data only).

  costs_2x       same trained models, re-evaluated with 2x commission + slippage
  other_seeds    retrain every fold with fresh random seeds
  other_assets   same trained models traded on assets they never saw
  shifted_dates  retrain with every fold boundary shifted (~6 months)

Normally called by run_all.py for the top candidates. Standalone:
    python robustness.py --exp ppo_base            (real data)
    python robustness.py --exp ppo_base --synthetic
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config_utils import CONFIGS_DIR, load_experiment, resolve_n_jobs, set_dotted
from evaluation import robust_check
from reporting import save_json
from walkforward import load_dev_assets, make_folds, run_experiment, summarize

log = logging.getLogger(__name__)

ALL_TESTS = ["costs_2x", "other_seeds", "other_assets", "shifted_dates"]


def run_robustness(cfg: dict, exp_dir: Path, model_dir: Path, n_jobs: int, tests: list[str]) -> dict:
    rc, pc = cfg["robustness"], cfg["pass_criteria"]
    assets, split = load_dev_assets(cfg)
    folds = make_folds(cfg, assets, split)
    out = {}
    for test in tests:
        rdir = exp_dir / "robustness" / test / "runs"
        try:
            if test == "costs_2x":
                runs = run_experiment(cfg, rdir, model_dir, folds, n_jobs, cost_mult=float(rc["cost_multiplier"]),
                                      eval_only=True, label=f"{cfg['name']}:{test}")
            elif test == "other_seeds":
                c2 = set_dotted(cfg, "agent.seeds", list(rc["other_seeds"]))
                runs = run_experiment(c2, rdir, model_dir, folds, n_jobs, label=f"{cfg['name']}:{test}")
            elif test == "other_assets":
                runs = run_experiment(cfg, rdir, model_dir, folds, n_jobs, eval_only=True,
                                      eval_tickers=list(rc["other_assets"]), label=f"{cfg['name']}:{test}")
            elif test == "shifted_dates":
                c3 = set_dotted(cfg, "walk_forward.date_shift_days", int(rc["shift_days"]))
                c3 = set_dotted(c3, "agent.seeds", list(rc["shift_seeds"]))
                f3 = make_folds(c3, assets, split)
                runs = run_experiment(c3, rdir, model_dir, f3, n_jobs, label=f"{cfg['name']}:{test}")
            else:
                raise ValueError(f"unknown robustness test {test}")
            s = summarize(f"{cfg['name']}:{test}", runs)
            ok, why = robust_check(s, pc)
            out[test] = {"status": "ok", "summary": s, "pass": ok, "reasons": why}
        except Exception as e:  # e.g. other-asset download failed
            log.exception("robustness %s failed", test)
            out[test] = {"status": f"error: {type(e).__name__}: {e}", "summary": None, "pass": False}
        log.info("robustness %s / %s -> %s", cfg["name"], test, out[test].get("pass"))
    save_json(exp_dir / "robustness.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name (configs/<name>.yaml)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tests", nargs="*", default=ALL_TESTS)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from run_all import mode_dirs, prepare_cfg

    cfg = prepare_cfg(load_experiment(CONFIGS_DIR / f"{args.exp}.yaml"), args.synthetic, smoke=False)
    res_root, model_root = mode_dirs(cfg, args.synthetic, smoke=False)
    n_jobs = args.n_jobs or resolve_n_jobs(cfg["runtime"].get("n_jobs"))
    res = run_robustness(cfg, res_root / "experiments" / cfg["name"], model_root / cfg["name"], n_jobs, args.tests)
    for t, r in res.items():
        s = r.get("summary") or {}
        print(f"{t:14s} {r['status']:8s} pass={r.get('pass')} return={s.get('mean_return')} "
              f"sharpe={s.get('mean_sharpe')} survival={s.get('survival_rate')}")


if __name__ == "__main__":
    main()
