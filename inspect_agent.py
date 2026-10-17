"""Look inside a trained agent - a learning tool, not part of the pipeline.

For each day of one validation window it prints the STATE summary, what the
network thinks (DQN: Q-value per action, PPO: probability per action and the
state value V(s)), the ACTION taken and the REWARD-relevant outcome.

    python inspect_agent.py --exp ppo_base --fold 3 --seed 0 --ticker SPY --days 15
    python inspect_agent.py --exp dqn_base --synthetic --smoke
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from agents import build_env, explain_state, load_agent, make_segment
from backtest import cost_rate
from config_utils import CONFIGS_DIR, load_experiment, strategy_hash
from run_all import mode_dirs, prepare_cfg
from walkforward import load_dev_assets, make_folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = prepare_cfg(load_experiment(CONFIGS_DIR / f"{args.exp}.yaml"), args.synthetic, args.smoke)
    _, model_root = mode_dirs(cfg, args.synthetic, args.smoke)
    path = model_root / cfg["name"] / strategy_hash(cfg) / f"fold{args.fold:02d}_seed{args.seed}.zip"
    if not path.exists():
        raise SystemExit(f"No model at {path}. Run run_all.py first (same flags).")
    model = load_agent(cfg, path)
    assets, split = load_dev_assets(cfg)
    fold = make_folds(cfg, assets, split)[args.fold]
    asset = assets[args.ticker or next(iter(assets))]
    seg = make_segment(asset, pd.Timestamp(fold.val_start), pd.Timestamp(fold.val_end), cfg)
    env = build_env([seg], cfg, cost_rate(cfg), train=False)
    obs, _ = env.reset(options={"segment": 0})

    print(f"{cfg['name']} ({cfg['agent']['algo']}) fold {fold.k} seed {args.seed} on {asset.ticker}, "
          f"validation {fold.val_start} -> {fold.val_end}")
    print("STATE = features of today (z-scored) + [base position (overlay only), position, drawdown*10]. "
          f"Features: {cfg['features']['list']}\n")
    for _ in range(args.days):
        date = seg.dates[env.t].date()
        think = explain_state(model, obs, cfg)
        action, _ = model.predict(obs, deterministic=True)
        obs_next, reward, term, trunc, info = env.step(int(action))
        brain = think.get("Q_values") or think.get("action_probs")
        extra = f" V(s)={think['state_value_V']:+.3f}" if "state_value_V" in think else ""
        label = "Q" if "Q_values" in think else "pi"
        base_txt = f"base={seg.base[env.t]:+.2f} " if env.inc_base else ""
        print(f"{date} state[{base_txt}pos={env.pos:+.2f}] "
              f"{label}={{{', '.join(f'{k}: {v:+.3f}' for k, v in brain.items())}}}{extra} "
              f"-> hold {info['positions'][0]:+.2f}  net {sum(info['nets']):+.3%} over {len(info['nets'])}d  reward {reward:+.3f}")
        obs = obs_next
        if term or trunc:
            break
    print("\nReading it: DQN picks the action with the highest Q. PPO samples from pi during training "
          "and takes the most likely action during evaluation. Reward = scaled log-return after costs "
          "minus penalties (see base_config.yaml: reward).")
    _ = np  # keep numpy import for interactive use


if __name__ == "__main__":
    main()
