"""Config loading.

Every run is driven by YAML:
  base_config.yaml      -> defaults for everything
  configs/<name>.yaml   -> one experiment = a small override deep-merged on top

You change reward weights, features, actions and hyperparameters by editing or
adding YAML files. No Python changes needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "base_config.yaml"
CONFIGS_DIR = ROOT / "configs"

# Keys that do not change the *strategy* (only how fast / where it runs).
# They are excluded from the config hash, so changing n_jobs does not
# invalidate finished runs.
NON_STRATEGY_KEYS = {"runtime", "name", "description", "robustness", "pass_criteria", "multiple_testing",
                     "final_test_criteria", "research", "research5",
                     "experiment_cap", "round"}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Return base with override applied recursively (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_base() -> dict:
    return load_yaml(BASE_CONFIG)


def load_experiment(path: str | Path, base: dict | None = None) -> dict:
    """Base config + experiment override. Experiment name = file stem."""
    base = base if base is not None else load_base()
    override = load_yaml(path)
    cfg = deep_merge(base, override)
    cfg["name"] = Path(path).stem
    cfg.setdefault("description", override.get("description", ""))
    validate(cfg)
    return cfg


def list_experiment_files(configs_dir: str | Path = CONFIGS_DIR) -> list[Path]:
    """All configs/*.yaml except files starting with '_' (disabled)."""
    files = sorted(Path(configs_dir).glob("*.yaml"))
    return [f for f in files if not f.name.startswith("_")]


def strategy_hash(cfg: dict) -> str:
    """Short hash of everything that defines the strategy + evaluation setup."""
    core = {k: v for k, v in cfg.items() if k not in NON_STRATEGY_KEYS}
    blob = json.dumps(core, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def set_dotted(cfg: dict, dotted: str, value) -> dict:
    """set_dotted(cfg, 'costs.commission', 0.002) -> modified copy."""
    out = copy.deepcopy(cfg)
    node = out
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return out


def resolve_n_jobs(value) -> int:
    """'auto' / None / 0 -> CPU cores - 1 (at least 1)."""
    import os
    if value in (None, "auto", 0, "0"):
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(value))


def validate(cfg: dict) -> None:
    if cfg.get("kind", "rl") == "rule":
        from baselines import BASELINES
        from rules import PORTFOLIO_RULES
        rule = cfg.get("rule") or {}
        if rule.get("name") not in {**BASELINES, **PORTFOLIO_RULES}:
            raise ValueError(f"rule.name must be one of {sorted({**BASELINES, **PORTFOLIO_RULES})}")
        params = set((rule.get("params") or {}).keys())
        for ov in rule.get("neighborhood") or []:
            if not set(ov) <= params:
                raise ValueError(f"neighborhood entry {ov} changes a parameter not in rule.params")
    elif cfg.get("kind", "rl") != "rl":
        raise ValueError("kind must be rl or rule")
    algo = str(cfg["agent"]["algo"]).upper()
    if algo not in ("PPO", "DQN"):
        raise ValueError(f"agent.algo must be PPO or DQN, got {algo}")
    acts = cfg["env"]["actions"]
    if not acts or any(abs(float(a)) > 1.0 for a in acts):
        raise ValueError("env.actions must be a non-empty list of positions in [-1, 1]")
    if not cfg["features"]["list"]:
        raise ValueError("features.list is empty")
    if cfg["survival"]["max_drawdown"] <= 0:
        raise ValueError("survival.max_drawdown must be > 0")
    if not cfg["agent"]["seeds"]:
        raise ValueError("agent.seeds is empty")
