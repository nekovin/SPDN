"""
path management utilities for spdn.

loads paths from config file and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """get the project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent.parent


def load_paths_config(config_path: Optional[str] = None) -> dict:
    """load paths configuration from yaml file.

    environment variables in the format ${VAR:-default} are expanded.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "paths.yaml"

    if not Path(config_path).exists():
        return get_default_paths()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return _expand_env_vars(config)


def _expand_env_vars(config: dict) -> dict:
    """recursively expand environment variables in config values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, str):
            result[key] = _expand_single_var(value)
        else:
            result[key] = value
    return result


def _expand_single_var(value: str) -> str:
    """expand a single environment variable with format ${VAR:-default}."""
    if not value.startswith("${"):
        return value

    value = value[2:-1]
    if ":-" in value:
        var_name, default = value.split(":-", 1)
        return os.environ.get(var_name, default)
    else:
        return os.environ.get(value, "")


def get_default_paths() -> dict:
    """return default paths configuration."""
    project_root = get_project_root()
    return {
        "datasets": {
            "sdoct": os.environ.get("SDOCT_PATH", str(project_root / "data" / "sdoct")),
            "chiu_mat": os.environ.get("CHIU_MAT_PATH", str(project_root / "data" / "chiu")),
            "stage1_outputs": os.environ.get("STAGE1_PATH", str(project_root / "data" / "stage1")),
        },
        "checkpoints": {
            "base_dir": os.environ.get("CHECKPOINT_DIR", str(project_root / "checkpoints")),
            "spdn": os.environ.get("SPDN_CHECKPOINT", str(project_root / "checkpoints" / "spdn")),
            "baselines": os.environ.get("BASELINES_CHECKPOINT", str(project_root / "checkpoints" / "baselines")),
            "ssm": os.environ.get("SSM_CHECKPOINT", str(project_root / "checkpoints" / "spdn" / "spdn_SSMAttention_mse_best.pth")),
        },
        "configs": {
            "eval": os.environ.get("EVAL_CONFIG", str(project_root / "configs" / "eval.yaml")),
            "n2": os.environ.get("N2_CONFIG", str(project_root / "configs" / "n2_config.yaml")),
            "pfn": os.environ.get("PFN_CONFIG", str(project_root / "configs" / "pfn_config.yaml")),
        },
        "external": {
            "ssn2v_path": os.environ.get("SSN2V_PATH", str(project_root / "external" / "ssn2v")),
        },
    }


# convenience functions for common paths
_paths_cache = None


def get_paths() -> dict:
    """get paths config (cached)."""
    global _paths_cache
    if _paths_cache is None:
        _paths_cache = load_paths_config()
    return _paths_cache


def get_sdoct_path() -> str:
    """get sdoct dataset path."""
    return get_paths()["datasets"]["sdoct"]


def get_checkpoint_dir() -> str:
    """get checkpoint base directory."""
    return get_paths()["checkpoints"]["base_dir"]


def get_ssm_checkpoint() -> str:
    """get ssm model checkpoint path."""
    return get_paths()["checkpoints"]["ssm"]
