from copy import deepcopy
from pathlib import Path

import yaml


def load_config(config_path):
    config_path = Path(config_path).resolve()
    config = _load_with_defaults(config_path)
    return config


def _load_with_defaults(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    defaults = raw_config.pop("defaults", [])
    merged = {}
    for default_entry in defaults:
        default_path = (config_path.parent / default_entry).resolve()
        default_config = _load_with_defaults(default_path)
        merged = deep_merge(merged, default_config)
    return deep_merge(merged, raw_config)


def deep_merge(base, override):
    base = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base
