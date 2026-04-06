from pathlib import Path

import yaml


def load_config(config_path):
    return _load_config_recursive(Path(config_path).resolve())


def _load_config_recursive(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    includes = config.pop("includes", [])
    merged = {}
    for relative_path in includes:
        default_path = (config_path.parent / relative_path).resolve()
        default_config = _load_config_recursive(default_path)
        merged = _deep_merge(merged, default_config)
    return _deep_merge(merged, config)


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override

    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
