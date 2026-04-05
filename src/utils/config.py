from pathlib import Path

import yaml


def load_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
