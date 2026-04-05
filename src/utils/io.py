import json
from pathlib import Path

import pandas as pd
import yaml


def ensure_run_directories(config):
    model_name = config["model"]["name"]
    run_name = config["run"]["run_name"]
    task_name = config["task"]["name"]

    logs_root = Path(config["run"]["logs_root"])
    results_root = Path(config["run"]["results_root"])

    log_dir = logs_root / task_name / model_name
    result_dir = results_root / task_name / model_name / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, result_dir


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_yaml(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_predictions(
    output_path: Path,
    cell_ids,
    predicted_labels,
    predicted_ids,
    true_labels=None,
    true_ids=None,
):
    frame = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "predicted_id": predicted_ids,
            "predicted_label": predicted_labels,
        }
    )
    if true_labels is not None:
        frame["true_id"] = true_ids
        frame["true_label"] = true_labels
    frame.to_csv(output_path, index=False)
