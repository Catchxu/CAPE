import numpy as np

from ..data.label_utils import build_label_encoder, encode_labels
from ..data.preprocess import load_and_prepare_adata, split_adata
from ..models import get_backend
from ..utils.io import (
    ensure_run_directories,
    save_json,
    save_predictions,
    save_yaml,
)
from ..utils.logger import setup_logger
from ..utils.seed import seed_everything


def run_cell_type_annotation(config):
    model_name = config["model"]["name"]
    run_name = config["run"]["run_name"]
    log_dir, result_dir = ensure_run_directories(config)
    logger = setup_logger(
        name=f"CTA.{model_name}.{run_name}",
        log_file=log_dir / f"{run_name}.log",
    )

    logger.info("Starting CTA pipeline for model=%s run=%s", model_name, run_name)
    seed_everything(config["run"]["seed"])

    adata = load_and_prepare_adata(config["data"], logger)
    logger.info("Loaded AnnData with %d cells and %d genes", adata.n_obs, adata.n_vars)

    split_data = split_adata(adata, config["data"]["split"], config["data"]["label_column"])
    train_adata = split_data["train"]
    val_adata = split_data["val"]
    test_adata = split_data["test"]

    label_column = config["data"]["label_column"]
    label_encoder = build_label_encoder(train_adata.obs[label_column].tolist())
    _ = encode_labels(train_adata.obs[label_column].tolist(), label_encoder)
    if val_adata is not None:
        _ = encode_labels(val_adata.obs[label_column].tolist(), label_encoder)
    if test_adata is not None:
        _ = encode_labels(test_adata.obs[label_column].tolist(), label_encoder)

    backend = get_backend(model_name)
    backend_result = backend.run_cta(
        config=config,
        train_adata=train_adata,
        val_adata=val_adata,
        test_adata=test_adata,
        label_encoder=label_encoder,
        logger=logger,
    )

    save_yaml(result_dir / "config_resolved.yaml", config)
    save_json(result_dir / "metrics.json", backend_result["metrics"])
    save_json(result_dir / "label_mapping.json", label_encoder.to_dict())

    summary = {
        "task": config["task"]["name"],
        "model": model_name,
        "run_name": run_name,
        "n_train": int(train_adata.n_obs),
        "n_val": int(val_adata.n_obs) if val_adata is not None else 0,
        "n_test": int(test_adata.n_obs),
        "metrics": backend_result["metrics"],
        "artifacts": {
            "log_file": str(log_dir / f"{run_name}.log"),
            "predictions": str(result_dir / "predictions.csv"),
            "metrics": str(result_dir / "metrics.json"),
        },
    }
    save_json(result_dir / "summary.json", summary)

    save_predictions(
        output_path=result_dir / "predictions.csv",
        cell_ids=test_adata.obs_names.tolist(),
        predicted_labels=backend_result["predicted_labels"],
        predicted_ids=backend_result["predicted_ids"],
        true_labels=backend_result.get("true_labels"),
        true_ids=backend_result.get("true_ids"),
    )

    if backend_result.get("probabilities") is not None and config["task"].get("save_probabilities", False):
        probability_path = result_dir / "probabilities.npy"
        np.save(probability_path, backend_result["probabilities"])

    logger.info("CTA pipeline completed successfully")
    logger.info("Results saved to %s", result_dir)
    return backend_result
