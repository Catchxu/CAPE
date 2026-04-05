import json

import anndata as ad
import numpy as np
import pandas as pd

from src.tasks.cell_type_annotation import run_cell_type_annotation


class FakeBackend:
    def run_cta(self, config, train_adata, val_adata, test_adata, label_encoder, logger):
        true_ids = label_encoder.transform(test_adata.obs[config["data"]["label_column"]].tolist())
        return {
            "metrics": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
            },
            "predicted_ids": true_ids,
            "predicted_labels": label_encoder.inverse_transform(true_ids),
            "true_ids": true_ids,
            "true_labels": label_encoder.inverse_transform(true_ids),
            "probabilities": np.eye(len(label_encoder.classes_))[true_ids],
        }


def test_pipeline_writes_standard_outputs(tmp_path, monkeypatch):
    adata = ad.AnnData(np.array([[1.0, 0.0], [0.5, 1.5], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    adata.obs["celltype"] = pd.Categorical(["A", "A", "B", "B"])
    adata.var_names = ["g1", "g2"]
    dataset_path = tmp_path / "toy.h5ad"
    adata.write_h5ad(dataset_path)

    config = {
        "task": {"name": "CTA", "save_probabilities": True},
        "model": {"name": "scbert"},
        "data": {
            "path": str(dataset_path),
            "label_column": "celltype",
            "batch_column": None,
            "gene_column": None,
            "input_layer": None,
            "preprocess": {"normalize_total": None, "log1p": False},
            "split": {
                "mode": "stratified",
                "random_state": 0,
                "ratios": {"train": 0.5, "val": 0.0, "test": 0.5},
            },
        },
        "train": {"batch_size": 2, "learning_rate": 1e-4, "weight_decay": 0.0, "epochs": 1},
        "run": {
            "seed": 0,
            "device": "cpu",
            "run_name": "smoke",
            "logs_root": str(tmp_path / "logs"),
            "results_root": str(tmp_path / "results"),
        },
    }

    monkeypatch.setattr("src.tasks.cell_type_annotation.get_backend", lambda _: FakeBackend())
    run_cell_type_annotation(config)

    result_dir = tmp_path / "results" / "CTA" / "scbert" / "smoke"
    assert (result_dir / "predictions.csv").exists()
    assert (result_dir / "metrics.json").exists()
    assert (result_dir / "summary.json").exists()
    assert (result_dir / "config_resolved.yaml").exists()
    assert (result_dir / "probabilities.npy").exists()

    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"accuracy", "macro_f1", "weighted_f1"} <= set(metrics.keys())
