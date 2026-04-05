import anndata as ad
import numpy as np

from src.data.label_utils import build_label_encoder, encode_labels
from src.data.preprocess import split_adata
from src.CTA.metrics import compute_cta_metrics


def _build_adata():
    matrix = np.vstack(
        [
            np.tile([1.0, 0.0, 3.0], (15, 1)),
            np.tile([0.0, 2.0, 1.0], (15, 1)),
        ]
    )
    adata = ad.AnnData(matrix)
    adata.obs["celltype"] = ["A"] * 15 + ["B"] * 15
    adata.obs["split"] = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    adata.var_names = ["g1", "g2", "g3"]
    return adata


def test_label_encoder_round_trip():
    encoder = build_label_encoder(["T", "B", "T"])
    encoded = encode_labels(["B", "T"], encoder)
    decoded = encoder.inverse_transform(encoded)
    assert decoded == ["B", "T"]


def test_stratified_split_is_deterministic():
    adata = _build_adata()
    split_cfg = {
        "mode": "stratified",
        "random_state": 7,
        "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
    }
    split_one = split_adata(adata, split_cfg, "celltype")
    split_two = split_adata(adata, split_cfg, "celltype")

    assert split_one["train"].obs_names.tolist() == split_two["train"].obs_names.tolist()
    assert split_one["test"].obs_names.tolist() == split_two["test"].obs_names.tolist()


def test_column_split_respects_labels():
    adata = _build_adata()
    split_cfg = {
        "mode": "column",
        "column": "split",
        "labels": {"train": "train", "val": "val", "test": "test"},
    }
    split_data = split_adata(adata, split_cfg, "celltype")
    assert split_data["train"].n_obs == 20
    assert split_data["val"].n_obs == 5
    assert split_data["test"].n_obs == 5


def test_metrics_include_required_keys():
    metrics = compute_cta_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert {"accuracy", "macro_f1", "weighted_f1"} <= set(metrics.keys())
