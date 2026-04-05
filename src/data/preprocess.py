from typing import Dict, Optional

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split


def load_and_prepare_adata(data_config: Dict, logger):
    path = data_config["path"]
    adata = ad.read_h5ad(path)

    label_column = data_config["label_column"]
    if label_column not in adata.obs:
        raise ValueError(f"Missing label column '{label_column}' in AnnData.obs")

    batch_column = data_config.get("batch_column")
    if batch_column and batch_column not in adata.obs:
        raise ValueError(f"Missing batch column '{batch_column}' in AnnData.obs")

    input_layer = data_config.get("input_layer")
    if input_layer:
        if input_layer not in adata.layers:
            raise ValueError(f"Missing input layer '{input_layer}' in AnnData.layers")
        adata = adata.copy()
        adata.X = adata.layers[input_layer].copy()
    else:
        adata = adata.copy()

    preprocess_cfg = data_config.get("preprocess", {})
    min_cell_counts = preprocess_cfg.get("filter_cells_min_counts")
    min_gene_counts = preprocess_cfg.get("filter_genes_min_counts")
    normalize_total = preprocess_cfg.get("normalize_total")
    log1p = preprocess_cfg.get("log1p", False)

    if min_gene_counts:
        logger.info("Filtering genes with min_counts=%s", min_gene_counts)
        adata = _filter_genes_by_counts(adata, int(min_gene_counts))
    if min_cell_counts:
        logger.info("Filtering cells with min_counts=%s", min_cell_counts)
        adata = _filter_cells_by_counts(adata, int(min_cell_counts))
    if normalize_total:
        logger.info("Normalizing total counts to %s", normalize_total)
        adata.X = _normalize_total(adata.X, float(normalize_total))
    if log1p:
        logger.info("Applying log1p transform")
        adata.X = _log1p(adata.X)

    return adata


def split_adata(adata, split_config: Dict, label_column: str):
    mode = split_config["mode"]
    if mode == "stratified":
        return _split_adata_stratified(adata, split_config, label_column)
    if mode == "column":
        return _split_adata_by_column(adata, split_config)
    raise ValueError(f"Unsupported split mode: {mode}")


def _split_adata_stratified(adata, split_config: Dict, label_column: str):
    ratios = split_config["ratios"]
    train_ratio = float(ratios["train"])
    val_ratio = float(ratios.get("val", 0.0))
    test_ratio = float(ratios["test"])
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    labels = adata.obs[label_column].astype(str).to_numpy()
    indices = np.arange(adata.n_obs)
    random_state = int(split_config.get("random_state", 0))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels,
    )

    if val_ratio > 0:
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=adjusted_val_ratio,
            random_state=random_state,
            stratify=labels[train_val_idx],
        )
        val_adata = adata[val_idx].copy()
    else:
        train_idx = train_val_idx
        val_adata = None

    return {
        "train": adata[train_idx].copy(),
        "val": val_adata,
        "test": adata[test_idx].copy(),
    }


def _split_adata_by_column(adata, split_config: Dict):
    column = split_config.get("column")
    labels_cfg = split_config.get("labels", {})
    if not column:
        raise ValueError("Column split mode requires split.column")
    if column not in adata.obs:
        raise ValueError(f"Split column '{column}' not found in AnnData.obs")

    split_values = adata.obs[column].astype(str)
    train_label = str(labels_cfg["train"])
    val_label = labels_cfg.get("val")
    test_label = str(labels_cfg["test"])

    train_mask = split_values == train_label
    test_mask = split_values == test_label
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Split column did not produce non-empty train/test partitions")

    val_adata = None
    if val_label is not None:
        val_mask = split_values == str(val_label)
        val_adata = adata[val_mask].copy() if val_mask.any() else None

    return {
        "train": adata[train_mask].copy(),
        "val": val_adata,
        "test": adata[test_mask].copy(),
    }


def _filter_cells_by_counts(adata, min_counts: int):
    matrix = adata.X
    counts = np.asarray(matrix.sum(axis=1)).reshape(-1)
    keep_mask = counts >= min_counts
    return adata[keep_mask].copy()


def _filter_genes_by_counts(adata, min_counts: int):
    matrix = adata.X
    counts = np.asarray(matrix.sum(axis=0)).reshape(-1)
    keep_mask = counts >= min_counts
    return adata[:, keep_mask].copy()


def _normalize_total(matrix, target_sum: float):
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    safe_totals = np.where(totals == 0, 1.0, totals)
    scale = target_sum / safe_totals
    if sparse.issparse(matrix):
        return sparse.diags(scale) @ matrix
    return matrix * scale[:, None]


def _log1p(matrix):
    if sparse.issparse(matrix):
        matrix = matrix.copy()
        matrix.data = np.log1p(matrix.data)
        return matrix
    return np.log1p(matrix)
