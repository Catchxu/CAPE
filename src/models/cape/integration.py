from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ...data.dataset import DictionaryTensorDataset
from ..pretrained import get_pretrained_source
from ..scbert.configuration_scbert import ScBertConfig
from ..scbert.processing_scbert import ScBertProcessor
from ..scgpt.configuration_scgpt import ScGptConfig
from ..scgpt.processing_scgpt import ScGptProcessor
from .dag_sem import DagSemModel
from .rope import priority_to_rank_positions


def run_shared_cape_stage(
    config: dict[str, Any],
    train_adata,
    val_adata,
    test_adata,
    logger,
    result_dir: Path,
) -> dict[str, Any] | None:
    cape_cfg = _resolve_cape_config(config)
    if not cape_cfg["enabled"]:
        logger.info("Shared CAPE stage disabled")
        return None

    model_name = str(cape_cfg.get("encoder_backend", config["model"]["name"]))
    pretrained_source = get_pretrained_source(config["model"])
    dataset_spec = _build_cape_dataset_spec(
        model_name=model_name,
        pretrained_source=pretrained_source,
        train_adata=train_adata,
        val_adata=val_adata,
        test_adata=test_adata,
        gene_column=config["data"].get("gene_column"),
        max_genes=int(cape_cfg["max_genes"]),
        logger=logger,
    )
    if dataset_spec["train_matrix"].shape[1] == 0:
        logger.warning("No genes available for CAPE stage after backend matching; skipping CAPE")
        return None

    device = _resolve_device(config["run"]["device"])
    cape_model = DagSemModel(
        num_genes=int(dataset_spec["train_matrix"].shape[1]),
        embed_dim=int(cape_cfg["embed_dim"]),
        hidden_dim=int(cape_cfg["hidden_dim"]),
        num_heads=int(cape_cfg["num_heads"]),
        tau=float(cape_cfg["tau"]),
        dropout=float(cape_cfg["dropout"]),
        beta_kl=float(cape_cfg["beta_kl"]),
        lambda_sparse=float(cape_cfg["lambda_sparse"]),
        use_flash_attn=bool(cape_cfg["use_flash_attn"]),
        num_predictor_layers=int(cape_cfg["num_predictor_layers"]),
        encoder_backend=model_name,
        encoder_pretrained_model_name_or_path=pretrained_source,
        freeze_encoder_backbone=True,
        encoder_gene_token_ids=dataset_spec["selected_gene_token_ids"],
    ).to(device)

    logger.info(
        "Training shared CAPE stage with backend=%s selected_genes=%d device=%s",
        model_name,
        dataset_spec["train_matrix"].shape[1],
        device,
    )
    _train_cape_model(
        model=cape_model,
        train_matrix=dataset_spec["train_matrix"],
        val_matrix=dataset_spec["val_matrix"],
        cape_cfg=cape_cfg,
        device=device,
        logger=logger,
    )

    split_outputs = {}
    for split_name, matrix in (
        ("train", dataset_spec["train_matrix"]),
        ("val", dataset_spec["val_matrix"]),
        ("test", dataset_spec["test_matrix"]),
    ):
        if matrix is None:
            split_outputs[split_name] = None
            continue
        split_outputs[split_name] = _infer_cape_outputs(
            model=cape_model,
            matrix=matrix,
            batch_size=int(cape_cfg["batch_size"]),
            device=device,
            max_positions=int(dataset_spec["max_positions"]),
        )

    artifact_dir = result_dir / "cape"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _save_cape_artifacts(
        artifact_dir=artifact_dir,
        dataset_spec=dataset_spec,
        split_outputs=split_outputs,
    )

    return {
        "artifact_dir": str(artifact_dir),
        "selected_gene_names": dataset_spec["selected_gene_names"],
        "selected_gene_token_ids": dataset_spec["selected_gene_token_ids"],
        "max_positions": int(dataset_spec["max_positions"]),
        "split_outputs": split_outputs,
        "model_name": model_name,
    }


def build_scbert_external_positions(
    rank_positions: Tensor | None,
    gene_token_ids: list[int],
    seq_len: int,
    device: torch.device,
) -> Tensor | None:
    if rank_positions is None:
        return None
    external_positions = torch.zeros(rank_positions.size(0), seq_len, dtype=torch.long, device=device)
    if gene_token_ids:
        external_positions[:, torch.tensor(gene_token_ids, dtype=torch.long, device=device)] = (
            rank_positions.to(device) + 1
        )
    return external_positions


def build_scgpt_external_positions(
    rank_positions: Tensor | None,
    gene_token_ids: list[int],
    input_gene_ids: Tensor,
    device: torch.device,
) -> Tensor | None:
    if rank_positions is None:
        return None
    if not gene_token_ids:
        return torch.zeros_like(input_gene_ids, device=device)

    max_token_id = int(max(max(gene_token_ids), int(input_gene_ids.max().item())))
    lookup = torch.full((max_token_id + 1,), -1, dtype=torch.long, device=device)
    lookup[torch.tensor(gene_token_ids, dtype=torch.long, device=device)] = torch.arange(
        len(gene_token_ids),
        device=device,
    )
    gathered_indices = lookup[input_gene_ids.to(device)]
    valid_mask = gathered_indices.ge(0)
    safe_indices = gathered_indices.clamp_min(0)
    selected_positions = torch.gather(rank_positions.to(device), 1, safe_indices)
    external_positions = torch.zeros_like(input_gene_ids, dtype=torch.long, device=device)
    external_positions[valid_mask] = selected_positions[valid_mask] + 1
    return external_positions


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_cape_config(config: dict[str, Any]) -> dict[str, Any]:
    model_name = config["model"]["name"]
    model_cfg = config["model"]
    pretrained_source = get_pretrained_source(model_cfg)
    if model_name == "scbert":
        foundation_cfg = ScBertConfig.from_pretrained(pretrained_source)
        default_embed_dim = int(foundation_cfg.hidden_size)
        default_max_positions = int(foundation_cfg.max_position_embeddings - 1)
    else:
        foundation_cfg = ScGptConfig.from_pretrained(pretrained_source)
        default_embed_dim = int(foundation_cfg.hidden_size)
        default_max_positions = int(foundation_cfg.max_position_embeddings - 1)

    cape_cfg = {
        "enabled": True,
        "batch_size": int(config["train"]["batch_size"]),
        "epochs": 5,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "embed_dim": default_embed_dim,
        "hidden_dim": max(default_embed_dim * 2, 128),
        "num_heads": _default_num_heads(default_embed_dim),
        "tau": 1.0,
        "dropout": 0.1,
        "beta_kl": 1.0,
        "lambda_sparse": 1e-4,
        "num_predictor_layers": 2,
        "use_flash_attn": True,
        "max_genes": min(default_max_positions, 512),
        "max_positions": default_max_positions,
    }
    cape_cfg.update(config.get("cape", {}))
    if int(cape_cfg["embed_dim"]) % int(cape_cfg["num_heads"]) != 0:
        raise ValueError(
            f"CAPE embed_dim={cape_cfg['embed_dim']} must be divisible by num_heads={cape_cfg['num_heads']}"
        )
    return cape_cfg


def _default_num_heads(embed_dim: int) -> int:
    for num_heads in range(min(8, embed_dim), 0, -1):
        if embed_dim % num_heads == 0:
            return num_heads
    return 1


def _build_cape_dataset_spec(
    model_name: str,
    pretrained_source: str,
    train_adata,
    val_adata,
    test_adata,
    gene_column: str | None,
    max_genes: int,
    logger,
) -> dict[str, Any]:
    train_gene_names = _get_gene_names(train_adata, gene_column)
    matched = _match_genes(train_gene_names, model_name, pretrained_source)
    if not matched:
        return {
            "selected_gene_names": [],
            "selected_gene_token_ids": [],
            "train_matrix": np.zeros((train_adata.n_obs, 0), dtype=np.float32),
            "val_matrix": None if val_adata is None else np.zeros((val_adata.n_obs, 0), dtype=np.float32),
            "test_matrix": np.zeros((test_adata.n_obs, 0), dtype=np.float32),
            "max_positions": 1,
        }

    selected = _select_genes_by_variance(train_adata, matched, max_genes=max_genes)
    selected_indices = [item["dataset_index"] for item in selected]
    selected_gene_names = [item["gene_name"] for item in selected]
    selected_gene_token_ids = [item["token_id"] for item in selected]

    logger.info(
        "CAPE matched %d genes to %s assets and selected %d for graph learning",
        len(matched),
        model_name,
        len(selected_gene_names),
    )

    return {
        "selected_gene_names": selected_gene_names,
        "selected_gene_token_ids": selected_gene_token_ids,
        "train_matrix": _extract_dense_matrix(train_adata, selected_indices),
        "val_matrix": None if val_adata is None else _extract_dense_matrix(val_adata, selected_indices),
        "test_matrix": _extract_dense_matrix(test_adata, selected_indices),
        "max_positions": int(_resolve_max_positions(model_name, pretrained_source)),
    }


def _resolve_max_positions(model_name: str, pretrained_source: str) -> int:
    if model_name == "scbert":
        return int(ScBertConfig.from_pretrained(pretrained_source).max_position_embeddings - 1)
    config = ScGptConfig.from_pretrained(pretrained_source)
    return int(config.max_position_embeddings - 1)


def _get_gene_names(adata, gene_column: str | None) -> list[str]:
    if gene_column:
        if gene_column not in adata.var:
            raise ValueError(f"Gene column '{gene_column}' not found in AnnData.var")
        return adata.var[gene_column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _match_genes(train_gene_names: list[str], model_name: str, pretrained_source: str) -> list[dict[str, Any]]:
    if model_name == "scbert":
        vocab = ScBertProcessor.from_pretrained(pretrained_source).vocab
        return [
            {"dataset_index": idx, "gene_name": gene_name, "token_id": int(vocab[gene_name])}
            for idx, gene_name in enumerate(train_gene_names)
            if gene_name in vocab
        ]

    vocab = ScGptProcessor.from_pretrained(pretrained_source).vocab.get_stoi()
    return [
        {"dataset_index": idx, "gene_name": gene_name, "token_id": int(vocab[gene_name])}
        for idx, gene_name in enumerate(train_gene_names)
        if gene_name in vocab and not gene_name.startswith("<")
    ]


def _extract_dense_matrix(adata, selected_indices: list[int]) -> np.ndarray:
    matrix = adata[:, selected_indices].X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _select_genes_by_variance(adata, matched_genes: list[dict[str, Any]], max_genes: int) -> list[dict[str, Any]]:
    if len(matched_genes) <= max_genes:
        return matched_genes

    selected_indices = [item["dataset_index"] for item in matched_genes]
    matrix = _extract_dense_matrix(adata, selected_indices)
    variances = matrix.var(axis=0)
    topk = np.argsort(variances)[::-1][:max_genes]
    topk_sorted = np.sort(topk)
    return [matched_genes[int(idx)] for idx in topk_sorted]


def _train_cape_model(
    model: DagSemModel,
    train_matrix: np.ndarray,
    val_matrix: np.ndarray | None,
    cape_cfg: dict[str, Any],
    device: torch.device,
    logger,
) -> None:
    train_loader = _build_cape_loader(train_matrix, int(cape_cfg["batch_size"]), shuffle=True)
    val_loader = (
        _build_cape_loader(val_matrix, int(cape_cfg["batch_size"]), shuffle=False)
        if val_matrix is not None
        else None
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(cape_cfg["learning_rate"]),
        weight_decay=float(cape_cfg["weight_decay"]),
    )

    best_metric = float("inf")
    best_state = deepcopy(model.state_dict())
    epochs = int(cape_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss = _run_cape_epoch(model, train_loader, optimizer, device)
        logger.info("CAPE epoch %d/%d | train_loss=%.4f", epoch, epochs, train_loss)
        if val_loader is None:
            best_state = deepcopy(model.state_dict())
            continue
        val_loss = _evaluate_cape(model, val_loader, device)
        logger.info("CAPE epoch %d/%d | val_loss=%.4f", epoch, epochs, val_loss)
        if val_loss <= best_metric:
            best_metric = val_loss
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)


def _build_cape_loader(matrix: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = DictionaryTensorDataset({"x": torch.from_numpy(matrix).float()})
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_cape_epoch(model: DagSemModel, loader: DataLoader, optimizer: AdamW, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        optimizer.zero_grad()
        outputs = model(x)
        losses = model.compute_loss(x, outputs)
        losses["loss"].backward()
        optimizer.step()
        total_loss += float(losses["loss"].detach().item())
    return total_loss / max(len(loader), 1)


def _evaluate_cape(model: DagSemModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            outputs = model(x)
            losses = model.compute_loss(x, outputs)
            total_loss += float(losses["loss"].detach().item())
    return total_loss / max(len(loader), 1)


def _infer_cape_outputs(
    model: DagSemModel,
    matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
    max_positions: int,
) -> dict[str, Tensor]:
    loader = _build_cape_loader(matrix, batch_size=batch_size, shuffle=False)
    priorities = []
    ranks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            outputs = model(x)
            priority = outputs["priority"]
            rank = priority_to_rank_positions(priority, max_positions=max_positions)
            priorities.append(priority.cpu())
            ranks.append(rank.cpu())
    return {
        "priority": torch.cat(priorities, dim=0),
        "rank": torch.cat(ranks, dim=0),
    }


def _save_cape_artifacts(
    artifact_dir: Path,
    dataset_spec: dict[str, Any],
    split_outputs: dict[str, dict[str, Tensor] | None],
) -> None:
    (artifact_dir / "selected_gene_names.json").write_text(
        json.dumps(dataset_spec["selected_gene_names"], indent=2),
        encoding="utf-8",
    )
    np.save(
        artifact_dir / "selected_gene_token_ids.npy",
        np.asarray(dataset_spec["selected_gene_token_ids"], dtype=np.int64),
    )
    for split_name, outputs in split_outputs.items():
        if outputs is None:
            continue
        np.save(artifact_dir / f"{split_name}_priority.npy", outputs["priority"].numpy())
        np.save(artifact_dir / f"{split_name}_rank.npy", outputs["rank"].numpy())
