from copy import deepcopy

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ...data.dataset import DictionaryTensorDataset
from ...data.label_utils import encode_labels
from ...utils.metrics import compute_classification_metrics
from .checkpoint import load_scbert_pretrained
from .model import ScBertClassifier
from .utils import align_adata_to_gene_order, load_gene_list, matrix_to_scbert_sequences


def _resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class ScBertBackend:
    def run_cta(self, config, train_adata, val_adata, test_adata, label_encoder, logger):
        model_cfg = config["model"]
        train_cfg = config["train"]
        device = _resolve_device(config["run"]["device"])
        logger.info("Using device %s", device)

        reference_genes = load_gene_list(model_cfg["reference_genes_path"])
        train_loader = self._build_loader(
            train_adata,
            label_encoder,
            config,
            reference_genes,
            shuffle=True,
        )
        val_loader = (
            self._build_loader(val_adata, label_encoder, config, reference_genes, shuffle=False)
            if val_adata is not None
            else None
        )
        test_loader = self._build_loader(
            test_adata,
            label_encoder,
            config,
            reference_genes,
            shuffle=False,
        )

        model = ScBertClassifier(
            architecture=model_cfg["architecture"],
            num_classes=len(label_encoder.classes_),
        )
        model = load_scbert_pretrained(model, model_cfg["pretrained_path"], logger)
        self._apply_freeze_policy(model, model_cfg.get("freeze", {}), logger)
        model.to(device)

        optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        criterion = nn.CrossEntropyLoss()

        best_metric = float("-inf")
        best_state = deepcopy(model.state_dict())
        epochs = int(train_cfg["epochs"])
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, device)
            logger.info("Epoch %d/%d | train_loss=%.4f", epoch, epochs, train_loss)
            if val_loader is not None:
                val_result = self._evaluate(model, val_loader, criterion, device)
                logger.info(
                    "Epoch %d/%d | val_loss=%.4f | val_macro_f1=%.4f | val_accuracy=%.4f",
                    epoch,
                    epochs,
                    val_result["loss"],
                    val_result["metrics"]["macro_f1"],
                    val_result["metrics"]["accuracy"],
                )
                if val_result["metrics"]["macro_f1"] >= best_metric:
                    best_metric = val_result["metrics"]["macro_f1"]
                    best_state = deepcopy(model.state_dict())
            else:
                best_state = deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        test_result = self._evaluate(model, test_loader, criterion, device)
        predicted_labels = label_encoder.inverse_transform(test_result["predictions"].tolist())
        true_labels = label_encoder.inverse_transform(test_result["targets"].tolist())
        return {
            "metrics": test_result["metrics"],
            "predicted_ids": test_result["predictions"].tolist(),
            "predicted_labels": predicted_labels,
            "true_ids": test_result["targets"].tolist(),
            "true_labels": true_labels,
            "probabilities": test_result["probabilities"],
        }

    def _build_loader(self, adata, label_encoder, config, reference_genes, shuffle: bool):
        sequences, labels = self._prepare_split(adata, label_encoder, config, reference_genes)
        dataset = DictionaryTensorDataset(
            {
                "input_ids": torch.from_numpy(sequences).long(),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )
        return DataLoader(
            dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=shuffle,
        )

    def _prepare_split(self, adata, label_encoder, config, reference_genes):
        aligned = align_adata_to_gene_order(
            adata,
            reference_genes,
            gene_column=config["data"].get("gene_column"),
        )
        sequences = matrix_to_scbert_sequences(
            aligned_matrix=aligned,
            max_bin=int(config["model"]["architecture"]["bin_num"]),
        )
        labels = encode_labels(
            adata.obs[config["data"]["label_column"]].astype(str).tolist(),
            label_encoder,
        )
        return sequences, labels

    def _apply_freeze_policy(self, model, freeze_cfg, logger):
        if not freeze_cfg.get("freeze_backbone", True):
            return
        for param in model.backbone.parameters():
            param.requires_grad = False
        if freeze_cfg.get("train_norm", True):
            for param in model.backbone.norm.parameters():
                param.requires_grad = True
        if freeze_cfg.get("train_last_performer_layer", True):
            for param in model.backbone.performer.net.layers[-2].parameters():
                param.requires_grad = True
        for param in model.backbone.to_out.parameters():
            param.requires_grad = True
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        logger.info("scBERT trainable parameters after freezing: %d", trainable)

    def _train_epoch(self, model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _evaluate(self, model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        predictions = []
        probabilities = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                total_loss += loss.item()
                predictions.append(preds.cpu())
                probabilities.append(probs.cpu())
                targets.append(labels.cpu())
        predictions = torch.cat(predictions, dim=0).numpy()
        probabilities = torch.cat(probabilities, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        metrics = compute_classification_metrics(targets, predictions)
        return {
            "loss": total_loss / max(len(loader), 1),
            "predictions": predictions,
            "probabilities": probabilities,
            "targets": targets,
            "metrics": metrics,
        }
