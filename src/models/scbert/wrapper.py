from copy import deepcopy

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ...data.dataset import DictionaryTensorDataset
from ...data.label_utils import encode_labels
from ...CTA.metrics import compute_cta_metrics
from ..cape.integration import build_scbert_external_positions
from ..pretrained import get_pretrained_source
from .model import ScBertClassifier
from .modeling_scbert import ScBertModel
from .processing_scbert import ScBertProcessor
from .configuration_scbert import ScBertConfig


def _resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class ScBertBackend:
    def run_cta(self, config, train_adata, val_adata, test_adata, label_encoder, logger, shared_state=None):
        model_cfg = config["model"]
        train_cfg = config["train"]
        device = _resolve_device(config["run"]["device"])
        pretrained_source = get_pretrained_source(model_cfg)
        logger.info("Using device %s", device)

        processor = ScBertProcessor.from_pretrained(pretrained_source)
        model_config = ScBertConfig.from_pretrained(pretrained_source)
        logger.info(
            "Loaded scBERT backbone source=%s vocab_size=%d max_seq_len=%d",
            pretrained_source,
            len(processor.vocab),
            model_config.max_position_embeddings,
        )

        train_loader = self._build_loader(
            train_adata,
            label_encoder,
            config,
            processor,
            model_config,
            split_name="train",
            shared_state=shared_state,
            shuffle=True,
        )
        val_loader = (
            self._build_loader(
                val_adata,
                label_encoder,
                config,
                processor,
                model_config,
                split_name="val",
                shared_state=shared_state,
                shuffle=False,
            )
            if val_adata is not None
            else None
        )
        test_loader = self._build_loader(
            test_adata,
            label_encoder,
            config,
            processor,
            model_config,
            split_name="test",
            shared_state=shared_state,
            shuffle=False,
        )

        backbone = ScBertModel.from_pretrained(
            pretrained_source,
            config=model_config,
        )
        model = ScBertClassifier(
            backbone=backbone,
            architecture={
                "max_seq_len": model_config.max_position_embeddings,
                "dropout": model_config.classifier_dropout,
                "head_hidden_dim": model_config.classifier_hidden_dim,
            },
            num_classes=len(label_encoder.classes_),
        )
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

    def _build_loader(
        self,
        adata,
        label_encoder,
        config,
        processor,
        model_config,
        split_name: str,
        shared_state,
        shuffle: bool,
    ):
        tensors = processor.encode_adata(
            adata,
            gene_column=config["data"].get("gene_column"),
            return_tensors="pt",
        )
        labels = encode_labels(
            adata.obs[config["data"]["label_column"]].astype(str).tolist(),
            label_encoder,
        )
        dataset_tensors = {
            "input_ids": tensors["input_ids"].long(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        external_positions = self._build_external_positions(
            split_name=split_name,
            shared_state=shared_state,
            model_config=model_config,
            device=dataset_tensors["input_ids"].device,
        )
        if external_positions is not None:
            dataset_tensors["external_positions"] = external_positions.long()
        dataset = DictionaryTensorDataset(dataset_tensors)
        return DataLoader(
            dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=shuffle,
        )

    def _build_external_positions(self, split_name, shared_state, model_config, device):
        if shared_state is None:
            return None
        split_outputs = shared_state["split_outputs"].get(split_name)
        if split_outputs is None:
            return None
        return build_scbert_external_positions(
            rank_positions=split_outputs["rank"],
            gene_token_ids=shared_state["selected_gene_token_ids"],
            seq_len=int(model_config.max_position_embeddings),
            device=device,
        )

    def _apply_freeze_policy(self, model, freeze_cfg, logger):
        if not freeze_cfg.get("freeze_backbone", True):
            return
        for param in model.backbone.parameters():
            param.requires_grad = False
        if freeze_cfg.get("train_norm", True):
            for param in model.backbone.scbert.norm.parameters():
                param.requires_grad = True
        if freeze_cfg.get("train_last_performer_layer", True):
            for param in model.backbone.scbert.performer.net.layers[-2].parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        logger.info("scBERT trainable parameters after freezing: %d", trainable)

    def _train_epoch(self, model, loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            external_positions = batch.get("external_positions")
            external_positions = external_positions.to(device) if external_positions is not None else None
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, external_positions=external_positions)
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
                external_positions = batch.get("external_positions")
                external_positions = external_positions.to(device) if external_positions is not None else None
                logits = model(input_ids=input_ids, external_positions=external_positions)
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
        metrics = compute_cta_metrics(targets, predictions)
        return {
            "loss": total_loss / max(len(loader), 1),
            "predictions": predictions,
            "probabilities": probabilities,
            "targets": targets,
            "metrics": metrics,
        }
