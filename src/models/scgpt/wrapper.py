from copy import deepcopy

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ...data.dataset import DictionaryTensorDataset
from ...data.label_utils import encode_labels
from ...CTA.metrics import compute_cta_metrics
from ..pretrained import get_pretrained_source
from .configuration_scgpt import ScGptConfig
from .model import ClsDecoder
from .modeling_scgpt import ScGptModel
from .processing_scgpt import ScGptProcessor
from .utils import get_batch_ids


def _resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class ScGptClassifier(nn.Module):
    def __init__(self, backbone: ScGptModel, num_classes: int, nlayers_cls: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = ClsDecoder(
            d_model=backbone.config.hidden_size,
            n_cls=num_classes,
            nlayers=nlayers_cls,
        )

    def forward(self, input_gene_ids, values, src_key_padding_mask, batch_labels=None):
        outputs = self.backbone(
            input_gene_ids=input_gene_ids,
            values=values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels,
        )
        return self.classifier(outputs.pooler_output)


class ScGptBackend:
    def run_cta(self, config, train_adata, val_adata, test_adata, label_encoder, logger):
        device = _resolve_device(config["run"]["device"])
        model_cfg = config["model"]
        pretrained_source = get_pretrained_source(model_cfg)
        logger.info("Using device %s", device)

        processor = ScGptProcessor.from_pretrained(pretrained_source)
        model_config = ScGptConfig.from_pretrained(pretrained_source)
        logger.info(
            "Loaded scGPT backbone source=%s vocab_size=%d max_seq_len=%d",
            pretrained_source,
            len(processor.vocab),
            model_config.max_position_embeddings,
        )

        train_loader, batch_categories = self._build_loader(
            train_adata, label_encoder, config, processor, shuffle=True
        )
        val_loader = (
            self._build_loader(val_adata, label_encoder, config, processor, shuffle=False)[0]
            if val_adata is not None
            else None
        )
        test_loader = self._build_loader(
            test_adata, label_encoder, config, processor, shuffle=False
        )[0]

        backbone = ScGptModel.from_pretrained(
            pretrained_source,
            config=model_config,
        )
        model = ScGptClassifier(
            backbone=backbone,
            num_classes=len(label_encoder.classes_),
            nlayers_cls=int(model_config.nlayers_cls),
        )
        model.to(device)

        train_cfg = config["train"]
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        best_metric = float("-inf")
        best_state = deepcopy(model.state_dict())
        epochs = int(train_cfg["epochs"])
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer, device)
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

    def _build_loader(self, adata, label_encoder, config, processor, shuffle: bool):
        tensors, batch_categories = self._prepare_split_tensors(adata, label_encoder, config, processor)
        dataset = DictionaryTensorDataset(tensors)
        loader = DataLoader(
            dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=shuffle,
        )
        return loader, batch_categories

    def _prepare_split_tensors(self, adata, label_encoder, config, processor):
        encoded = processor.encode_adata(
            adata,
            gene_column=config["data"].get("gene_column"),
            return_tensors="pt",
            mask_ratio=0.0,
        )
        filtered = encoded.pop("adata")
        labels = encode_labels(
            filtered.obs[config["data"]["label_column"]].astype(str).tolist(),
            label_encoder,
        )
        batch_ids, batch_categories = get_batch_ids(filtered, config["data"].get("batch_column"))
        tensors = {
            "gene_ids": encoded["gene_ids"].long(),
            "values": encoded["values"].float(),
            "celltype_labels": torch.tensor(labels, dtype=torch.long),
            "batch_labels": torch.tensor(batch_ids, dtype=torch.long),
        }
        return tensors, batch_categories

    def _train_epoch(self, model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_gene_ids = batch["gene_ids"].to(device)
            input_values = batch["values"].to(device)
            labels = batch["celltype_labels"].to(device)
            batch_labels = batch["batch_labels"].to(device)
            padding_mask = input_gene_ids.eq(model.backbone.config.pad_token_id)
            optimizer.zero_grad()
            logits = model(
                input_gene_ids=input_gene_ids,
                values=input_values,
                src_key_padding_mask=padding_mask,
                batch_labels=batch_labels if model.backbone.config.use_batch_labels else None,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _evaluate(self, model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                input_gene_ids = batch["gene_ids"].to(device)
                input_values = batch["values"].to(device)
                labels = batch["celltype_labels"].to(device)
                batch_labels = batch["batch_labels"].to(device)
                padding_mask = input_gene_ids.eq(model.backbone.config.pad_token_id)
                logits = model(
                    input_gene_ids=input_gene_ids,
                    values=input_values,
                    src_key_padding_mask=padding_mask,
                    batch_labels=batch_labels if model.backbone.config.use_batch_labels else None,
                )
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                total_loss += loss.item()
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
        predictions = torch.cat(all_preds, dim=0).numpy()
        probabilities = torch.cat(all_probs, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        metrics = compute_cta_metrics(targets, predictions)
        return {
            "loss": total_loss / max(len(loader), 1),
            "predictions": predictions,
            "probabilities": probabilities,
            "targets": targets,
            "metrics": metrics,
        }
