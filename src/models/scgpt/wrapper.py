import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ...data.dataset import DictionaryTensorDataset
from ...data.label_utils import encode_labels
from ...utils.metrics import compute_classification_metrics
from .checkpoint import load_scgpt_checkpoint_assets, load_scgpt_pretrained
from .model import TransformerModel
from .tokenizer import random_mask_value, tokenize_and_pad_batch
from .utils import bin_matrix, filter_adata_by_vocab, get_batch_ids, get_input_matrix


def _resolve_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class ScGptBackend:
    def run_cta(self, config, train_adata, val_adata, test_adata, label_encoder, logger):
        device = _resolve_device(config["run"]["device"])
        logger.info("Using device %s", device)
        checkpoint_args, vocab, pretrained_weights, checkpoint_dir = load_scgpt_checkpoint_assets(
            config["model"]
        )
        logger.info(
            "Loaded scGPT bundle: path=%s hf_repo_id=%s vocab_size=%d",
            checkpoint_dir,
            config["model"].get("hf_repo_id"),
            len(vocab),
        )
        for token in ["<pad>", "<cls>", "<eoc>"]:
            vocab.append_token(token)

        train_loader, batch_categories = self._build_loader(
            train_adata, label_encoder, config, vocab, shuffle=True
        )
        val_loader = (
            self._build_loader(val_adata, label_encoder, config, vocab, shuffle=False)[0]
            if val_adata is not None
            else None
        )
        test_loader = self._build_loader(
            test_adata, label_encoder, config, vocab, shuffle=False
        )[0]

        model = self._build_model(
            config=config,
            checkpoint_args=checkpoint_args,
            vocab=vocab,
            num_classes=len(label_encoder.classes_),
            num_batch_labels=len(batch_categories),
        )
        model = load_scgpt_pretrained(model, pretrained_weights, logger)
        model.to(device)

        train_cfg = config["train"]
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        best_metric = float("-inf")
        best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        epochs = int(train_cfg["epochs"])
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer, device, config, vocab)
            logger.info("Epoch %d/%d | train_loss=%.4f", epoch, epochs, train_loss)
            if val_loader is not None:
                val_result = self._evaluate(model, val_loader, criterion, device, config, vocab)
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
                    best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            else:
                best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_result = self._evaluate(model, test_loader, criterion, device, config, vocab)
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

    def _build_model(self, config, checkpoint_args, vocab, num_classes: int, num_batch_labels: int):
        arch_cfg = config["model"]["architecture"]
        embsize = int(checkpoint_args.get("embsize", arch_cfg["embsize"]))
        nhead = int(checkpoint_args.get("nheads", arch_cfg["nhead"]))
        d_hid = int(checkpoint_args.get("d_hid", arch_cfg["d_hid"]))
        nlayers = int(checkpoint_args.get("nlayers", arch_cfg["nlayers"]))
        nlayers_cls = int(checkpoint_args.get("n_layers_cls", arch_cfg["nlayers_cls"]))
        return TransformerModel(
            ntoken=len(vocab),
            d_model=embsize,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            nlayers_cls=nlayers_cls,
            n_cls=num_classes,
            vocab=vocab,
            dropout=float(arch_cfg["dropout"]),
            pad_token=arch_cfg["pad_token"],
            pad_value=int(arch_cfg["pad_value"]),
            do_mvc=False,
            do_dab=False,
            use_batch_labels=bool(arch_cfg.get("use_batch_labels", False)),
            num_batch_labels=num_batch_labels,
            domain_spec_batchnorm=arch_cfg.get("domain_spec_batchnorm", False),
            input_emb_style=arch_cfg.get("input_emb_style", "category"),
            n_input_bins=int(arch_cfg.get("n_input_bins", 51)),
            cell_emb_style=arch_cfg.get("cell_emb_style", "cls"),
            use_fast_transformer=bool(arch_cfg.get("use_fast_transformer", False)),
        )

    def _build_loader(self, adata, label_encoder, config, vocab, shuffle: bool):
        tensors, batch_categories = self._prepare_split_tensors(adata, label_encoder, config, vocab)
        dataset = DictionaryTensorDataset(tensors)
        loader = DataLoader(
            dataset,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=shuffle,
        )
        return loader, batch_categories

    def _prepare_split_tensors(self, adata, label_encoder, config, vocab):
        filtered, _, gene_ids = filter_adata_by_vocab(
            adata,
            vocab,
            gene_column=config["data"].get("gene_column"),
        )
        matrix = get_input_matrix(filtered)
        n_bins = int(config["model"]["architecture"].get("n_input_bins", 51))
        binned = bin_matrix(matrix, n_bins)
        tokenized = tokenize_and_pad_batch(
            data=binned,
            gene_ids=gene_ids,
            max_len=int(config["model"]["architecture"]["max_seq_len"]),
            vocab=vocab,
            pad_token=config["model"]["architecture"]["pad_token"],
            pad_value=int(config["model"]["architecture"]["pad_value"]),
            append_cls=True,
            include_zero_gene=bool(config["model"]["architecture"].get("include_zero_gene", False)),
        )
        values = random_mask_value(
            tokenized["values"],
            mask_ratio=0.0,
            mask_value=int(config["model"]["architecture"]["mask_value"]),
            pad_value=int(config["model"]["architecture"]["pad_value"]),
        )
        labels = encode_labels(
            filtered.obs[config["data"]["label_column"]].astype(str).tolist(),
            label_encoder,
        )
        batch_ids, batch_categories = get_batch_ids(filtered, config["data"].get("batch_column"))
        tensors = {
            "gene_ids": tokenized["genes"].long(),
            "values": values,
            "celltype_labels": torch.tensor(labels, dtype=torch.long),
            "batch_labels": torch.tensor(batch_ids, dtype=torch.long),
        }
        return tensors, batch_categories

    def _train_epoch(self, model, loader, criterion, optimizer, device, config, vocab):
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_gene_ids = batch["gene_ids"].to(device)
            input_values = batch["values"].to(device)
            labels = batch["celltype_labels"].to(device)
            batch_labels = batch["batch_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[config["model"]["architecture"]["pad_token"]])
            optimizer.zero_grad()
            output = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config["model"]["architecture"].get("use_batch_labels", False) else None,
                CLS=True,
                MVC=False,
                ECS=False,
            )
            loss = criterion(output["cls_output"], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def _evaluate(self, model, loader, criterion, device, config, vocab):
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
                src_key_padding_mask = input_gene_ids.eq(vocab[config["model"]["architecture"]["pad_token"]])
                output = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config["model"]["architecture"].get("use_batch_labels", False) else None,
                    CLS=True,
                    MVC=False,
                    ECS=False,
                )
                logits = output["cls_output"]
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
        metrics = compute_classification_metrics(targets, predictions)
        return {
            "loss": total_loss / max(len(loader), 1),
            "predictions": predictions,
            "probabilities": probabilities,
            "targets": targets,
            "metrics": metrics,
        }
