import json
from pathlib import Path

import torch

from .tokenizer import GeneVocab


def load_scgpt_checkpoint_assets(model_cfg):
    checkpoint_dir = Path(model_cfg["path"])
    if not checkpoint_dir.exists():
        hf_repo_id = model_cfg.get("hf_repo_id")
        if hf_repo_id:
            raise FileNotFoundError(
                f"Local scGPT asset bundle not found at {checkpoint_dir}. "
                f"Download assets from Hugging Face repo '{hf_repo_id}' first."
            )
        raise FileNotFoundError(f"Local scGPT asset bundle not found at {checkpoint_dir}")

    file_cfg = model_cfg.get("files", {})
    model_path = checkpoint_dir / file_cfg.get("weights", "best_model.pt")
    args_path = checkpoint_dir / file_cfg.get("config", "args.json")
    vocab_path = checkpoint_dir / file_cfg.get("vocab", "vocab.json")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing scGPT checkpoint weights: {model_path}")
    if not args_path.exists():
        raise FileNotFoundError(f"Missing scGPT args file: {args_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing scGPT vocab file: {vocab_path}")

    with args_path.open("r", encoding="utf-8") as handle:
        checkpoint_args = json.load(handle)
    vocab = GeneVocab.from_file(vocab_path)
    weights = torch.load(model_path, map_location="cpu")
    return checkpoint_args, vocab, weights, checkpoint_dir


def load_scgpt_pretrained(model, pretrained_params, logger):
    model_dict = model.state_dict()
    compatible = {
        key: value
        for key, value in pretrained_params.items()
        if key in model_dict and value.shape == model_dict[key].shape
    }
    logger.info("Loading %d compatible pretrained parameters into scGPT", len(compatible))
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    return model
