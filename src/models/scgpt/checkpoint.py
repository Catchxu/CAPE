import json
from pathlib import Path

import torch

from .tokenizer import GeneVocab


def load_scgpt_checkpoint_assets(pretrained_dir: str):
    checkpoint_dir = Path(pretrained_dir)
    model_path = checkpoint_dir / "best_model.pt"
    args_path = checkpoint_dir / "args.json"
    vocab_path = checkpoint_dir / "vocab.json"

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
    return checkpoint_args, vocab, weights


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
