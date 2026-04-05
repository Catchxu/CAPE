from pathlib import Path


def get_tokenizer(model_name: str, config: dict):
    if model_name == "scgpt":
        from ..models.scgpt.tokenizer import GeneVocab

        return GeneVocab.from_file(Path(config["model"]["pretrained_dir"]) / "vocab.json")
    if model_name == "scbert":
        return None
    raise ValueError(f"Unsupported tokenizer backend: {model_name}")
