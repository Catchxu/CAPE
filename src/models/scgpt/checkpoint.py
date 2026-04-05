import json

from safetensors.torch import load_file

from .configuration_scgpt import ScGptConfig
from .pretrained import resolve_pretrained_dir
from .tokenizer import GeneVocab


def load_scgpt_checkpoint_assets(pretrained_model_name_or_path, **kwargs):
    checkpoint_dir = resolve_pretrained_dir(pretrained_model_name_or_path, **kwargs)
    config = ScGptConfig.from_pretrained(str(checkpoint_dir), **kwargs)
    vocab = GeneVocab.from_file(checkpoint_dir / config.vocab_file)
    weights = load_file(str(checkpoint_dir / config.weight_file))
    preprocessor_config = json.loads(
        (checkpoint_dir / config.preprocessor_file).read_text(encoding="utf-8")
    )
    return config, vocab, weights, preprocessor_config, checkpoint_dir
