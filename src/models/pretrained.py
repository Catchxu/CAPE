from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ASSET_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
    "model.safetensors",
]


def resolve_pretrained_dir(
    pretrained_model_name_or_path,
    *,
    cache_dir=None,
    revision=None,
    local_files_only=False,
    allow_patterns=None,
):
    source = str(pretrained_model_name_or_path)
    path = Path(source)
    if path.exists():
        return path.resolve()

    downloaded = snapshot_download(
        repo_id=source,
        repo_type="model",
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=allow_patterns or DEFAULT_ASSET_PATTERNS,
    )
    return Path(downloaded).resolve()


def get_pretrained_source(model_cfg):
    path = model_cfg.get("path")
    if path is not None and Path(str(path)).exists():
        return str(path)
    return model_cfg.get("hf_repo_id") or str(path)


def resolve_pretrained_from_kwargs(pretrained_model_name_or_path, kwargs):
    return resolve_pretrained_dir(
        pretrained_model_name_or_path,
        cache_dir=kwargs.get("cache_dir"),
        revision=kwargs.get("revision"),
        local_files_only=kwargs.get("local_files_only", False),
    )
