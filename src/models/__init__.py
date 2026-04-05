from .scbert.wrapper import ScBertBackend
from .scgpt.wrapper import ScGptBackend


def get_backend(model_name: str):
    if model_name == "scbert":
        return ScBertBackend()
    if model_name == "scgpt":
        return ScGptBackend()
    raise ValueError(f"Unsupported model backend: {model_name}")
