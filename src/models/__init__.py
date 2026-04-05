def get_backend(model_name: str):
    if model_name == "scbert":
        from .scbert.wrapper import ScBertBackend

        return ScBertBackend()
    if model_name == "scgpt":
        from .scgpt.wrapper import ScGptBackend

        return ScGptBackend()
    raise ValueError(f"Unsupported model backend: {model_name}")
