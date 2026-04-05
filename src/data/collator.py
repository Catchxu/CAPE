def get_collator(model_name: str):
    if model_name in {"scbert", "scgpt"}:
        return None
    raise ValueError(f"Unsupported collator backend: {model_name}")
