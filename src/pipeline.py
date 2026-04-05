from .CTA import run_cta


def run_pipeline(config):
    task_name = config["task"]["name"]
    if task_name == "CTA":
        return run_cta(config)
    raise ValueError(f"Unsupported task: {task_name}")
