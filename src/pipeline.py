from .tasks.cell_type_annotation import run_cell_type_annotation


def run_pipeline(config):
    task_name = config["task"]["name"]
    if task_name == "CTA":
        return run_cell_type_annotation(config)
    raise ValueError(f"Unsupported task: {task_name}")
