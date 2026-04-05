from ..tasks.cell_type_annotation import run_cell_type_annotation


def run_cta_pipeline(config):
    if config["task"]["name"] != "CTA":
        raise ValueError(f"Unsupported task: {config['task']['name']}")
    return run_cell_type_annotation(config)
