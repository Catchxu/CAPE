from src.utils.config import load_config


def test_experiment_configs_share_common_schema():
    scbert_cfg = load_config("configs/CTA/scbert_CTA.yaml")
    scgpt_cfg = load_config("configs/CTA/scgpt_CTA.yaml")

    assert set(scbert_cfg.keys()) == {"data", "model", "run", "task", "train"}
    assert set(scgpt_cfg.keys()) == {"data", "model", "run", "task", "train"}
    assert scbert_cfg["task"]["name"] == "CTA"
    assert scgpt_cfg["task"]["name"] == "CTA"
    assert scbert_cfg["model"]["name"] == "scbert"
    assert scgpt_cfg["model"]["name"] == "scgpt"
