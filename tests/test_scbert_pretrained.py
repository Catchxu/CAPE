from pathlib import Path

from src.models.scbert.pretrained import get_pretrained_source


def test_local_scbert_path_takes_precedence_over_hf_repo_id(tmp_path):
    local_bundle = tmp_path / "scbert"
    local_bundle.mkdir()

    source = get_pretrained_source(
        {
            "path": str(local_bundle),
            "hf_repo_id": "kaichenxu/cape_scbert",
        }
    )

    assert Path(source) == local_bundle


def test_hf_repo_id_used_when_local_scbert_path_missing(tmp_path):
    missing_bundle = tmp_path / "missing_scbert"

    source = get_pretrained_source(
        {
            "path": str(missing_bundle),
            "hf_repo_id": "kaichenxu/cape_scbert",
        }
    )

    assert source == "kaichenxu/cape_scbert"
