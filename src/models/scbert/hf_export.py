import shutil
from pathlib import Path


HF_CODE_FILES = [
    "__init__.py",
    "configuration_scbert.py",
    "modeling_scbert.py",
    "processing_scbert.py",
    "performer_pytorch.py",
    "reversible.py",
]


def sync_hf_code_files(target_dir):
    source_dir = Path(__file__).resolve().parent
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in HF_CODE_FILES:
        shutil.copy2(source_dir / filename, target_dir / filename)
