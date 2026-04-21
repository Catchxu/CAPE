# CAPE: Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features

Official implementation for the CAPE paper.

CAPE is a positional encoding method for transformer-based representation
learning of non-sequential features. It learns causality-aware feature positions
from data and integrates them into transformer self-attention through a rotary
positional encoding form. The current code supports CAPE integration with
scBERT and scGPT, with cell type annotation (CTA) provided as an application
workflow.

<br/>
<div align=center>
<img src="/docs/CAPE.jpg" width="70%">
</div>
<br/>

## Highlights

- CAPE positional encoding for non-sequential, causally-related features.
- Current backbone support for `scgpt` and `scbert`.
- CTA application workflow for `.h5ad` AnnData inputs.
- Optional local pretrained assets with Hugging Face fallback.
- YAML-based experiment configuration with reusable defaults.
- Standard outputs for metrics, predictions, resolved configs, probabilities,
  logs, and CAPE position artifacts.

## Paper

The manuscript is available at [docs/CAPE.pdf](docs/CAPE.pdf).

## Pretrained Assets

We provide organized scBERT and scGPT pretrained assets on Hugging Face for use
with this pipeline:

- scBERT: `kaichenxu/cape_scbert`
- scGPT: `kaichenxu/cape_scgpt`

Each repository includes the model weights and companion files expected by the
CAPE wrappers. The example configs reference these IDs through
`model.hf_repo_id`, and the pipeline downloads them automatically when
`model.path` does not point to an existing local asset directory.

## Repository Layout

```text
configs/              Experiment configs and shared CAPE defaults
docs/                 Paper and project figures
scripts/              Convenience scripts for application workflows
src/                  CAPE modules, model wrappers, data utilities, pipelines
tests/                Smoke and unit tests
```

## Installation

Create and activate a Python environment, then install the package in editable
mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project requires Python 3.10 or newer. GPU training is supported through
PyTorch when a CUDA-enabled installation is available.

## CTA Data Format

The provided CTA workflow expects an AnnData `.h5ad` file. At minimum, the file
must include:

- `adata.X`: expression matrix, or set `data.input_layer` to use a layer from
  `adata.layers`.
- `adata.obs[data.label_column]`: cell type labels.
- Gene identifiers in `adata.var_names`, or in `adata.var[data.gene_column]`
  when `data.gene_column` is set.

Optional fields include a batch column in `adata.obs` and a split column for
predefined train/validation/test partitions.

## CTA Configuration

Starter configs are provided for both supported backends:

- `configs/CTA/scgpt_CTA.yaml`
- `configs/CTA/scbert_CTA.yaml`

Before running, update at least the dataset path and label column:

```yaml
data:
  path: /path/to/dataset.h5ad
  label_column: celltype
  gene_column: null
  input_layer: null
```

Pretrained assets are resolved from `model.path` when that directory exists.
If it does not exist, the pipeline uses `model.hf_repo_id`, for example
`kaichenxu/cape_scgpt` or `kaichenxu/cape_scbert`.

The default configs use a stratified split:

```yaml
data:
  split:
    mode: stratified
    ratios:
      train: 0.8
      val: 0.1
      test: 0.1
```

To use a predefined split column, set `mode: column` and provide the split
column plus label values for train, validation, and test.

## Running CTA

Run scGPT CTA:

```bash
python -m src.main --config configs/CTA/scgpt_CTA.yaml
```

Run scBERT CTA:

```bash
python -m src.main --config configs/CTA/scbert_CTA.yaml
```

Equivalent convenience scripts are also available:

```bash
bash scripts/run_scgpt_CTA.sh
bash scripts/run_scbert_CTA.sh
```

Set `run.device` to `auto`, `cpu`, `cuda`, or a specific CUDA device string
supported by PyTorch.

## Outputs

For the CTA workflow, a run named `scgpt_cta_run` writes outputs under:

```text
results/CTA/scgpt/scgpt_cta_run/
```

Standard artifacts include:

- `metrics.json`: test accuracy, macro F1, and weighted F1.
- `predictions.csv`: cell IDs, predicted labels, and true labels.
- `probabilities.npy`: class probabilities when `save_probabilities` is true.
- `label_mapping.json`: label-to-ID mapping learned from the training split.
- `config_resolved.yaml`: fully merged run configuration.
- `summary.json`: compact run summary and artifact paths.
- `cape/`: selected features, token IDs, priority scores, and rank positions.

Logs are written under:

```text
logs/CTA/<model_name>/<run_name>.log
```

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover config loading, data preprocessing helpers, pretrained model
source resolution, and pipeline output smoke checks.