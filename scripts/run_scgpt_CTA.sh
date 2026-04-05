#!/usr/bin/env bash
set -euo pipefail

python -m src.main --config configs/CTA/scgpt_CTA.yaml "$@"
