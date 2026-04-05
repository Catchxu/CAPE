#!/usr/bin/env bash
set -euo pipefail

python -m src.main --config configs/CTA/scbert_CTA.yaml "$@"
