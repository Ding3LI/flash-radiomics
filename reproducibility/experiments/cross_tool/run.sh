#!/usr/bin/env bash
set -euo pipefail

python reproducibility/common/run_ibsi_reference_verification.py \
    --suite both \
    --tools flash,mirp,pyradiomics \
    --flash-backends cpu \
    --num-threads 1 \
    --repeats 1
