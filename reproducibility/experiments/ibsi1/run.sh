#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cpu}"

case "${backend}" in
    cpu|mps|cuda) ;;
    *)
        echo "Usage: bash reproducibility/experiments/ibsi1/run.sh [cpu|mps|cuda]" >&2
        exit 2
        ;;
esac

python reproducibility/flash/benchmark_flash_segment.py \
    --dataset-suite ibsi1 \
    --backend "${backend}" \
    --num-threads 1 \
    --segment-class-workers 1 \
    --repeats 1 \
    --skip-feature-timings
