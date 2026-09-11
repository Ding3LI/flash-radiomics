#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cpu}"

case "${backend}" in
    cpu|mps|cuda) ;;
    *)
        echo "Usage: bash reproducibility/experiments/ibsi2/run.sh [cpu|mps|cuda]" >&2
        exit 2
        ;;
esac

python reproducibility/flash/benchmark_flash_ibsi2.py \
    --backend "${backend}" \
    --num-threads 1
