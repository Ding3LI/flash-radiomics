#!/usr/bin/env bash
set -euo pipefail

MODES=(
    scalar-partial
    scalar-full
    spatial-partial
    spatial-full
    combined-partial
    combined-full
)

for backend in $(python test_functionality/run_functionality_tests.py --list-available-backends); do
    for mode in "${MODES[@]}"; do
        output_mode="${mode//-/_}"
        python test_functionality/run_functionality_tests.py \
            --backend "${backend}" \
            --mode "${mode}" \
            --output "test_functionality/output/${backend}_${output_mode}_results.h5" \
            --overwrite
    done
done
