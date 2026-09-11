#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build --parallel
