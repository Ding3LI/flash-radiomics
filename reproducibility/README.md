# Reproducibility Guide

This directory contains the compact data, scripts, configurations, and
runbooks needed to reproduce the supported IBSI and cross-tool checks. Run all
commands from the repository root.

## Start here

| Goal | Runbook | Command |
|---|---|---|
| Validate Flash against IBSI-1 | [`experiments/ibsi1/README.md`](experiments/ibsi1/README.md) | `bash reproducibility/experiments/ibsi1/run.sh cpu` |
| Validate Flash against IBSI-2 | [`experiments/ibsi2/README.md`](experiments/ibsi2/README.md) | `bash reproducibility/experiments/ibsi2/run.sh cpu` |
| Compare Flash, MIRP, and PyRadiomics | [`experiments/cross_tool/README.md`](experiments/cross_tool/README.md) | `bash reproducibility/experiments/cross_tool/run.sh` |

## Directory map

```text
common/         Shared IBSI validation and spatial benchmark helpers
data/ibsi/      Bundled IBSI phantoms, configurations, references, and licenses
experiments/    Stable wrappers and step-by-step runbooks
flash/          Flash IBSI and spatial benchmark runners
ibsi/           Curated 165-feature cross-tool name map
mirp/           MIRP IBSI-1 runner used by the cross-tool experiment
pyradiomics/    PyRadiomics IBSI-1 and spatial benchmark runners
```

## Generated outputs

Experiment outputs are written to timestamped `exp/` directories below the
corresponding tool folder. These directories, along with benchmark logs and
result caches, are ignored by version control.

## Dependencies

Install Flash and the optional comparison tools with:

```bash
python -m pip install -e .
python -m pip install -r requirements-reproducibility.txt
```

The Flash-only functionality suite is documented separately in
[`../test_functionality/README.md`](../test_functionality/README.md).
