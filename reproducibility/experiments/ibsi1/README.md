# IBSI-1 Flash Validation

This experiment validates Flash segment-based features against the included
IBSI-1 CT phantom, configurations A through E, and the checked-in 165-feature
high-consensus mapping.

## 1. Install and build

From the repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements-reproducibility.txt
```

## 2. Confirm inputs

The wrapper expects these included assets:

```text
reproducibility/data/ibsi/IBSI-1-submission-table.xlsx
reproducibility/data/ibsi/ibsi1_configs_A_to_E.json
reproducibility/data/ibsi/data_sets/ibsi_1_ct_radiomics_phantom/
reproducibility/ibsi/ibsi_high_consensus_165_feature_map.csv
```

## 3. Run

Run the CPU validation:

```bash
bash reproducibility/experiments/ibsi1/run.sh cpu
```

On a compatible build, replace `cpu` with `mps` or `cuda`.

## 4. Review results

The benchmark runner creates a timestamped directory below:

```text
reproducibility/flash/exp/
```

Review the generated summary JSON and validation CSV. A nonzero command exit
status indicates that setup or extraction failed; feature-level compliance is
reported in the generated summary.
