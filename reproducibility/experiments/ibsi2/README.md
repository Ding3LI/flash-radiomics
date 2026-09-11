# IBSI-2 Flash Validation

This experiment validates Flash against the included IBSI-2 CT phantom,
phase 1 and phase 2 configurations, submission template, reference values, and
reference response maps.

## 1. Install and build

From the repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements-reproducibility.txt
```

## 2. Confirm inputs

The wrapper uses:

```text
reproducibility/data/ibsi/IBSI-2-Phase2-Submission-Template.csv
reproducibility/data/ibsi/ibsi2_configs_phase1_phase2_validation.json
reproducibility/data/ibsi/data_sets/ibsi_2_ct_radiomics_phantom/
reproducibility/data/ibsi/ibsi_2_reference_data/
```

## 3. Run

Run the CPU validation:

```bash
bash reproducibility/experiments/ibsi2/run.sh cpu
```

On a compatible build, replace `cpu` with `mps` or `cuda`.

## 4. Review results

The runner writes a timestamped experiment under
`reproducibility/flash/exp/`. Review its configuration-level CSV files and
summary JSON for pass, fail, unsupported, and error counts.
