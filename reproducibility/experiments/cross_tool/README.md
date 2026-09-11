# IBSI Cross-Tool Validation

This experiment runs the IBSI-1 and IBSI-2 scalar reference checks across
Flash, MIRP, and PyRadiomics. The included phantoms and reference files are
shared by each runner.

## 1. Install dependencies

From the repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements-reproducibility.txt
```

Confirm that MIRP and PyRadiomics can be imported before running the combined
experiment.

## 2. Run

```bash
bash reproducibility/experiments/cross_tool/run.sh
```

The wrapper uses the CPU backend, one thread, and one repeat to provide a
portable reference run.

## 3. Customize

Run only selected tools or suites with the underlying command:

```bash
python reproducibility/common/run_ibsi_reference_verification.py \
    --suite ibsi1 \
    --tools flash,pyradiomics \
    --flash-backends cpu \
    --num-threads 1 \
    --repeats 1
```

Accepted suites are `ibsi1`, `ibsi2`, and `both`. Accelerator backends require
their corresponding native libraries.

## 4. Review results

The combined runner writes a timestamped directory below
`reproducibility/common/exp/`. The top-level summary JSON records every command,
return code, log location, and tool-specific summary path.
