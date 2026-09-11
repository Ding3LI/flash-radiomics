# Functionality Test

This suite exercises the public Flash Radiomics extraction workflow with the
small bundled IBSI digital phantom. It derives a two-dimensional phantom slice
from the three-dimensional input and checks both dimensionalities.

The suite covers:

- file-path and in-memory image inputs;
- YAML configuration loading;
- all scalar feature families supported for 2D and 3D inputs;
- all supported spatial feature-map families;
- scalar-only, spatial-only, and combined extraction with partial selections;
- scalar-only, spatial-only, and combined extraction with full selections;
- explicit HDF5 output and HDF5-to-returned-result consistency;
- one-container scalar-plus-spatial HDF5 merging;
- independent first-order mean verification;
- separate execution for every available CPU, MPS, or CUDA backend.

## 1. Install and build

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Source builds detect the CUDA compiler automatically. Set
`FLASH_RADIOMICS_ENABLE_CUDA=on` to require CUDA or `off` to disable it.

## 2. Run the suite

```bash
bash test_functionality/run.sh
```

The launcher queries the installed native libraries, then starts one independent
Python process for every backend-and-mode pair. CPU always runs. MPS and CUDA
are included only when the corresponding library and device are available;
their absence is not a failure.

List the backends that can be tested:

```bash
python test_functionality/run_functionality_tests.py --list-available-backends
```

Run one backend explicitly:

```bash
python test_functionality/run_functionality_tests.py \
    --backend cpu \
    --mode scalar-partial \
    --output test_functionality/output/cpu_scalar_partial_results.h5 \
    --overwrite

python test_functionality/run_functionality_tests.py \
    --backend cuda \
    --mode combined-full \
    --output test_functionality/output/cuda_combined_full_results.h5 \
    --overwrite
```

An explicitly requested unavailable backend stops before extraction with a
clear error. The accepted modes are `scalar-partial`, `scalar-full`,
`spatial-partial`, `spatial-full`, `combined-partial`, and `combined-full`.
Passing `--mode all` runs all modes in one process for ad hoc use.

## 3. Review the output

The launcher writes one result file per backend-and-mode pair, for example:

```text
test_functionality/output/cpu_scalar_partial_results.h5
test_functionality/output/mps_spatial_full_results.h5
test_functionality/output/cuda_combined_full_results.h5
```

Full scalar and full spatial files each contain validated 3D and 2D runs under
`/runs/<backend>`. Each partial or combined file contains its corresponding 3D
run. All partial tests use the single `configs/extraction_template.yaml`
selection. Temporary native HDF5 files are deleted after being validated and
copied into the result file.

The full single-mode configurations rely on their non-null mode section to
select scalar or spatial extraction automatically. The partial single-mode
tests explicitly select one section with `spatialMode`; the combined partial
test omits the flag and verifies that both enabled YAML sections run.

## 4. Supply custom inputs or configurations

Run from the repository root. This example uses the included CT phantom and
test configurations; change the input paths to test another case:

```bash
python test_functionality/run_functionality_tests.py \
    --image-3d reproducibility/data/ibsi/data_sets/ibsi_1_ct_radiomics_phantom/nifti/image/phantom.nii.gz \
    --mask-3d reproducibility/data/ibsi/data_sets/ibsi_1_ct_radiomics_phantom/nifti/mask/mask.nii.gz \
    --scalar-3d-config test_functionality/configs/scalar_3d.yaml \
    --spatial-config test_functionality/configs/spatial.yaml \
    --backend cpu \
    --output test_functionality/output/custom_results.h5 \
    --overwrite
```

Use `--image-2d` and `--mask-2d` together to test custom 2D files. When they
are omitted, the suite derives a valid 2D slice from the supplied 3D pair.
