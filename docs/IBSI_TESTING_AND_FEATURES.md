# IBSI Testing, Reproducibility, and Feature Families

This guide describes the reproducibility assets included with Flash Radiomics.
All commands are run from the repository root.

## Included IBSI suite

`reproducibility/data/ibsi/` contains the compact IBSI-1 and IBSI-2 test suite:

- IBSI-1 digital and CT radiomics phantoms in DICOM and NIfTI formats
- IBSI-2 digital and CT radiomics phantoms in DICOM and NIfTI formats
- IBSI-1 configurations A through E
- IBSI-2 phase 1 and phase 2 configurations
- submission templates and reference feature values
- IBSI-2 reference response maps
- source licenses and dataset provenance files

The embedded `ibsi_2_reference_data` directory is copied without its nested Git
metadata. Its license and reference files are retained.

## Experiment runbooks

Each supported reproducibility experiment has its own step-by-step runbook and
an environment-neutral wrapper script:

- [IBSI-1 Flash validation](../reproducibility/experiments/ibsi1/README.md)
- [IBSI-2 Flash validation](../reproducibility/experiments/ibsi2/README.md)
- [IBSI cross-tool validation](../reproducibility/experiments/cross_tool/README.md)

Generated experiment folders are written below `reproducibility/<tool>/exp/`
and are ignored by version control.

## Feature families by class

| Python class | Feature class key | Family | Scope |
|---|---|---|---|
| `RadiomicsFirstOrder` | `firstorder` | First-order intensity statistics | Segment and voxel |
| `RadiomicsIH` | `ih` | Discretized intensity histogram | Segment |
| `RadiomicsIVH` | `ivh` | Intensity-volume histogram | Segment |
| `RadiomicsGLCM` | `glcm` | Gray-level co-occurrence matrix | Segment and voxel |
| `RadiomicsGLRLM` | `glrlm` | Gray-level run-length matrix | Segment and voxel |
| `RadiomicsGLSZM` | `glszm` | Gray-level size-zone matrix | Segment and voxel |
| `RadiomicsGLDZM` | `gldzm` | Gray-level distance-zone matrix | Segment |
| `RadiomicsGLDM` | `gldm` | Gray-level dependence matrix, corresponding to the IBSI NGLDM family | Segment and voxel |
| `RadiomicsNGTDM` | `ngtdm` | Neighbouring gray-tone difference matrix | Segment and voxel |
| `RadiomicsShape` | `shape` | Three-dimensional morphology and shape | Segment |
| `RadiomicsShape2D` | `shape2D` | Two-dimensional morphology and shape | Segment |

The repository includes
`reproducibility/ibsi/ibsi_high_consensus_165_feature_map.csv`, which maps the
default 165-feature IBSI shortlist to Flash, PyRadiomics, MIRP, and CERR names
where direct mappings exist. Feature availability can depend on segment versus
voxel mode, dimensionality, and backend.

## Selecting families

For the Original image type, the extractor defaults to `firstorder`, `glcm`,
`glrlm`, `glszm`, `gldm`, `ngtdm`, and `shape`. Enable `ih`, `ivh`, `gldzm`,
and `shape2D` explicitly. To run a specific family:

```python
from flash_radiomics import RadiomicsFeatureExtractor

extractor = RadiomicsFeatureExtractor(backend="cpu", binWidth=25)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("glcm")
result = extractor.execute("image.nii.gz", "mask.nii.gz", label=1)
```

Use `enableFeaturesByName` when only selected named features are required.

## Reproducibility dependencies

The Flash-only IBSI wrappers use the package dependencies plus `pandas` and
`openpyxl`. Cross-tool runs additionally require PyRadiomics and MIRP. CUDA
execution requires a CUDA-capable host and a CUDA-enabled build.

Install the Python-side reproducibility dependencies with:

```bash
python -m pip install -r requirements-reproducibility.txt
```

The third-party tools retain their own licensing and platform requirements.
