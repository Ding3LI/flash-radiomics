# Flash-Radiomics

Flash-Radiomics is a performance-oriented radiomics feature-extraction library
with a Python API and native C implementations. It supports scalar
segment-based features and spatial feature maps, accepts file or in-memory
inputs, loads YAML or JSON configurations, and persists every completed
extraction as HDF5.

CPU support is always built. Release wheels contain CPU + CUDA on Linux and
CPU + MPS on macOS. Source builds detect the available CUDA toolkit, and
`backend="auto"` selects a usable backend at runtime.

## Repository layout

```text
flash_radiomics/       Python package
src/                   C, MPS, and CUDA native implementations
configs/               Editable YAML configuration template
tutorials/             End-to-end Jupyter tutorial
test_functionality/    Self-contained 2D/3D extraction and HDF5 test
reproducibility/       IBSI data, validation scripts, and experiment runbooks
docs/                  HDF5 schema and IBSI feature/validation reference
```

Start with [`reproducibility/README.md`](reproducibility/README.md) for the
experiment layout or [`test_functionality/README.md`](test_functionality/README.md)
for the end-to-end functionality test.

## Installation

Python 3.10 through 3.14 is supported. Install a prebuilt wheel from PyPI or
build from a local source checkout. Python dependencies are installed separately.

### Install from PyPI

Install the dependencies, then the production wheel:

```bash
python -m pip install numpy h5py SimpleITK pykwalify PyWavelets scipy scikit-image scikit-learn trimesh
python -m pip install --index-url https://pypi.org/simple/ --only-binary=flash-radiomics --no-deps flash-radiomics
```

No compiler or CUDA toolkit is needed to install a wheel. Linux x86-64 and
ARM64 (SBSA) wheels target glibc 2.28 or newer and include the CPU backend,
CUDA backend, and required CUDA runtime shared libraries. GPU execution still
requires a compatible NVIDIA GPU and driver; the driver is not included.
The release build uses CUDA 12.9. GPU execution requires a driver compatible
with CUDA 12.9, including PTX JIT support for the target GPU.
macOS Intel and Apple Silicon wheels include CPU and MPS libraries, with MPS
available only on compatible hardware. Windows wheels are not currently built.

Use `backend="auto"` to select from available backends, or explicitly request
`"cpu"`, `"cuda"`, or `"mps"`. Automatic selection may use CPU for smaller
workloads or features without accelerator support. Inspect availability with:

```python
from flash_radiomics import detect_backends

print(detect_backends())
```

### Install from source

Clone or download this repository and run from its root directory:

```bash
python -m pip install -r requirements.txt
python -m pip install . --no-deps
```

For development, use an editable installation instead of the second command:

```bash
python -m pip install -e . --no-deps
```

Editable installation uses the checkout directly, so Python source edits are
available without reinstalling. Restart the notebook kernel to reload imported
modules after editing them.

Source builds require a C11 compiler, zlib development files, and a native
build tool such as Make. macOS requires Xcode Command Line Tools. The isolated
Python build installs CMake automatically. CPU is always compiled; macOS also
builds MPS. The default `FLASH_RADIOMICS_ENABLE_CUDA=auto` detects `nvcc` on
PATH, so CUDA additionally requires a compatible CUDA toolkit. GPU hardware
is detected at runtime, not required for compilation. A driver alone does not
provide the CUDA compiler. The default GPU targets require CUDA 11.8 or newer
in the CUDA 11/12 series; CUDA 13 removes some of these targets.

To require CUDA or explicitly disable it for a source installation:

```bash
FLASH_RADIOMICS_ENABLE_CUDA=on python -m pip install . --no-deps
FLASH_RADIOMICS_ENABLE_CUDA=off python -m pip install . --no-deps
```

Choose one command; add `-e` before `.` for editable development. Re-run the
source installation after changing native C, CUDA, or MPS code, and restart
the Python process to load rebuilt libraries.
Build-time flags do not alter an already installed wheel.

`bash COMPILE.sh` is useful for native-only development: it requires CMake
already installed and builds libraries into `build/`. It does not install or
register the Python package. Prefer the editable installation for Python and
notebook development; package-local libraries can take precedence over a
separate `build/` directory.

For the cross-tool reproducibility experiments, install the additional Python
dependencies:

```bash
python -m pip install -r requirements-reproducibility.txt
```

Repository-only tests, tutorials, and IBSI assets are excluded from installable
distributions.

## Quick start

The examples use local image and mask files. Set the input paths to your data
and use an integer label present in the mask.

```python
from flash_radiomics import RadiomicsFeatureExtractor

extractor = RadiomicsFeatureExtractor(
    backend="cpu",
    binWidth=25,
    distances=[1],
)

features = extractor.execute(
    "images/case_001.nii.gz",
    "masks/case_001.nii.gz",
    label=1,
    output_hdf5_path="outputs/case_001.h5",
    case_id="case_001",
    roi_id="tumor",
    roi_name="Primary tumor",
)

print(f"Computed {len(features)} outputs")
print(f"Saved to {extractor.last_hdf5_path}")
```

`execute` returns an ordered feature dictionary and also writes the extraction
to HDF5. The image and mask arguments may be file paths or `SimpleITK.Image`
objects.

## Input formats

Flash-Radiomics uses SimpleITK to load inputs. Pass each input as either a
string path or an already loaded `SimpleITK.Image`. Common supported formats
include:

| Format | Common extensions | Image | Mask |
|---|---|---:|---:|
| NIfTI | `.nii`, `.nii.gz` | Yes | Yes |
| NRRD | `.nrrd`, `.nhdr` | Yes | Yes |
| MetaImage | `.mha`, `.mhd` | Yes | Yes |
| Analyze/NIfTI pair | `.hdr`, `.img`, `.img.gz` | Yes | Yes |
| GIPL | `.gipl`, `.gipl.gz` | Yes | Yes |
| MINC | `.mnc` | Yes | Yes |
| MRC | `.mrc`, `.rec` | Yes | Yes |
| VTK image | `.vtk` | Yes | Yes |
| Single-frame DICOM | `.dcm` or no extension | Yes | Yes, when it is a raster label map |
| DICOM image series | one directory per series | Yes | No |

The precise list is determined by the ImageIO modules registered in the
installed SimpleITK build. Other SimpleITK-readable formats, including TIFF,
PNG, JPEG, and BMP, can be passed as single-file 2D inputs, although
medical-volume formats that preserve geometry and voxel values are recommended
for quantitative radiomics. Inspect the local readers with:

```python
import SimpleITK as sitk

print(sitk.ImageFileReader().GetRegisteredImageIOs())
```

For a volumetric DICOM image, provide a directory containing exactly one image
series. Flash-Radiomics sorts and combines the slices through SimpleITK/GDCM:

```python
from flash_radiomics import RadiomicsFeatureExtractor

extractor = RadiomicsFeatureExtractor(backend="cpu")
features = extractor.execute(
    "dicom/case_001/ct",
    "masks/case_001.nii.gz",
    label=1,
)
```

The mask must be a raster label map supplied as one readable file or as a
`SimpleITK.Image`. DICOM RTSTRUCT contours and DICOM SEG objects are not
interpreted directly; convert or rasterize them into the image geometry first,
preferably as NIfTI, NRRD, or MetaImage. The mask must contain the requested
integer label and match the image's dimension and physical geometry (size,
spacing, origin, and direction).

Color or other multi-component images are not accepted directly as intensity
images. Select one scalar component with SimpleITK and pass the resulting
`SimpleITK.Image`. Vector masks are supported through `label_channel`.

See the official [SimpleITK ImageIO documentation](https://simpleitk.readthedocs.io/en/release/IO.html)
for the underlying format behavior.

## YAML and JSON configuration

Pass a YAML configuration as the first extractor argument. JSON
files and Python dictionaries use the same top-level structure.

```yaml
setting:
  label: 1
  binWidth: 25.0
  distances: [1]
  normalize: false
  additionalInfo: true
  hdf5Compression: lzf
  hdf5OutputDirectory: outputs

voxelSetting:
  kernelRadius: 1
  maskedKernel: true
  initValue: 0.0
  voxelBatch: -1

imageType:
  Original: {}

scalarFeatureClass:
  firstorder: [Mean, Variance]
  glcm: [Contrast]

spatialFeatureClass:
  firstorder: [Mean]
  glcm: [Contrast]
```

Save this example as `configs/extraction.yaml`, then load it with a string or
`pathlib.Path`:

```python
from pathlib import Path
from flash_radiomics import RadiomicsFeatureExtractor

extractor = RadiomicsFeatureExtractor(
    Path("configs/extraction.yaml"),
    backend="cpu",
    num_threads=1,
)
result = extractor.execute("image.nii.gz", "mask.nii.gz")
```

Constructor keyword settings override values from the configuration file.
`backend` and `num_threads` are runtime constructor arguments rather than YAML
schema settings.

The mode-specific sections also control which extraction modes run:

- a non-null `scalarFeatureClass` enables scalar extraction;
- a non-null `spatialFeatureClass` enables spatial extraction;
- setting a section to `null`, or omitting it while the other mode-specific
  section is present, disables that mode;
- when both sections are enabled, `execute()` automatically extracts both modes
  and stores them in one HDF5 file.

An explicit `spatialMode=False` or `spatialMode=True` selects just one enabled
mode. It is unnecessary for the usual configuration-driven workflow.

Feature selection uses the following forms:

- `scalarFeatureClass: null` or `spatialFeatureClass: null` disables that mode;
- `scalarFeatureClass: {}` enables every supported scalar feature class and
  feature;
- `spatialFeatureClass: {}` enables every supported spatial feature class and
  feature;
- an empty list after a class name, such as `firstorder: []`, enables every
  non-deprecated feature in that class;
- a list of feature names enables only those features in that class.

For example, extract every scalar feature and disable spatial extraction:

```yaml
scalarFeatureClass: {}
spatialFeatureClass: null
```

Extract every spatial feature and disable scalar extraction:

```yaml
scalarFeatureClass: null
spatialFeatureClass: {}
```

Extract every available scalar and spatial feature:

```yaml
scalarFeatureClass: {}
spatialFeatureClass: {}
```

Alternatively, select specific classes and enable every feature within those
classes by assigning an empty list:

```yaml
scalarFeatureClass:
  firstorder: []
  glcm: []

spatialFeatureClass:
  firstorder: []
  glcm: []
```

Separate maps can also select individual features for each mode:

```yaml
scalarFeatureClass:
  firstorder: [Mean, Variance]
  glcm: [Contrast]

spatialFeatureClass:
  firstorder: [Mean]
  glcm: [Contrast]
```

Copy and edit [`configs/extraction_template.yaml`](configs/extraction_template.yaml)
when preparing a custom extraction. It includes every global and voxel setting
with its default value, plus separate partial scalar and spatial selections.
The legacy `featureClass` key remains supported and applies one selection to
both modes, but it cannot be combined with the two mode-specific keys.

## Feature selection from Python

```python
from flash_radiomics import RadiomicsFeatureExtractor

extractor = RadiomicsFeatureExtractor(backend="cpu", binWidth=25)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("glcm")
extractor.enableFeaturesByName(firstorder=["Mean", "Variance"])
result = extractor.execute("image.nii.gz", "mask.nii.gz", label=1)
```

For a complete 3D extraction without listing individual families:

```python
scalar_extractor = RadiomicsFeatureExtractor(backend="cpu")
scalar_extractor.enableAllScalarFeatures()

spatial_extractor = RadiomicsFeatureExtractor(backend="cpu")
spatial_extractor.enableAllSpatialFeatures()
```

These methods enable all supported features for the selected extraction mode.
Use a YAML configuration to select a subset.

To calculate both modes and store them in one HDF5 file, use the combined
helper. With no configuration, it enables every supported family for each
mode:

```python
from flash_radiomics import extract_scalar_and_spatial

result = extract_scalar_and_spatial(
    "image.nii.gz",
    "mask.nii.gz",
    output_hdf5_path="outputs/case_001.h5",
    backend="cpu",
)
```

For partial scalar and spatial extraction, pass the one template containing
both mode-specific selections:

```python
result = extract_scalar_and_spatial(
    "image.nii.gz",
    "mask.nii.gz",
    config="configs/extraction_template.yaml",
    output_hdf5_path="outputs/case_001_partial.flashrad.h5",
    backend="cpu",
)
```

Use `scalar_config=` and `spatial_config=` when the modes must come from two
different files. All variants still produce one HDF5 file.

Available scalar families are:

- `firstorder`: first-order intensity statistics;
- `ih`: discretized intensity histogram;
- `ivh`: intensity-volume histogram;
- `glcm`: gray-level co-occurrence matrix;
- `glrlm`: gray-level run-length matrix;
- `glszm`: gray-level size-zone matrix;
- `gldzm`: gray-level distance-zone matrix;
- `gldm`: gray-level dependence matrix, corresponding to IBSI NGLDM;
- `ngtdm`: neighbouring gray-tone difference matrix;
- `shape`: 3D morphology;
- `shape2D`: 2D morphology.

See [`docs/IBSI_TESTING_AND_FEATURES.md`](docs/IBSI_TESTING_AND_FEATURES.md)
for the class mapping and supported scalar/spatial scope.

## Spatial feature maps

Set `spatialMode=True` to explicitly select spatial feature maps. If the YAML
contains only a non-null `spatialFeatureClass`, `execute()` selects spatial mode
automatically. Spatial outputs are
returned as `SimpleITK.Image` objects and stored as the channel-first
`feature_maps` dataset paired with `feature_names` in HDF5.

```python
from flash_radiomics import RadiomicsFeatureExtractor

extractor = RadiomicsFeatureExtractor(
    "test_functionality/configs/spatial.yaml",
    backend="cpu",
)
maps = extractor.execute(
    "image.nii.gz",
    "mask.nii.gz",
    label=1,
    spatialMode=True,
    output_hdf5_path="outputs/case_001.flashrad.h5",
)
```

The supported spatial families are `firstorder`, `glcm`, `glrlm`, `glszm`,
`gldm`, and `ngtdm`.

## HDF5 output

Each completed extraction writes to one HDF5 container. Choose its location in one of
three ways, in descending precedence:

1. pass `output_hdf5_path` to `execute`;
2. set `hdf5OutputPath` or `hdf5OutputDirectory` in the configuration;
3. use the default `flash_radiomics_hdf5/` directory.

`hdf5OutputPath` identifies one exact case-level container. Scalar and spatial
writes that use the same path, case identifier, and ROI identifier are merged
atomically. Re-running one mode updates that mode while preserving the other.

For batch processing, `hdf5OutputDirectory` creates one deterministic
`<case_id>_<roi_id>.flashrad.h5` filename shared by both modes.

The principal hierarchy is:

```text
/rois/<roi_id>/
  image
  mask
  metadata/extractions/scalar/
    settings_json
    enabled_image_types_json
    enabled_features_json
  metadata/extractions/spatial/
    settings_json
    enabled_image_types_json
    enabled_features_json
  scalar_features/names
  scalar_features/values
  feature_names
  feature_maps
```

`feature_maps` is a channel-first tensor aligned
with `feature_names`; partial and full selections therefore differ only in
channel count. A combined container also includes `scalar_features`. The root
`mode` attribute is `scalar`, `spatial`, or `combined`; `modes_json` records the
included modes. Mode-specific configuration metadata is stored under
`metadata/extractions`. Image geometry, backend, compression, case identifier,
and ROI identifier are stored as attributes.

See [`docs/HDF5_SCHEMA.md`](docs/HDF5_SCHEMA.md) for the complete dataset and
merge contract.

## Backend selection

Supported backend names are `cpu`, `mps`, `cuda`, and `auto`.

```python
from flash_radiomics import detect_backends

print(detect_backends())
```

`auto` selects a usable CUDA, MPS, or CPU library, in that order. Individual
features may use CPU paths. Use `detect_backends()` to check device availability
before explicitly requesting CUDA or MPS; a missing accelerator library raises
an error.

## Preprocessing settings

Common configuration settings include:

- `normalize`, `normalizeScale`, and `removeOutliers`;
- `resampledPixelSpacing` and `interpolator`;
- `preCrop` and `padDistance`;
- `resegmentRange`, `resegmentMode`, and `resegmentShape`;
- `binWidth` or `binCount`;
- `force2D` and `force2Ddimension`;
- `geometryTolerance`, `label`, and `label_channel`.

The complete accepted schema is available at
`flash_radiomics/schemas/paramSchema.yaml`.

## Functionality test

From the repository root, run the 2D/3D suite after installation:

```bash
bash test_functionality/run.sh
```

The test uses the small IBSI digital phantom and runs full scalar, partial
scalar, full spatial, partial spatial, full combined, and partial combined
extraction. Each available backend runs in a separate process and writes its
own result for each mode. The suite also covers 2D extraction, validates returned
values against HDF5 contents, and independently checks the first-order mean.

Explicit validation commands are documented in
[`test_functionality/README.md`](test_functionality/README.md).

## Tutorial

The notebook [`tutorials/flash_radiomics_tutorial.ipynb`](tutorials/flash_radiomics_tutorial.ipynb)
uses the bundled 3D IBSI CT radiomics phantom. It demonstrates installed-package
imports, YAML overrides, full combined extraction, and direct
inspection of the fixed HDF5 schema.

## IBSI and cross-tool reproduction

The compact IBSI data and reference assets are included under
`reproducibility/data/ibsi/`. Use the runbooks in
[`reproducibility/README.md`](reproducibility/README.md) for IBSI-1, IBSI-2,
and Flash-Radiomics/MIRP/PyRadiomics comparisons.

## Manual native build

Build CPU and optional macOS MPS support:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build --parallel
```

Enable CUDA detection on a CUDA-capable host:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build --parallel
```

Native CPU instruction tuning is disabled by default for portability. Enable
it explicitly with `-DFLASH_RADIOMICS_NATIVE_CPU=ON` only for a controlled
deployment target.

## Citation

If you use Flash-Radiomics in your research, please cite the software:

```bibtex
@software{ding_flash_radiomics_2026,
  author  = {
    Ding, Shanli and
    Hu, Yiyi and
    Fu, Ziyu and
    Lin, Chia-Hsin and
    Luo, Ruihan and
    Chun, Jaehee and
    Zhang, Xinyue and
    Mawlawi, Osama
  },
  title   = {Flash-Radiomics: A Scalable Hybrid CPU–CUDA Engine for Standardized Scalar Radiomics and Accelerated Spatial Mapping},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/Ding3LI/flash-radiomics}
}
```

A manuscript describing Flash-Radiomics is currently under review, and a preprint is in preparation. The corresponding publication citation will be added here once it becomes publicly available.

## License

Flash-Radiomics source code is distributed under the BSD 3-Clause License. See
[`LICENSE`](LICENSE) for the complete terms. Bundled third-party datasets and
reference materials remain subject to the licenses included with those assets.
