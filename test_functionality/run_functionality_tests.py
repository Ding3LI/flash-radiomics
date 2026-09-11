#!/usr/bin/env python3
"""Exercise Flash Radiomics scalar, spatial, YAML, and HDF5 workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import h5py
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_radiomics import (
    RadiomicsFeatureExtractor,
    detect_backends,
    extract_scalar_and_spatial,
)


IBSI_DIGITAL_ROOT = (
    PROJECT_ROOT
    / "reproducibility"
    / "data"
    / "ibsi"
    / "data_sets"
    / "ibsi_1_digital_phantom"
    / "nifti"
)
DEFAULT_IMAGE_3D = IBSI_DIGITAL_ROOT / "image" / "phantom.nii.gz"
DEFAULT_MASK_3D = IBSI_DIGITAL_ROOT / "mask" / "mask.nii.gz"
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "test_functionality" / "configs"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "test_functionality" / "output" / "functionality_results.h5"
)

SCALAR_3D_FAMILIES = (
    "firstorder",
    "ih",
    "ivh",
    "glcm",
    "glrlm",
    "glszm",
    "gldzm",
    "gldm",
    "ngtdm",
    "shape",
)
SCALAR_2D_FAMILIES = (
    "firstorder",
    "ih",
    "ivh",
    "glcm",
    "glrlm",
    "glszm",
    "gldzm",
    "gldm",
    "ngtdm",
    "shape2D",
)
SPATIAL_FAMILIES = (
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)
SCALAR_PARTIAL_FEATURE_NAMES = {
    "original_firstorder_Mean",
    "original_firstorder_Variance",
    "original_glcm_Contrast",
}
SPATIAL_PARTIAL_FEATURE_NAMES = {
    "original_firstorder_Mean",
    "original_glcm_Contrast",
}
TEST_MODES = (
    "scalar-partial",
    "scalar-full",
    "spatial-partial",
    "spatial-full",
    "combined-partial",
    "combined-full",
)


@dataclass
class RunResult:
    backend: str
    test_name: str
    mode: str
    dimension: int
    native_hdf5: Path
    feature_vector: dict[str, Any]
    required_families: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate scalar and spatial extraction with YAML configurations and "
            "write one consolidated HDF5 result."
        )
    )
    parser.add_argument("--image-3d", type=Path, default=DEFAULT_IMAGE_3D)
    parser.add_argument("--mask-3d", type=Path, default=DEFAULT_MASK_3D)
    parser.add_argument("--image-2d", type=Path)
    parser.add_argument("--mask-2d", type=Path)
    parser.add_argument(
        "--scalar-3d-config",
        type=Path,
        default=DEFAULT_CONFIG_ROOT / "scalar_3d.yaml",
    )
    parser.add_argument(
        "--scalar-2d-config",
        type=Path,
        default=DEFAULT_CONFIG_ROOT / "scalar_2d.yaml",
    )
    parser.add_argument(
        "--spatial-config",
        type=Path,
        default=DEFAULT_CONFIG_ROOT / "spatial.yaml",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--backend",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="Run the complete suite for exactly one backend.",
    )
    parser.add_argument(
        "--mode",
        choices=("all", *TEST_MODES),
        default="all",
        help="Run one extraction mode independently, or all modes.",
    )
    parser.add_argument("--list-available-backends", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _load_pair(image_path: Path, mask_path: Path) -> tuple[sitk.Image, sitk.Image]:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if image.GetDimension() != mask.GetDimension():
        raise ValueError("Image and mask dimensions differ")
    if image.GetSize() != mask.GetSize():
        raise ValueError("Image and mask sizes differ")
    if 1 not in np.unique(sitk.GetArrayFromImage(mask)):
        raise ValueError("Mask does not contain label 1")
    return image, mask


def _derive_2d_pair(
    image_3d: sitk.Image,
    mask_3d: sitk.Image,
) -> tuple[sitk.Image, sitk.Image]:
    image_array = sitk.GetArrayFromImage(image_3d)
    mask_array = sitk.GetArrayFromImage(mask_3d)
    candidate_indices = [
        index
        for index in range(mask_array.shape[0])
        if np.any(mask_array[index] == 1) and np.any(mask_array[index] != 1)
    ]
    if not candidate_indices:
        raise ValueError("No 2D slice contains both ROI and background voxels")
    index = candidate_indices[0]

    image_2d = sitk.GetImageFromArray(image_array[index])
    mask_2d = sitk.GetImageFromArray(mask_array[index].astype(np.uint8))
    spacing = image_3d.GetSpacing()
    origin = image_3d.GetOrigin()
    image_2d.SetSpacing(spacing[:2])
    mask_2d.SetSpacing(spacing[:2])
    image_2d.SetOrigin(origin[:2])
    mask_2d.SetOrigin(origin[:2])
    return image_2d, mask_2d


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def _assert_family_coverage(
    feature_vector: dict[str, Any],
    required_families: tuple[str, ...],
) -> None:
    missing = [
        family
        for family in required_families
        if not any(name.startswith(f"original_{family}_") for name in feature_vector)
    ]
    if missing:
        raise AssertionError(f"Missing feature-family outputs: {', '.join(missing)}")


def _assert_independent_mean(
    feature_vector: dict[str, Any],
    image: sitk.Image,
    mask: sitk.Image,
) -> None:
    key = "original_firstorder_Mean"
    if key not in feature_vector:
        raise AssertionError(f"Required feature is absent: {key}")
    image_array = sitk.GetArrayFromImage(image).astype(np.float64)
    mask_array = sitk.GetArrayFromImage(mask)
    expected = float(np.mean(image_array[mask_array == 1]))
    actual = float(feature_vector[key])
    if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"First-order mean mismatch: expected {expected}, received {actual}"
        )


def _validate_native_hdf5(
    path: Path,
    *,
    mode: str,
    backend: str,
    case_id: str,
    roi_id: str,
    feature_vector: dict[str, Any],
    expected_bin_width: float,
) -> None:
    if not path.is_file():
        raise AssertionError(f"Extraction did not create HDF5 output: {path}")

    with h5py.File(path, "r") as handle:
        if handle.attrs["format_name"] != "Flash-Radiomics HDF5":
            raise AssertionError("Unexpected HDF5 format name")
        if handle.attrs["mode"] != mode:
            raise AssertionError(f"Expected HDF5 mode {mode}")
        if handle.attrs["backend"] != backend:
            raise AssertionError(f"Expected HDF5 backend {backend}")

        if handle.attrs["schema_name"] != "flashrad_hdf5":
            raise AssertionError("Unexpected HDF5 schema name")
        if handle.attrs["case_id"] != case_id:
            raise AssertionError("Unexpected HDF5 case identifier")
        roi_group = handle[f"rois/{roi_id}"]
        settings = json.loads(
            roi_group[f"metadata/extractions/{mode}/settings_json"][()].decode("utf-8")
        )
        if float(settings["binWidth"]) != expected_bin_width:
            raise AssertionError("YAML binWidth setting was not persisted")
        if "image" not in roi_group or "mask" not in roi_group:
            raise AssertionError("HDF5 input image or feature mask is missing")

        if mode == "scalar":
            scalar_group = roi_group["scalar_features"]
            names = _decode_strings(scalar_group["names"][...])
            values = scalar_group["values"][...]
            expected = {
                name: float(value)
                for name, value in feature_vector.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and not str(name).startswith("diagnostics_")
            }
            actual = dict(zip(names, values, strict=True))
            if set(actual) != set(expected):
                raise AssertionError("HDF5 scalar feature names differ from returned results")
            for name, expected_value in expected.items():
                if not np.isclose(
                    actual[name], expected_value, rtol=0.0, atol=0.0, equal_nan=True
                ):
                    raise AssertionError(f"HDF5 scalar value differs for {name}")
        else:
            names = _decode_strings(roi_group["feature_names"][...])
            expected_maps = {
                name: value
                for name, value in feature_vector.items()
                if isinstance(value, sitk.Image)
                and not str(name).startswith("diagnostics_")
            }
            if set(names) != set(expected_maps):
                raise AssertionError("HDF5 spatial feature names differ from returned results")
            stored_maps = dict(zip(names, roi_group["feature_maps"][...], strict=True))
            for name, feature_map in expected_maps.items():
                expected_array = sitk.GetArrayFromImage(feature_map).astype(np.float32)
                np.testing.assert_allclose(
                    stored_maps[name], expected_array, rtol=0.0, atol=0.0, equal_nan=True
                )


def _run_extraction(
    *,
    backend: str,
    test_name: str,
    mode: str,
    dimension: int,
    config: Path,
    image_input: str | sitk.Image,
    mask_input: str | sitk.Image,
    reference_image: sitk.Image,
    reference_mask: sitk.Image,
    required_families: tuple[str, ...],
    work_dir: Path,
    expected_feature_names: set[str] | None = None,
    expected_bin_width: float = 1.0,
) -> RunResult:
    extractor = RadiomicsFeatureExtractor(config, backend=backend, num_threads=1)
    native_hdf5 = work_dir / f"{backend}_{test_name}.h5"
    case_id = f"ibsi_{test_name}"
    roi_id = "label_1"
    execute_options = {
        "label": 1,
        "output_hdf5_path": native_hdf5,
        "case_id": case_id,
        "roi_id": roi_id,
        "roi_name": "IBSI digital phantom",
    }
    if config.name == "extraction_template.yaml":
        execute_options["spatialMode"] = mode == "spatial"
    feature_vector = extractor.execute(image_input, mask_input, **execute_options)
    if extractor.last_hdf5_path != native_hdf5:
        raise AssertionError("last_hdf5_path does not match the requested output")

    _assert_family_coverage(feature_vector, required_families)
    if expected_feature_names is not None:
        if mode == "scalar":
            actual_names = {
                name for name in feature_vector if name.startswith("original_")
            }
        else:
            actual_names = {
                name
                for name, value in feature_vector.items()
                if isinstance(value, sitk.Image)
            }
        if actual_names != expected_feature_names:
            raise AssertionError(
                f"Unexpected {mode} feature selection: {sorted(actual_names)}"
            )
    if mode == "scalar":
        _assert_independent_mean(feature_vector, reference_image, reference_mask)
    _validate_native_hdf5(
        native_hdf5,
        mode=mode,
        backend=backend,
        case_id=case_id,
        roi_id=roi_id,
        feature_vector=feature_vector,
        expected_bin_width=expected_bin_width,
    )
    return RunResult(
        backend=backend,
        test_name=test_name,
        mode=mode,
        dimension=dimension,
        native_hdf5=native_hdf5,
        feature_vector=dict(feature_vector),
        required_families=required_families,
    )


def _validate_combined_hdf5(
    *,
    backend: str,
    image_path: Path,
    mask_path: Path,
    work_dir: Path,
    test_name: str,
    config: Path | None,
) -> RunResult:
    output_path = work_dir / f"{backend}_{test_name}.flashrad.h5"
    options = {
        "output_hdf5_path": output_path,
        "backend": backend,
        "num_threads": 1,
        "label": 1,
        "case_id": f"ibsi_{test_name}",
        "roi_id": "label_1",
        "roi_name": "IBSI digital phantom",
    }
    if config is not None:
        options["config"] = config
    if config is None:
        result = extract_scalar_and_spatial(str(image_path), str(mask_path), **options)
    else:
        extractor = RadiomicsFeatureExtractor(
            config,
            backend=backend,
            num_threads=1,
        )
        result = extractor.execute(
            str(image_path),
            str(mask_path),
            label=1,
            output_hdf5_path=output_path,
            case_id=f"ibsi_{test_name}",
            roi_id="label_1",
            roi_name="IBSI digital phantom",
        )
    if result["hdf5_path"] != output_path:
        raise AssertionError("Combined extraction returned an unexpected HDF5 path")
    if config is None:
        _assert_family_coverage(result["scalar"], SCALAR_3D_FAMILIES)
        _assert_family_coverage(result["spatial"], SPATIAL_FAMILIES)
    else:
        scalar_names = {
            name for name in result["scalar"] if name.startswith("original_")
        }
        spatial_names = {
            name
            for name, value in result["spatial"].items()
            if isinstance(value, sitk.Image)
        }
        if (scalar_names != SCALAR_PARTIAL_FEATURE_NAMES
                or spatial_names != SPATIAL_PARTIAL_FEATURE_NAMES):
            raise AssertionError("Shared partial configuration returned unexpected features")

    with h5py.File(output_path, "r") as handle:
        if handle.attrs["mode"] != "combined":
            raise AssertionError("Combined HDF5 output is not marked as combined")
        modes = set(json.loads(str(handle.attrs["modes_json"])))
        if modes != {"scalar", "spatial"}:
            raise AssertionError(f"Unexpected combined HDF5 modes: {sorted(modes)}")
        roi_group = handle["rois/label_1"]
        if "scalar_features" not in roi_group or "feature_maps" not in roi_group:
            raise AssertionError("Combined HDF5 output does not contain both modes")
        if roi_group["feature_maps"].shape[1:] != roi_group["image"].shape[1:]:
            raise AssertionError("Spatial maps and image are not aligned")
        if roi_group["feature_maps"].shape[1:] != roi_group["mask"].shape[1:]:
            raise AssertionError("Spatial maps and mask are not aligned")
        for mode in ("scalar", "spatial"):
            if f"metadata/extractions/{mode}" not in roi_group:
                raise AssertionError(f"Combined HDF5 metadata is missing {mode}")

    return RunResult(
        backend=backend,
        test_name=test_name,
        mode="combined",
        dimension=3,
        native_hdf5=output_path,
        feature_vector={"scalar": result["scalar"], "spatial": result["spatial"]},
        required_families=(),
    )


def _write_consolidated_hdf5(
    output_path: Path,
    results: list[RunResult],
    availability: dict[str, bool],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    try:
        with h5py.File(temporary_path, "w") as destination:
            destination.attrs["suite_name"] = "Flash Radiomics functionality"
            destination.attrs["suite_version"] = "1.0"
            destination.attrs["status"] = "pass"
            destination.attrs["backend_availability_json"] = json.dumps(
                availability, sort_keys=True
            )
            destination.attrs["executed_backends_json"] = json.dumps(
                sorted({item.backend for item in results})
            )
            destination.attrs["run_count"] = len(results)

            runs_group = destination.require_group("runs")
            for item in results:
                run_group = runs_group.require_group(
                    f"{item.backend}/{item.test_name}"
                )
                run_group.attrs["status"] = "pass"
                run_group.attrs["mode"] = item.mode
                run_group.attrs["dimension"] = item.dimension
                run_group.attrs["required_families_json"] = json.dumps(
                    item.required_families
                )
                native_group = run_group.require_group("native_output")
                with h5py.File(item.native_hdf5, "r") as source:
                    for name, value in source.attrs.items():
                        native_group.attrs[name] = value
                    for name in source:
                        source.copy(name, native_group)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    availability = {key: bool(value) for key, value in detect_backends().items()}
    if args.list_available_backends:
        for backend in ("cpu", "mps", "cuda"):
            if availability.get(backend, False):
                print(backend)
        return

    if not availability.get(args.backend, False):
        raise RuntimeError(f"Requested backend is unavailable: {args.backend}")

    image_3d_path = _require_file(args.image_3d, "3D image")
    mask_3d_path = _require_file(args.mask_3d, "3D mask")
    scalar_3d_config = _require_file(args.scalar_3d_config, "3D scalar config")
    scalar_2d_config = _require_file(args.scalar_2d_config, "2D scalar config")
    spatial_config = _require_file(args.spatial_config, "spatial config")

    if (args.image_2d is None) != (args.mask_2d is None):
        raise ValueError("--image-2d and --mask-2d must be supplied together")

    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    image_3d, mask_3d = _load_pair(image_3d_path, mask_3d_path)
    if image_3d.GetDimension() != 3:
        raise ValueError("The 3D input must be a three-dimensional image")

    if args.image_2d is not None:
        image_2d_path = _require_file(args.image_2d, "2D image")
        mask_2d_path = _require_file(args.mask_2d, "2D mask")
        image_2d, mask_2d = _load_pair(image_2d_path, mask_2d_path)
    else:
        image_2d, mask_2d = _derive_2d_pair(image_3d, mask_3d)
    if image_2d.GetDimension() != 2:
        raise ValueError("The 2D input must be a two-dimensional image")

    results: list[RunResult] = []
    with tempfile.TemporaryDirectory(prefix="flash_radiomics_functionality_") as tmp:
        work_dir = Path(tmp)
        backend = args.backend
        if args.mode in {"all", "scalar-full"}:
            results.append(
                _run_extraction(
                    backend=backend,
                    test_name="scalar_3d",
                    mode="scalar",
                    dimension=3,
                    config=scalar_3d_config,
                    image_input=str(image_3d_path),
                    mask_input=str(mask_3d_path),
                    reference_image=image_3d,
                    reference_mask=mask_3d,
                    required_families=SCALAR_3D_FAMILIES,
                    work_dir=work_dir,
                )
            )
            results.append(
                _run_extraction(
                    backend=backend,
                    test_name="scalar_2d",
                    mode="scalar",
                    dimension=2,
                    config=scalar_2d_config,
                    image_input=image_2d,
                    mask_input=mask_2d,
                    reference_image=image_2d,
                    reference_mask=mask_2d,
                    required_families=SCALAR_2D_FAMILIES,
                    work_dir=work_dir,
                )
            )
        if args.mode in {"all", "spatial-full"}:
            results.append(
                _run_extraction(
                    backend=backend,
                    test_name="spatial_3d",
                    mode="spatial",
                    dimension=3,
                    config=spatial_config,
                    image_input=str(image_3d_path),
                    mask_input=str(mask_3d_path),
                    reference_image=image_3d,
                    reference_mask=mask_3d,
                    required_families=SPATIAL_FAMILIES,
                    work_dir=work_dir,
                )
            )
            results.append(
                _run_extraction(
                    backend=backend,
                    test_name="spatial_2d",
                    mode="spatial",
                    dimension=2,
                    config=spatial_config,
                    image_input=image_2d,
                    mask_input=mask_2d,
                    reference_image=image_2d,
                    reference_mask=mask_2d,
                    required_families=SPATIAL_FAMILIES,
                    work_dir=work_dir,
                )
            )
        if args.mode in {"all", "scalar-partial"}:
            results.append(
                _run_extraction(
                    backend=backend,
                    test_name="scalar_partial_3d",
                    mode="scalar",
                    dimension=3,
                    config=PROJECT_ROOT / "configs" / "extraction_template.yaml",
                    image_input=str(image_3d_path),
                    mask_input=str(mask_3d_path),
                    reference_image=image_3d,
                    reference_mask=mask_3d,
                    required_families=("firstorder", "glcm"),
                    work_dir=work_dir,
                    expected_feature_names=SCALAR_PARTIAL_FEATURE_NAMES,
                    expected_bin_width=25.0,
                )
            )
        if args.mode in {"all", "spatial-partial"}:
            results.append(
                _run_extraction(
                    backend=backend,
                    test_name="spatial_partial_3d",
                    mode="spatial",
                    dimension=3,
                    config=PROJECT_ROOT / "configs" / "extraction_template.yaml",
                    image_input=str(image_3d_path),
                    mask_input=str(mask_3d_path),
                    reference_image=image_3d,
                    reference_mask=mask_3d,
                    required_families=("firstorder", "glcm"),
                    work_dir=work_dir,
                    expected_feature_names=SPATIAL_PARTIAL_FEATURE_NAMES,
                    expected_bin_width=25.0,
                )
            )
        if args.mode in {"all", "combined-partial"}:
            results.append(
                _validate_combined_hdf5(
                    backend=backend,
                    image_path=image_3d_path,
                    mask_path=mask_3d_path,
                    work_dir=work_dir,
                    test_name="combined_partial_3d",
                    config=PROJECT_ROOT / "configs" / "extraction_template.yaml",
                )
            )
        if args.mode in {"all", "combined-full"}:
            results.append(
                _validate_combined_hdf5(
                    backend=backend,
                    image_path=image_3d_path,
                    mask_path=mask_3d_path,
                    work_dir=work_dir,
                    test_name="combined_full_3d",
                    config=None,
                )
            )
        _write_consolidated_hdf5(output_path, results, availability)

    print(f"backend_availability={json.dumps(availability, sort_keys=True)}")
    print(f"executed_backend={args.backend}")
    print(f"executed_mode={args.mode}")
    print(f"completed_runs={len(results)}")
    try:
        displayed_output = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        displayed_output = output_path
    print(f"output_hdf5={displayed_output}")
    print("status=pass")


if __name__ == "__main__":
    main()
