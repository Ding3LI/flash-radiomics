#!/usr/bin/env python3
"""Extract and compare the 93 spatial maps used by the scaling benchmark.

Run this program once with ``--backend cpu`` and once with ``--backend cuda``
against the same ``--run-dir`` on shared storage.  Each invocation writes its
93 NIfTI maps, waits for the peer backend, then writes
``spatial_map_concordance.csv`` containing one row per map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flash_scale_time.run_voxel_scaling_benchmark import PYRADIOMICS_COMMON_93_FEATURES


DEFAULT_IMAGE = PROJECT_ROOT / "reproducibility/data/mama_mia/nifti/DUKE_097/DUKE_097_0001.nii.gz"
DEFAULT_MASK = PROJECT_ROOT / "reproducibility/data/mama_mia/segmentations/expert/DUKE_097.nii.gz"
DEFAULT_RUN_DIR = PROJECT_ROOT / "reproducibility/spatial_map_concordance/DUKE_097"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _expected_feature_keys() -> tuple[str, ...]:
    keys = tuple(
        f"original_{feature_class}_{feature_name}"
        for feature_class, feature_names in PYRADIOMICS_COMMON_93_FEATURES.items()
        for feature_name in feature_names
    )
    if len(keys) != 93:
        raise RuntimeError(f"The scaling benchmark contract has {len(keys)} features, not 93")
    return keys


def _build_extractor(backend: str, voxel_batch: int):
    import flash_radiomics
    from flash_radiomics.featureextractor import RadiomicsFeatureExtractor

    flash_radiomics.setVerbosity(40)
    extractor = RadiomicsFeatureExtractor(backend=backend, num_threads=1)
    extractor.disableAllFeatures()
    for feature_class, feature_names in PYRADIOMICS_COMMON_93_FEATURES.items():
        extractor.enableFeaturesByName(**{feature_class: list(feature_names)})
    extractor.settings.update(
        {
            "additionalInfo": False,
            "kernelRadius": 1,
            "maskedKernel": True,
            "binWidth": 25.0,
            "distances": [1],
            "gldm_a": 0,
            "initValue": float("nan"),
            "voxelBatch": int(voxel_batch),
        }
    )
    return extractor


def _save_backend_maps(
    *,
    backend: str,
    image_path: Path,
    mask_path: Path,
    label: int,
    run_dir: Path,
    voxel_batch: int,
) -> None:
    expected_keys = _expected_feature_keys()
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    extractor = _build_extractor(backend, voxel_batch)
    result = extractor.execute(image, mask, label=int(label), spatialMode=True)
    maps = {
        str(key): value
        for key, value in dict(result).items()
        if str(key).startswith("original_") and isinstance(value, sitk.Image)
    }
    actual_keys = tuple(sorted(maps))
    if set(actual_keys) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(actual_keys))
        extra = sorted(set(actual_keys) - set(expected_keys))
        raise RuntimeError(
            f"{backend} did not produce exactly the 93-map contract "
            f"(count={len(actual_keys)}, missing={missing}, extra={extra})"
        )

    backend_dir = run_dir / "maps" / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    for key in expected_keys:
        sitk.WriteImage(maps[key], str(backend_dir / f"{key}.nii.gz"), True)

    manifest = {
        "backend": backend,
        "feature_count": len(expected_keys),
        "feature_keys": list(expected_keys),
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "label": int(label),
        "voxel_batch": int(voxel_batch),
    }
    marker_tmp = run_dir / f".{backend}_complete.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    marker_final = run_dir / f"{backend}_complete.json"
    marker_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    marker_tmp.replace(marker_final)


def _wait_for_peer(run_dir: Path, backend: str, timeout_s: float) -> str:
    peer = "cuda" if backend == "cpu" else "cpu"
    marker = run_dir / f"{peer}_complete.json"
    deadline = time.monotonic() + float(timeout_s)
    while not marker.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_s:.0f} s waiting for {peer} maps at {marker}"
            )
        time.sleep(5)
    return peer


def _geometry_matches(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), rtol=0.0, atol=1e-8)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), rtol=0.0, atol=1e-8)
        and np.allclose(left.GetDirection(), right.GetDirection(), rtol=0.0, atol=1e-8)
    )


def _roi_in_reference(mask: sitk.Image, label: int, reference: sitk.Image) -> np.ndarray:
    roi = sitk.Equal(mask, int(label))
    resampled = sitk.Resample(
        roi,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return sitk.GetArrayFromImage(resampled).astype(bool, copy=False)


def _compare_map(
    feature_key: str,
    cpu_path: Path,
    cuda_path: Path,
    mask: sitk.Image,
    label: int,
    rel_tol: float,
    abs_tol: float,
) -> dict[str, Any]:
    cpu_map = sitk.ReadImage(str(cpu_path))
    cuda_map = sitk.ReadImage(str(cuda_path))
    geometry_match = _geometry_matches(cpu_map, cuda_map)
    cpu_array = sitk.GetArrayFromImage(cpu_map).astype(np.float64, copy=False)
    cuda_array = sitk.GetArrayFromImage(cuda_map).astype(np.float64, copy=False)
    roi = _roi_in_reference(mask, label, cpu_map)

    if roi.shape != cpu_array.shape or cuda_array.shape != cpu_array.shape:
        return {
            "feature_key": feature_key,
            "concordant": False,
            "geometry_match": geometry_match,
            "roi_voxels": int(np.sum(roi)),
            "finite_value_count_cpu": 0,
            "finite_value_count_cuda": 0,
            "nonfinite_pattern_match": False,
            "max_abs_difference": math.nan,
            "max_relative_difference": math.nan,
            "reason": "array_shape_mismatch",
        }

    cpu_values = cpu_array[roi]
    cuda_values = cuda_array[roi]
    finite_cpu = np.isfinite(cpu_values)
    finite_cuda = np.isfinite(cuda_values)
    nonfinite_pattern_match = bool(
        np.array_equal(np.isnan(cpu_values), np.isnan(cuda_values))
        and np.array_equal(np.isposinf(cpu_values), np.isposinf(cuda_values))
        and np.array_equal(np.isneginf(cpu_values), np.isneginf(cuda_values))
    )
    both_finite = finite_cpu & finite_cuda
    if np.any(both_finite):
        absolute = np.abs(cpu_values[both_finite] - cuda_values[both_finite])
        denominator = np.maximum(np.abs(cpu_values[both_finite]), 1.0)
        relative = absolute / denominator
        max_absolute = float(np.max(absolute))
        max_relative = float(np.max(relative))
        numeric_match = bool(
            np.all((absolute <= float(abs_tol)) | (relative <= float(rel_tol)))
        )
    else:
        max_absolute = math.nan
        max_relative = math.nan
        numeric_match = True
    concordant = bool(geometry_match and nonfinite_pattern_match and numeric_match)
    return {
        "feature_key": feature_key,
        "concordant": concordant,
        "geometry_match": geometry_match,
        "roi_voxels": int(np.sum(roi)),
        "finite_value_count_cpu": int(np.sum(finite_cpu)),
        "finite_value_count_cuda": int(np.sum(finite_cuda)),
        "nonfinite_pattern_match": nonfinite_pattern_match,
        "max_abs_difference": max_absolute,
        "max_relative_difference": max_relative,
        "reason": "" if concordant else "value_or_geometry_difference",
    }


def _write_concordance_csv(
    run_dir: Path,
    mask_path: Path,
    label: int,
    rel_tol: float,
    abs_tol: float,
) -> tuple[Path, bool]:
    expected_keys = _expected_feature_keys()
    mask = sitk.ReadImage(str(mask_path))
    rows = [
        _compare_map(
            feature_key=key,
            cpu_path=run_dir / "maps/cpu" / f"{key}.nii.gz",
            cuda_path=run_dir / "maps/cuda" / f"{key}.nii.gz",
            mask=mask,
            label=label,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        for key in expected_keys
    ]
    all_concordant = all(bool(row["concordant"]) for row in rows)
    for row in rows:
        row["expected_feature_count"] = len(expected_keys)
        row["all_93_concordant"] = all_concordant
        row["criterion"] = "absolute_or_relative"
        row["relative_tolerance"] = float(rel_tol)
        row["absolute_tolerance"] = float(abs_tol)

    fields = [
        "feature_key",
        "concordant",
        "geometry_match",
        "roi_voxels",
        "finite_value_count_cpu",
        "finite_value_count_cuda",
        "nonfinite_pattern_match",
        "max_abs_difference",
        "max_relative_difference",
        "reason",
        "expected_feature_count",
        "all_93_concordant",
        "criterion",
        "relative_tolerance",
        "absolute_tolerance",
    ]
    output = run_dir / "spatial_map_concordance.csv"
    temporary = run_dir / f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return output, all_concordant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--voxel-batch", type=int, default=0)
    parser.add_argument("--peer-wait-s", type=float, default=7200.0)
    parser.add_argument("--rel-tol", "--rtol", dest="rel_tol", type=float, default=1e-11)
    parser.add_argument("--abs-tol", "--atol", dest="abs_tol", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    image_path = _resolve_path(args.image)
    mask_path = _resolve_path(args.mask)
    run_dir = _resolve_path(args.run_dir)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if int(args.label) <= 0:
        raise ValueError("--label must be positive")
    if float(args.rel_tol) < 0 or float(args.abs_tol) < 0:
        raise ValueError("Concordance tolerances must be non-negative")

    voxel_batch = int(args.voxel_batch)
    if voxel_batch <= 0:
        voxel_batch = 32768 if args.backend == "cuda" else 2048
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_backend_maps(
        backend=args.backend,
        image_path=image_path,
        mask_path=mask_path,
        label=int(args.label),
        run_dir=run_dir,
        voxel_batch=voxel_batch,
    )
    peer = _wait_for_peer(run_dir, args.backend, float(args.peer_wait_s))
    output, all_concordant = _write_concordance_csv(
        run_dir,
        mask_path,
        int(args.label),
        float(args.rel_tol),
        float(args.abs_tol),
    )
    print(f"backend={args.backend}")
    print(f"peer_backend={peer}")
    print(f"concordance_csv={output}")
    if not all_concordant:
        raise SystemExit("One or more of the 93 spatial maps are not concordant")


if __name__ == "__main__":
    main()
