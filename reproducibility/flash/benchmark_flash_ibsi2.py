#!/usr/bin/env python3
"""Run current Flash against IBSI-2 phase 2 scalar feature references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import SimpleITK as sitk

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from reproducibility.common.ibsi1_validation_common import (
    _apply_resegmentation,
    _read_rounded_phantom,
    _resample_case,
)
from reproducibility.common.official_benchmark_common import (
    ensure_workspace_on_path,
    make_experiment_dir,
    relative_to_workspace,
    write_csv,
    write_json,
)


ensure_workspace_on_path()

from flash_radiomics import setVerbosity
from flash_radiomics.featureextractor import RadiomicsFeatureExtractor


IBSI2_CONFIG_JSON = WORKSPACE_ROOT / "reproducibility" / "data" / "ibsi" / "ibsi2_configs_phase1_phase2_validation.json"
IBSI2_IMAGE = WORKSPACE_ROOT / "reproducibility" / "data" / "ibsi" / "data_sets" / "ibsi_2_ct_radiomics_phantom" / "nifti" / "image" / "phantom.nii.gz"
IBSI2_MASK = WORKSPACE_ROOT / "reproducibility" / "data" / "ibsi" / "data_sets" / "ibsi_2_ct_radiomics_phantom" / "nifti" / "mask" / "mask.nii.gz"
IBSI2_TEMPLATE = WORKSPACE_ROOT / "reproducibility" / "data" / "ibsi" / "IBSI-2-Phase2-Submission-Template.csv"
IBSI_HIGH_CONSENSUS_MAP = WORKSPACE_ROOT / "reproducibility" / "ibsi" / "ibsi_high_consensus_165_feature_map.csv"
IBSI_LABEL = 1


FLASH_FIRSTORDER_BY_TAG = {
    "stat_mean": "Mean",
    "stat_var": "Variance",
    "stat_skew": "Skewness",
    "stat_kurt": "Kurtosis",
    "stat_median": "Median",
    "stat_min": "Minimum",
    "stat_p10": "10Percentile",
    "stat_p90": "90Percentile",
    "stat_max": "Maximum",
    "stat_iqr": "InterquartileRange",
    "stat_range": "Range",
    "stat_mad": "MeanAbsoluteDeviation",
    "stat_rmad": "RobustMeanAbsoluteDeviation",
    "stat_medad": "MedianAbsoluteDeviation",
    "stat_cov": "CoefficientOfVariation",
    "stat_qcod": "QuartileCoefficientOfDispersion",
    "stat_energy": "Energy",
    "stat_rms": "RootMeanSquared",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline current Flash against IBSI-2 phase 2 scalar references.")
    parser.add_argument("--config-json", type=Path, default=IBSI2_CONFIG_JSON)
    parser.add_argument("--image", type=Path, default=IBSI2_IMAGE)
    parser.add_argument("--mask", type=Path, default=IBSI2_MASK)
    parser.add_argument("--template", type=Path, default=IBSI2_TEMPLATE)
    parser.add_argument("--high-consensus-map", type=Path, default=IBSI_HIGH_CONSENSUS_MAP)
    parser.add_argument("--backend", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def _phase2_configs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "phase_2_feature_benchmark" in payload:
        phase2 = payload["phase_2_feature_benchmark"]
        if isinstance(phase2, dict) and "configs" in phase2:
            phase2 = phase2["configs"]
    else:
        phase2 = payload.get("configs", {})
    return {
        str(key): value
        for key, value in dict(phase2).items()
        if isinstance(value, dict) and "." in str(key)
    }


def _load_template_stats(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    return [row for row in rows if row.get("family") == "Statistics"]


def _load_165_mirp_tags(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["mirp"] for row in csv.DictReader(handle) if row.get("mirp")}


def _parse_angle_radians(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(" ", "")
    if text == "pi":
        return math.pi
    if text == "-pi":
        return -math.pi
    if "pi" in text:
        factor = text.replace("*pi", "").replace("pi", "")
        if factor in {"", "+"}:
            return math.pi
        if factor == "-":
            return -math.pi
        if "/" in factor:
            numerator, denominator = factor.split("/", 1)
            numerator_value = 1.0 if numerator in {"", "+"} else (-1.0 if numerator == "-" else float(numerator))
            return numerator_value * math.pi / float(denominator)
        return float(factor) * math.pi
    return float(text)


def _support_plan(config_id: str, config: dict[str, Any]) -> tuple[bool, str, dict[str, Any], str]:
    filter_cfg = dict(config.get("filter") or {})
    filter_name = str(filter_cfg.get("name", "none")).lower()
    mode = str(config.get("image_processing", {}).get("extraction_mode", "")).upper()
    boundary_condition = str(
        config.get("image_processing", {}).get("image_filtering", {}).get("boundary_condition", "mirror")
    )

    if filter_name == "none":
        return True, "Original", {}, "current_flash_original"

    if filter_name == "mean":
        kernel_size = int(filter_cfg.get("support_voxels", 5))
        custom_args = {
            "meanKernelSize": kernel_size,
            "meanBoundaryCondition": boundary_condition,
        }
        if mode == "2D":
            custom_args["force2D"] = True
            custom_args["force2Ddimension"] = 0
        return True, "Mean", custom_args, "current_flash_mean_filter"

    if filter_name == "laplacian_of_gaussian":
        sigma = float(filter_cfg.get("scale_sigma_star_mm", 1.5))
        custom_args = {
            "sigma": [sigma],
            "logNormalizeAcrossScale": False,
            "logBoundaryCondition": boundary_condition,
            "logUseScipy": True,
        }
        if mode == "2D":
            custom_args["force2D"] = True
            custom_args["force2Ddimension"] = 0
            return True, "LoG", custom_args, "current_flash_log_2d_ibsi_unscaled"
        return True, "LoG", custom_args, "current_flash_log_3d_ibsi_unscaled"

    if filter_name == "laws":
        custom_args = {
            "lawsResponseMap": str(filter_cfg.get("response_map")),
            "lawsEnergyMap": bool(filter_cfg.get("energy_map", False)),
            "lawsDelta": int(filter_cfg.get("energy_map_distance_delta_voxels", 7)),
            "lawsRotationalInvariance": bool(filter_cfg.get("pseudo_rotational_invariance", False)),
            "lawsPoolingMethod": str(filter_cfg.get("response_map_pooling", "max")),
            "lawsBoundaryCondition": boundary_condition,
        }
        return True, "Laws", custom_args, "current_flash_laws_ibsi"

    if filter_name == "gabor":
        if filter_cfg.get("filter_orientation_theta") is not None:
            theta_values = [_parse_angle_radians(filter_cfg.get("filter_orientation_theta"))]
        else:
            theta_step = _parse_angle_radians(filter_cfg.get("filter_orientation_stepping_delta_theta", "pi/8"))
            theta_values = [float(value) for value in list(math.pi * index / round(math.pi / theta_step) for index in range(round(math.pi / theta_step)))]
        custom_args = {
            "gaborSigma": float(filter_cfg.get("scale_sigma_star_mm", 5.0)),
            "gaborLambda": float(filter_cfg.get("wavelength_lambda_star_mm", 2.0)),
            "gaborGamma": float(filter_cfg.get("ellipticity_gamma", 1.5)),
            "gaborThetaValues": theta_values,
            "gaborResponse": str(filter_cfg.get("response_map", "modulus")),
            "gaborPoolingMethod": str(filter_cfg.get("response_map_pooling", "mean")),
            "gaborBoundaryCondition": boundary_condition,
            "gaborStackAxes": [0, 1, 2] if filter_cfg.get("average_2d_responses_over_orthogonal_planes") else [0],
        }
        return True, "Gabor", custom_args, "current_flash_gabor_ibsi"

    if filter_name == "daubechies_3_wavelet":
        custom_args = {
            "ibsiWaveletName": "db3",
            "ibsiWaveletCombination": str(filter_cfg.get("wavelet_filter_combination")),
            "ibsiWaveletLevel": int(filter_cfg.get("wavelet_decomposition_level", 1)),
            "ibsiWaveletBoundaryCondition": boundary_condition,
            "ibsiWaveletRotationalInvariance": bool(filter_cfg.get("pseudo_rotational_invariance", False)),
            "ibsiWaveletPoolingMethod": str(filter_cfg.get("response_map_pooling", "mean")),
        }
        return True, "IBSIWavelet", custom_args, "current_flash_db3_wavelet_ibsi"

    if filter_name == "simoncelli_wavelet":
        custom_args = {
            "simoncelliLevel": int(filter_cfg.get("wavelet_decomposition_level", 1)),
            "simoncelliResponse": "real",
        }
        if mode == "2D":
            custom_args["force2D"] = True
            custom_args["force2Ddimension"] = 0
        return True, "SimoncelliWavelet", custom_args, "current_flash_simoncelli_wavelet_ibsi"

    if filter_name == "riesz_transformed_simoncelli_wavelet":
        custom_args = {
            "rieszSimoncelliLevel": int(filter_cfg.get("wavelet_decomposition_level", 1)),
            "rieszSimoncelliResponse": "real",
            "rieszOrder": list(filter_cfg.get("riesz_transformation_order", [])),
            "rieszSteering": bool(filter_cfg.get("riesz_filter_steering", False)),
            "rieszTensorSigma": float(filter_cfg.get("riesz_structure_tensor_window_scale_sigma_star_mm", 1.0)),
        }
        if mode == "2D":
            custom_args["force2D"] = True
            custom_args["force2Ddimension"] = 0
        note = "current_flash_riesz_steered_simoncelli_wavelet" if custom_args["rieszSteering"] else "current_flash_riesz_simoncelli_wavelet"
        return True, "RieszTransformedSimoncelliWavelet", custom_args, note

    return False, "", {}, f"unsupported_current_flash_filter:{filter_name}"


def _prepare_case(image_path: Path, mask_path: Path, config: dict[str, Any]) -> tuple[sitk.Image, sitk.Image]:
    image, mask = _read_rounded_phantom(image_path, mask_path)
    processing = dict(config.get("image_processing") or {})
    image, mask = _resample_case(image, mask, processing)
    mask = _apply_resegmentation(image, mask, processing)
    return image, mask


def _run_flash_config(
    *,
    image_path: Path,
    mask_path: Path,
    config_id: str,
    config: dict[str, Any],
    backend: str,
    num_threads: int,
) -> tuple[dict[str, float], dict[str, str], str]:
    supported, image_type, custom_args, note = _support_plan(config_id, config)
    if not supported:
        return {}, {}, note

    image, mask = _prepare_case(image_path, mask_path, config)

    extractor = RadiomicsFeatureExtractor(
        backend=backend,
        num_threads=max(1, int(num_threads)),
        additionalInfo=False,
        voxelArrayShift=0,
    )
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()
    extractor.enableImageTypeByName(image_type, customArgs=custom_args)
    extractor.enableFeaturesByName(firstorder=list(FLASH_FIRSTORDER_BY_TAG.values()))

    raw_result = extractor.execute(image, mask, label=IBSI_LABEL, spatialMode=False)
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    for tag, flash_name in FLASH_FIRSTORDER_BY_TAG.items():
        suffix = f"_firstorder_{flash_name}"
        matches = [key for key in raw_result if str(key).endswith(suffix)]
        if not matches:
            continue
        source_key = str(matches[0])
        value = raw_result[source_key]
        try:
            values[tag] = float(value)
            sources[tag] = source_key
        except (TypeError, ValueError):
            continue
    return values, sources, note


def _evaluate(
    config_id: str,
    config: dict[str, Any],
    template_stats: list[dict[str, str]],
    values: dict[str, float],
    sources: dict[str, str],
    note: str,
    in_165: set[str],
) -> list[dict[str, Any]]:
    reference_values = config.get("reference_values")
    if not isinstance(reference_values, dict):
        reference_values = {}

    rows: list[dict[str, Any]] = []
    for stat_row in template_stats:
        tag = stat_row["feature_tag"]
        ref = reference_values.get(tag)
        reference_value = None
        tolerance = None
        if isinstance(ref, dict):
            reference_value = ref.get("consensus_value")
            tolerance = ref.get("tolerance")

        result_value = values.get(tag)
        status = "unsupported"
        passed = ""
        abs_diff = float("nan")
        if (
            reference_value is None
            or tolerance is None
            or not math.isfinite(float(reference_value))
            or not math.isfinite(float(tolerance))
        ):
            status = "not_standardized"
        elif result_value is None:
            status = "unsupported" if values == {} else "error"
        elif not math.isfinite(float(result_value)):
            status = "error"
        else:
            abs_diff = abs(float(result_value) - float(reference_value))
            status = "pass" if abs_diff <= float(tolerance) else "fail"
            passed = "1" if status == "pass" else "0"

        rows.append(
            {
                "config_id": config_id,
                "filter_name": str((config.get("filter") or {}).get("name", "none")),
                "feature_tag": tag,
                "feature_name": stat_row["feature_name"],
                "ibsi_identifier": stat_row["ibsi_identifier"],
                "in_165_feature_set": "1" if tag in in_165 else "0",
                "reference_value": "" if reference_value is None else float(reference_value),
                "tolerance": "" if tolerance is None else float(tolerance),
                "flash_value": "" if result_value is None else float(result_value),
                "absolute_difference": abs_diff,
                "status": status,
                "passed": passed,
                "source_key": sources.get(tag, ""),
                "note": note,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    setVerbosity(60)

    payload = json.loads(args.config_json.read_text())
    configs = _phase2_configs(payload)
    template_stats = _load_template_stats(args.template)
    in_165 = _load_165_mirp_tags(args.high_consensus_map)

    exp_dir = make_experiment_dir("flash", "ibsi2", args.backend)
    validation_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []

    for config_id in sorted(configs, key=lambda text: (int(text.split(".")[0]), text.split(".")[1])):
        config = configs[config_id]
        reference_available = bool(config.get("reference_values_available"))
        supported, image_type, custom_args, support_note = _support_plan(config_id, config)
        values: dict[str, float] = {}
        sources: dict[str, str] = {}
        note = support_note
        error = ""
        if supported:
            try:
                values, sources, note = _run_flash_config(
                    image_path=args.image,
                    mask_path=args.mask,
                    config_id=config_id,
                    config=config,
                    backend=args.backend,
                    num_threads=args.num_threads,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                note = f"flash_error:{type(exc).__name__}"

        rows = _evaluate(config_id, config, template_stats, values, sources, note, in_165)
        if error:
            for row in rows:
                if row["status"] == "unsupported":
                    row["status"] = "error"
                row["note"] = note
        validation_rows.extend(rows)
        config_rows.append(
            {
                "config_id": config_id,
                "filter_name": str((config.get("filter") or {}).get("name", "none")),
                "reference_values_available": int(reference_available),
                "current_flash_supported": int(bool(supported)),
                "current_flash_image_type": image_type,
                "current_flash_custom_args": json.dumps(custom_args, sort_keys=True),
                "note": note,
                "error": error,
                "pass": sum(1 for row in rows if row["status"] == "pass"),
                "fail": sum(1 for row in rows if row["status"] == "fail"),
                "unsupported": sum(1 for row in rows if row["status"] == "unsupported"),
                "error_count": sum(1 for row in rows if row["status"] == "error"),
                "not_standardized": sum(1 for row in rows if row["status"] == "not_standardized"),
            }
        )

    status_counts: dict[str, int] = {}
    for row in validation_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    reference_rows = [row for row in validation_rows if row["reference_value"] != ""]
    finite_reference_rows = [
        row
        for row in reference_rows
        if (
            math.isfinite(float(row["reference_value"]))
            and row["tolerance"] != ""
            and math.isfinite(float(row["tolerance"]))
        )
    ]
    pass_rows = [row for row in finite_reference_rows if row["status"] == "pass"]
    fail_rows = [row for row in finite_reference_rows if row["status"] == "fail"]
    unsupported_rows = [row for row in finite_reference_rows if row["status"] == "unsupported"]
    error_rows = [row for row in finite_reference_rows if row["status"] == "error"]

    validation_csv = exp_dir / "flash_ibsi2_validation.csv"
    config_csv = exp_dir / "flash_ibsi2_config_status.csv"
    summary_json = exp_dir / "flash_ibsi2_validation_summary.json"

    validation_fields = [
        "config_id",
        "filter_name",
        "feature_tag",
        "feature_name",
        "ibsi_identifier",
        "in_165_feature_set",
        "reference_value",
        "tolerance",
        "flash_value",
        "absolute_difference",
        "status",
        "passed",
        "source_key",
        "note",
    ]
    config_fields = [
        "config_id",
        "filter_name",
        "reference_values_available",
        "current_flash_supported",
        "current_flash_image_type",
        "current_flash_custom_args",
        "note",
        "error",
        "pass",
        "fail",
        "unsupported",
        "error_count",
        "not_standardized",
    ]
    write_csv(validation_csv, validation_fields, validation_rows)
    write_csv(config_csv, config_fields, config_rows)
    write_json(
        summary_json,
        {
            "tool": "Flash",
            "mode": "ibsi2_phase2_current_baseline",
            "backend": args.backend,
            "config_json": relative_to_workspace(args.config_json),
            "image": relative_to_workspace(args.image),
            "mask": relative_to_workspace(args.mask),
            "template": relative_to_workspace(args.template),
            "high_consensus_map": relative_to_workspace(args.high_consensus_map),
            "configs_total": len(configs),
            "template_statistical_features": len(template_stats),
            "all_template_stats_in_165": all(row["feature_tag"] in in_165 for row in template_stats),
            "status_counts_all_rows": status_counts,
            "reference_rows": len(reference_rows),
            "finite_reference_rows": len(finite_reference_rows),
            "pass_count": len(pass_rows),
            "fail_count": len(fail_rows),
            "unsupported_count": len(unsupported_rows),
            "error_count": len(error_rows),
            "pass_rate_finite_including_unsupported": (
                len(pass_rows) / len(finite_reference_rows) if finite_reference_rows else float("nan")
            ),
            "pass_rate_finite_computed_only": (
                len(pass_rows) / (len(pass_rows) + len(fail_rows)) if (len(pass_rows) + len(fail_rows)) else float("nan")
            ),
            "validation_csv": relative_to_workspace(validation_csv),
            "config_status_csv": relative_to_workspace(config_csv),
        },
    )

    print(f"[INFO] Wrote {relative_to_workspace(validation_csv)}")
    print(f"[INFO] Wrote {relative_to_workspace(config_csv)}")
    print(f"[INFO] Wrote {relative_to_workspace(summary_json)}")
    print(
        "[INFO] reference rows finite/total = "
        f"{len(finite_reference_rows)}/{len(reference_rows)}"
    )
    print(
        "[INFO] finite rows pass/fail/unsupported/error = "
        f"{len(pass_rows)}/{len(fail_rows)}/{len(unsupported_rows)}/{len(error_rows)}"
    )


if __name__ == "__main__":
    main()
