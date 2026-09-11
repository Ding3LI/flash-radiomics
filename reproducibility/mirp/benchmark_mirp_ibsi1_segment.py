#!/usr/bin/env python3
"""Run MIRP segment-based IBSI-1 validation for configurations A-E."""

from __future__ import annotations

import argparse
import math
import numbers
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from mirp.extract_features_and_images import extract_features
from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
from mirp.settings.generic import SettingsClass
from mirp.settings.general_parameters import GeneralSettingsClass
from mirp.settings.image_processing_parameters import ImagePostProcessingClass
from mirp.settings.interpolation_parameters import ImageInterpolationSettingsClass, MaskInterpolationSettingsClass
from mirp.settings.perturbation_parameters import ImagePerturbationSettingsClass
from mirp.settings.resegmentation_parameters import ResegmentationSettingsClass
from mirp.settings.transformation_parameters import ImageTransformationSettingsClass

from reproducibility.common.ibsi1_validation_common import (
    ALL_ROWS_STATUS_ORDER,
    DEFAULT_IBSI1_TEMPLATE,
    DEFAULT_IBSI_HIGH_CONSENSUS_MAP,
    _evaluate_validation_rows,
    filter_reference_rows_to_high_consensus,
    load_high_consensus_feature_tags,
    load_ibsi1_reference_rows,
)
from reproducibility.common.official_benchmark_common import make_experiment_dir, relative_to_workspace, write_csv, write_json


CONFIG_IDS = ("A", "B", "C", "D", "E")
IBSI1_DATA_ROOT = WORKSPACE_ROOT / "reproducibility" / "data" / "ibsi" / "data_sets"
IBSI1_DICOM_IMAGE = IBSI1_DATA_ROOT / "ibsi_1_ct_radiomics_phantom" / "dicom" / "image"
IBSI1_DICOM_MASK = IBSI1_DATA_ROOT / "ibsi_1_ct_radiomics_phantom" / "dicom" / "mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIRP segment IBSI-1 validation against the local 165-feature map.")
    parser.add_argument("--ibsi-template", type=Path, default=DEFAULT_IBSI1_TEMPLATE)
    parser.add_argument("--high-consensus-map", type=Path, default=DEFAULT_IBSI_HIGH_CONSENSUS_MAP)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def _settings_for_config(config_id: str) -> SettingsClass:
    config = config_id.upper()
    by_slice = config in {"A", "B"}
    general_settings = GeneralSettingsClass(by_slice=by_slice)

    interpolation_kwargs: dict[str, Any] = {
        "by_slice": general_settings.by_slice,
        "anti_aliasing": False,
    }
    if config == "B":
        interpolation_kwargs.update({"spline_order": 1, "new_spacing": 2.0})
    elif config in {"C", "D"}:
        interpolation_kwargs.update({"spline_order": 1, "new_spacing": 2.0})
    elif config == "E":
        interpolation_kwargs.update({"spline_order": 3, "new_spacing": 2.0})

    resegmentation_kwargs: dict[str, Any] = {}
    if config in {"A", "B"}:
        resegmentation_kwargs["resegmentation_intensity_range"] = [-500.0, 400.0]
    elif config == "C":
        resegmentation_kwargs["resegmentation_intensity_range"] = [-1000.0, 400.0]
    elif config == "D":
        resegmentation_kwargs["resegmentation_sigma"] = 3.0
    elif config == "E":
        resegmentation_kwargs["resegmentation_intensity_range"] = [-1000.0, 400.0]
        resegmentation_kwargs["resegmentation_sigma"] = 3.0

    feature_kwargs: dict[str, Any] = {
        "by_slice": general_settings.by_slice,
        "no_approximation": False,
        "base_feature_families": "all",
        "glcm_distance": 1.0,
        "ngldm_distance": 1.0,
        "ngldm_difference_level": 0.0,
    }
    if config in {"A", "C"}:
        feature_kwargs.update(
            {
                "base_discretisation_method": "fixed_bin_size",
                "base_discretisation_bin_width": 25.0,
            }
        )
    else:
        feature_kwargs.update(
            {
                "base_discretisation_method": "fixed_bin_number",
                "base_discretisation_n_bins": 32,
            }
        )

    if config in {"A", "B"}:
        feature_kwargs.update(
            {
                "ivh_discretisation_method": "none",
                "glcm_spatial_method": ["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"],
                "glrlm_spatial_method": ["2d_average", "2d_slice_merge", "2.5d_direction_merge", "2.5d_volume_merge"],
                "glszm_spatial_method": ["2d", "2.5d"],
                "gldzm_spatial_method": ["2d", "2.5d"],
                "ngtdm_spatial_method": ["2d", "2.5d"],
                "ngldm_spatial_method": ["2d", "2.5d"],
            }
        )
    else:
        feature_kwargs.update(
            {
                "glcm_spatial_method": ["3d_average", "3d_volume_merge"],
                "glrlm_spatial_method": ["3d_average", "3d_volume_merge"],
                "glszm_spatial_method": "3d",
                "gldzm_spatial_method": "3d",
                "ngtdm_spatial_method": "3d",
                "ngldm_spatial_method": "3d",
            }
        )
        if config == "C":
            feature_kwargs.update(
                {
                    "ivh_discretisation_method": "fixed_bin_size",
                    "ivh_discretisation_bin_width": 2.5,
                }
            )
        elif config == "E":
            feature_kwargs.update(
                {
                    "ivh_discretisation_method": "fixed_bin_number",
                    "ivh_discretisation_n_bins": 1000,
                }
            )

    return SettingsClass(
        general_settings=general_settings,
        post_process_settings=ImagePostProcessingClass(),
        img_interpolate_settings=ImageInterpolationSettingsClass(**interpolation_kwargs),
        roi_interpolate_settings=MaskInterpolationSettingsClass(),
        roi_resegment_settings=ResegmentationSettingsClass(**resegmentation_kwargs),
        perturbation_settings=ImagePerturbationSettingsClass(),
        img_transform_settings=ImageTransformationSettingsClass(
            by_slice=general_settings.by_slice,
            response_map_feature_settings=None,
        ),
        feature_extr_settings=FeatureExtractionSettingsClass(**feature_kwargs),
    )


def _normalise_mirp_key(key: str) -> str:
    tag = str(key)
    for suffix in ("_fbs_w25.0", "_fbs_w2.5", "_fbn_n32", "_fbn_n1000"):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
            break

    tag = re.sub(r"_d1(?:\.0)?_", "_", tag)
    tag = re.sub(r"_a0(?:\.0)?_", "_", tag)

    replacements = (
        ("_2d_s_mrg", "_2D_comb"),
        ("_2d_avg", "_2D_avg"),
        ("_2.5d_d_mrg", "_2_5D_avg"),
        ("_2.5d_v_mrg", "_2_5D_comb"),
        ("_3d_v_mrg", "_3D_comb"),
        ("_3d_avg", "_3D_avg"),
        ("_2.5d", "_2_5D"),
        ("_2d", "_2D"),
        ("_3d", "_3D"),
    )
    for old, new in replacements:
        if tag.endswith(old):
            tag = tag[: -len(old)] + new
            break
    return tag


def _run_mirp_once(config_id: str) -> dict[str, Any]:
    data = extract_features(
        write_features=False,
        export_features=True,
        image=str(IBSI1_DICOM_IMAGE),
        mask=str(IBSI1_DICOM_MASK),
        roi_name="GTV-1",
        settings=_settings_for_config(config_id),
    )
    if not data:
        raise RuntimeError(f"MIRP returned no data for configuration {config_id}.")
    return dict(data[0])


def _normalised_values(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    source_by_tag: dict[str, str] = {}
    for key, value in row.items():
        if hasattr(value, "values"):
            value = value.values[0]
        if not isinstance(value, numbers.Real):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        tag = _normalise_mirp_key(str(key))
        values[tag] = numeric
        source_by_tag[tag] = str(key)
    return values, source_by_tag


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be >= 1")

    allowed_tags = load_high_consensus_feature_tags(args.high_consensus_map)
    reference_rows, reference_filter = filter_reference_rows_to_high_consensus(
        load_ibsi1_reference_rows(args.ibsi_template),
        allowed_tags,
    )

    exp_dir = make_experiment_dir("mirp", "segment_ibsi1", "cpu")
    validation_csv = exp_dir / "mirp_ibsi1_validation.csv"
    timing_csv = exp_dir / "mirp_ibsi1_case_timing.csv"
    feature_csv = exp_dir / "mirp_ibsi1_feature_values_wide.csv"
    summary_json = exp_dir / "mirp_ibsi1_validation_summary.json"

    validation_rows: list[dict] = []
    timing_rows: list[dict] = []
    feature_rows: list[dict] = []
    summary_rows: list[dict] = []

    for config_id in CONFIG_IDS:
        durations: list[float] = []
        last_result: dict[str, Any] = {}
        for _ in range(args.repeats):
            start = time.perf_counter()
            last_result = _run_mirp_once(config_id)
            durations.append(float(time.perf_counter() - start))

        values_by_tag, source_by_tag = _normalised_values(last_result)
        notes_by_tag = {tag: "mirp_native_feature" for tag in values_by_tag}
        config_rows = _evaluate_validation_rows(
            reference_rows.get(config_id, []),
            values_by_tag,
            source_by_tag,
            notes_by_tag,
        )
        validation_rows.extend(config_rows)

        status_counts = {status: 0 for status in ALL_ROWS_STATUS_ORDER}
        for row in config_rows:
            status_counts[str(row["status"])] += 1
        comparable_rows = status_counts["pass"] + status_counts["fail"]

        timing_rows.append(
            {
                "config_id": config_id,
                "tool": "mirp",
                "backend": "cpu",
                "mode": "segment",
                "mean_wall_time_s": sum(durations) / len(durations),
                "min_wall_time_s": min(durations),
                "max_wall_time_s": max(durations),
                "feature_count": int(len(values_by_tag)),
                "comparable_rows": int(comparable_rows),
                "pass_count": int(status_counts["pass"]),
                "fail_count": int(status_counts["fail"]),
                "unsupported_count": int(status_counts["unsupported"]),
                "not_standardized_count": int(status_counts["not_standardized"]),
                "error_count": int(status_counts["error"]),
            }
        )
        summary_rows.append(
            {
                "config_id": config_id,
                "rows_total": int(len(config_rows)),
                "rows_comparable": int(comparable_rows),
                "rows_pass": int(status_counts["pass"]),
                "rows_fail": int(status_counts["fail"]),
                "rows_unsupported": int(status_counts["unsupported"]),
                "rows_not_standardized": int(status_counts["not_standardized"]),
                "rows_error": int(status_counts["error"]),
                "pass_rate": float(status_counts["pass"] / comparable_rows) if comparable_rows > 0 else float("nan"),
            }
        )
        feature_rows.append({"config_id": config_id, **values_by_tag})

    total_status = {status: 0 for status in ALL_ROWS_STATUS_ORDER}
    for row in validation_rows:
        total_status[str(row["status"])] += 1
    total_comparable = total_status["pass"] + total_status["fail"]

    validation_fieldnames = [
        "config_id",
        "dataset",
        "family",
        "feature",
        "consensus",
        "reference_value",
        "tolerance",
        "result_value",
        "absolute_difference",
        "status",
        "passed",
        "source_key",
        "note",
        "tag",
    ]
    timing_fieldnames = [
        "config_id",
        "tool",
        "backend",
        "mode",
        "mean_wall_time_s",
        "min_wall_time_s",
        "max_wall_time_s",
        "feature_count",
        "comparable_rows",
        "pass_count",
        "fail_count",
        "unsupported_count",
        "not_standardized_count",
        "error_count",
    ]
    feature_fieldnames = sorted({key for row in feature_rows for key in row})

    write_csv(validation_csv, validation_fieldnames, validation_rows)
    write_csv(timing_csv, timing_fieldnames, timing_rows)
    write_csv(feature_csv, feature_fieldnames, feature_rows)
    write_json(
        summary_json,
        {
            "tool": "mirp",
            "backend": "cpu",
            "mode": "segment",
            "suite": "ibsi1",
            "repeat": int(args.repeats),
            "config_ids": list(CONFIG_IDS),
            "high_consensus_only": True,
            "high_consensus_feature_count": int(len(allowed_tags)),
            "high_consensus_map_csv": relative_to_workspace(args.high_consensus_map),
            "reference_filter": reference_filter,
            "rows_total": int(len(validation_rows)),
            "rows_comparable": int(total_comparable),
            "rows_pass": int(total_status["pass"]),
            "rows_fail": int(total_status["fail"]),
            "rows_unsupported": int(total_status["unsupported"]),
            "rows_not_standardized": int(total_status["not_standardized"]),
            "rows_error": int(total_status["error"]),
            "pass_rate": float(total_status["pass"] / total_comparable) if total_comparable > 0 else float("nan"),
            "validation_csv": relative_to_workspace(validation_csv),
            "timing_csv": relative_to_workspace(timing_csv),
            "feature_values_csv": relative_to_workspace(feature_csv),
            "configs": summary_rows,
            "notes": [
                "MIRP is run with the IBSI-1 configurations A-E against the bundled DICOM phantom.",
                "MIRP native output suffixes are normalised to IBSI workbook tags before scoring.",
                "Only rows represented by reproducibility/ibsi/ibsi_high_consensus_165_feature_map.csv are scored.",
            ],
        },
    )

    print(f"exp_dir={relative_to_workspace(exp_dir)}")
    print(f"summary_json={relative_to_workspace(summary_json)}")


if __name__ == "__main__":
    main()
