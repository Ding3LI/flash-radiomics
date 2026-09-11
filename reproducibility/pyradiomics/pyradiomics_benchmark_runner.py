#!/usr/bin/env python3
"""PyRadiomics-only official benchmark runner for segment and voxel workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from reproducibility.common.official_benchmark_common import (
    build_feature_value_matrix,
    ensure_workspace_on_path,
    make_experiment_dir,
    relative_to_workspace,
    sum_mean_and_std,
    write_csv,
    write_json,
)
import reproducibility.common.segment_benchmark_helpers as SEGMENT_LEGACY
import reproducibility.common.spatial_benchmark_helpers as SPATIAL_HELPERS
from reproducibility.common.ibsi1_validation_common import (
    DEFAULT_IBSI1_CONFIG_JSON,
    DEFAULT_IBSI1_DATA_ROOT,
    DEFAULT_IBSI1_TEMPLATE,
    run_ibsi1_segment_validation,
)


ensure_workspace_on_path()


def _measure_feature_timings(
    legacy: Any,
    *,
    image: sitk.Image,
    mask: sitk.Image,
    label: int,
    feature_keys: list[str],
    repeats: int,
    repeat_workers: int,
    voxel_based: bool,
    build_extractor,
) -> dict[str, dict]:
    timings_by_feature: dict[str, dict] = {}
    for feature_key in sorted(feature_keys):
        parts = legacy._feature_parts(feature_key)
        if parts is None:
            continue
        feature_class, feature_name = parts
        try:
            extractor_factory = lambda: build_extractor(feature_class, feature_name)
            extractor = extractor_factory()
            run_timings, _, rss_stats = legacy._time_execute(
                extractor,
                image,
                mask,
                label,
                repeats,
                0,
                voxel_based=voxel_based,
                repeat_workers=repeat_workers,
                build_extractor=extractor_factory,
                collect_result=False,
            )
            stats = legacy._timing_stats(run_timings, rss_stats=rss_stats)
            stats["note"] = ""
            timings_by_feature[feature_key] = stats
        except Exception as exc:
            timings_by_feature[feature_key] = legacy._nan_timing_stats(
                note=f"pyradiomics_feature_timing_error:{type(exc).__name__}"
            )
    return timings_by_feature


def _build_segment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PyRadiomics segment-based extraction and write outputs into reproducibility/pyradiomics/exp."
    )
    parser.add_argument("--dataset-suite", choices=("kits", "ibsi1"), default="kits")
    parser.add_argument("--dataset-dir", type=Path, default=Path("kits23/dataset"))
    parser.add_argument("--ibsi-data-root", type=Path, default=DEFAULT_IBSI1_DATA_ROOT)
    parser.add_argument("--ibsi-template", type=Path, default=DEFAULT_IBSI1_TEMPLATE)
    parser.add_argument("--ibsi-config-json", type=Path, default=DEFAULT_IBSI1_CONFIG_JSON)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--labels", default="1,2")
    parser.add_argument("--bin-width", type=float, default=25.0)
    parser.add_argument("--distances", default="1")
    parser.add_argument("--segment-style", choices=("3d", "2d"), default="3d")
    parser.add_argument("--include-shape2d", action="store_true")
    parser.add_argument("--force2d-dimension", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-workers", type=int, default=0)
    return parser


def _build_voxel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PyRadiomics voxel-based extraction and write outputs into reproducibility/pyradiomics/exp."
    )
    parser.add_argument(
        "--dataset-suite",
        choices=("kits", "ibsi_validation"),
        default="kits",
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("kits23/dataset"))
    parser.add_argument(
        "--ibsi-validation-study-dir",
        type=Path,
        default=SPATIAL_HELPERS.DEFAULT_IBSI_VALIDATION_STUDY_DIR,
    )
    parser.add_argument("--modalities", default="CT,PET,MR_T1")
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--labels", default="1,2")
    parser.add_argument("--kernel-radius", type=int, default=1)
    parser.add_argument("--masked-kernel", type=int, default=1)
    parser.add_argument("--bin-width", type=float, default=25.0)
    parser.add_argument("--distances", default="1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-workers", type=int, default=0)
    parser.add_argument("--skip-feature-timings", action="store_true")
    parser.add_argument("--skip-map-save", action="store_true")
    return parser


def _segment_items(args) -> tuple[list[dict], list[str], list[int], dict]:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    labels = SEGMENT_LEGACY._parse_labels(args.labels)
    distances = SEGMENT_LEGACY._parse_distances(args.distances)
    case_ids = SEGMENT_LEGACY._resolve_case_ids(dataset_dir, args.case_ids, args.max_cases)
    if not case_ids:
        raise RuntimeError("No valid cases found to benchmark")

    force2d = bool(args.segment_style == "2d")
    base_settings = SEGMENT_LEGACY._base_settings(
        bin_width=args.bin_width,
        distances=distances,
        force2d=force2d,
        force2d_dimension=args.force2d_dimension,
        segment_class_workers=0,
    )

    items: list[dict] = []
    for case_id in case_ids:
        image_path = dataset_dir / case_id / "imaging.nii.gz"
        mask_path = dataset_dir / case_id / "segmentation.nii.gz"
        if not image_path.exists() or not mask_path.exists():
            for label in labels:
                items.append(
                    {
                        "case_id": case_id,
                        "label": int(label),
                        "roi_name": SEGMENT_LEGACY.ROI_NAME_BY_LABEL[label],
                        "roi_voxels": 0,
                        "note": "missing_case_files",
                        "can_run": False,
                        "feature_classes": tuple(),
                    }
                )
            continue

        mask = sitk.ReadImage(str(mask_path))
        mask_arr = sitk.GetArrayFromImage(mask)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(mask)

        for label in labels:
            note = ""
            roi_voxels = int(np.sum(mask_arr == int(label)))
            feature_classes, shape2d_note = SEGMENT_LEGACY._feature_classes_for_roi(
                mask,
                label_stats,
                int(label),
                bool(args.include_shape2d),
                force2d,
                int(args.force2d_dimension),
            )
            if shape2d_note:
                note = SEGMENT_LEGACY._append_note(note, shape2d_note)
            if roi_voxels <= 0:
                note = SEGMENT_LEGACY._append_note(note, "empty_roi")

            items.append(
                {
                    "case_id": case_id,
                    "label": int(label),
                    "roi_name": SEGMENT_LEGACY.ROI_NAME_BY_LABEL[label],
                    "roi_voxels": roi_voxels,
                    "note": note,
                    "can_run": bool(roi_voxels > 0),
                    "feature_classes": tuple(feature_classes),
                }
            )

    return items, case_ids, labels, base_settings


def _voxel_items(args) -> tuple[list[dict], list[str], list[int], dict]:
    labels = SPATIAL_HELPERS._parse_labels(args.labels)
    distances = SPATIAL_HELPERS._parse_distances(args.distances)
    base_settings = SPATIAL_HELPERS._base_settings(
        kernel_radius=args.kernel_radius,
        masked_kernel=bool(int(args.masked_kernel)),
        bin_width=args.bin_width,
        distances=distances,
        voxel_batch=1,
    )

    if args.dataset_suite == "ibsi_validation":
        case_records = SPATIAL_HELPERS._resolve_ibsi_validation_voxel_cases(
            Path(args.ibsi_validation_study_dir),
            args.modalities,
            args.max_cases,
        )
    else:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        case_ids = SPATIAL_HELPERS._resolve_case_ids(dataset_dir, args.case_ids, args.max_cases)
        case_records = [
            {
                "case_id": case_id,
                "study_id": "",
                "modality": "",
                "image_path": dataset_dir / case_id / "imaging.nii.gz",
                "mask_path": dataset_dir / case_id / "segmentation.nii.gz",
            }
            for case_id in case_ids
        ]
    if not case_records:
        raise RuntimeError("No valid cases found to benchmark")

    items: list[dict] = []
    for case_record in case_records:
        case_id = str(case_record["case_id"])
        image_path = Path(case_record["image_path"])
        mask_path = Path(case_record["mask_path"])
        if not image_path.exists() or not mask_path.exists():
            for label in labels:
                items.append(
                    {
                        "case_id": case_id,
                        "label": int(label),
                        "roi_name": SPATIAL_HELPERS.ROI_NAME_BY_LABEL[label],
                        "roi_voxels": 0,
                        "note": "missing_case_files",
                        "can_run": False,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "modality": str(case_record.get("modality", "")),
                        "study_id": str(case_record.get("study_id", "")),
                    }
                )
            continue

        mask = sitk.ReadImage(str(mask_path))
        mask_arr = sitk.GetArrayFromImage(mask)
        for label in labels:
            roi_voxels = int(np.sum(mask_arr == int(label)))
            items.append(
                {
                    "case_id": case_id,
                    "label": int(label),
                    "roi_name": SPATIAL_HELPERS.ROI_NAME_BY_LABEL[label],
                    "roi_voxels": roi_voxels,
                    "note": "" if roi_voxels > 0 else "empty_roi",
                    "can_run": bool(roi_voxels > 0),
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "modality": str(case_record.get("modality", "")),
                    "study_id": str(case_record.get("study_id", "")),
                }
            )

    case_ids = [str(record["case_id"]) for record in case_records]
    return items, case_ids, labels, base_settings


def _segment_run(args) -> None:
    SEGMENT_LEGACY._apply_runtime_patches()
    if args.dataset_suite == "ibsi1":
        exp_dir = make_experiment_dir("pyradiomics", "segment_ibsi1", "cpu")
        run_ibsi1_segment_validation(
            tool_name="pyradiomics",
            backend="cpu",
            exp_dir=exp_dir,
            repeats=args.repeats,
            data_root=Path(args.ibsi_data_root),
            template_path=Path(args.ibsi_template),
            config_json_path=Path(args.ibsi_config_json),
            build_extractor=lambda feature_classes, settings: SEGMENT_LEGACY._build_pyr_all(
                settings,
                feature_classes,
            ),
        )
        return

    items, case_ids, labels, base_settings = _segment_items(args)
    exp_dir = make_experiment_dir("pyradiomics", "segment", "cpu")

    overall_rows: list[dict] = []
    per_feature_rows: list[dict] = []

    dataset_dir = Path(args.dataset_dir)
    for item in items:
        overall_row = {
            "case_id": item["case_id"],
            "label": int(item["label"]),
            "roi_name": item["roi_name"],
            "roi_voxels": int(item["roi_voxels"]),
            "tool": "pyradiomics",
            "mode": "segment",
            "backend": "cpu",
            "mean_wall_time_s": float("nan"),
            "median_wall_time_s": float("nan"),
            "std_wall_time_s": float("nan"),
            "min_wall_time_s": float("nan"),
            "max_wall_time_s": float("nan"),
            "peak_rss_mb": float("nan"),
            "avg_rss_mb": float("nan"),
            "feature_count": 0,
            "note": str(item["note"]),
        }
        if not item["can_run"]:
            overall_rows.append(overall_row)
            continue

        image_path = dataset_dir / item["case_id"] / "imaging.nii.gz"
        mask_path = dataset_dir / item["case_id"] / "segmentation.nii.gz"
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))

        try:
            extractor_factory = lambda: SEGMENT_LEGACY._build_pyr_all(
                base_settings,
                item["feature_classes"],
            )
            extractor = extractor_factory()
            run_timings, result, rss_stats = SEGMENT_LEGACY._time_execute(
                extractor,
                image,
                mask,
                item["label"],
                args.repeats,
                0,
                voxel_based=False,
                repeat_workers=args.repeat_workers,
                build_extractor=extractor_factory,
                collect_result=True,
            )
            overall_stats = SEGMENT_LEGACY._timing_stats(run_timings, rss_stats=rss_stats)
            features = SEGMENT_LEGACY._extract_scalar_original_features(result)
            feature_timings = _measure_feature_timings(
                SEGMENT_LEGACY,
                image=image,
                mask=mask,
                label=item["label"],
                feature_keys=sorted(features.keys()),
                repeats=args.repeats,
                repeat_workers=args.repeat_workers,
                voxel_based=False,
                build_extractor=lambda feature_class, feature_name: SEGMENT_LEGACY._build_pyr_single_feature(
                    base_settings,
                    feature_class,
                    feature_name,
                ),
            )
            overall_row.update(
                {
                    "mean_wall_time_s": float(overall_stats["mean_s"]),
                    "median_wall_time_s": float(overall_stats["median_s"]),
                    "std_wall_time_s": float(overall_stats["std_s"]),
                    "min_wall_time_s": float(overall_stats["min_s"]),
                    "max_wall_time_s": float(overall_stats["max_s"]),
                    "peak_rss_mb": float(overall_stats["peak_rss_mb"]),
                    "avg_rss_mb": float(overall_stats["avg_rss_mb"]),
                    "feature_count": int(len(features)),
                }
            )
            for feature_key in sorted(features):
                parts = SEGMENT_LEGACY._feature_parts(feature_key)
                if parts is None:
                    continue
                feature_class, feature_name = parts
                timing_stats = SEGMENT_LEGACY._coerce_timing_stats(feature_timings.get(feature_key))
                per_feature_rows.append(
                    {
                        "case_id": item["case_id"],
                        "label": int(item["label"]),
                        "roi_name": item["roi_name"],
                        "roi_voxels": int(item["roi_voxels"]),
                        "tool": "pyradiomics",
                        "mode": "segment",
                        "backend": "cpu",
                        "feature_class": feature_class,
                        "feature": feature_name,
                        "feature_key": feature_key,
                        "mean_execution_time_s": float(timing_stats["mean_s"]),
                        "median_execution_time_s": float(timing_stats["median_s"]),
                        "std_execution_time_s": float(timing_stats["std_s"]),
                        "min_execution_time_s": float(timing_stats["min_s"]),
                        "max_execution_time_s": float(timing_stats["max_s"]),
                        "peak_rss_mb": float(timing_stats["peak_rss_mb"]),
                        "avg_rss_mb": float(timing_stats["avg_rss_mb"]),
                        "value_kind": "scalar",
                        "value": float(features[feature_key]),
                        "note": str(timing_stats.get("note", "")),
                    }
                )
        except Exception as exc:
            overall_row["note"] = SEGMENT_LEGACY._append_note(
                overall_row["note"],
                f"pyradiomics_segment_error:{type(exc).__name__}",
            )

        overall_rows.append(overall_row)

    overall_fieldnames = [
        "case_id",
        "label",
        "roi_name",
        "roi_voxels",
        "tool",
        "mode",
        "backend",
        "mean_wall_time_s",
        "median_wall_time_s",
        "std_wall_time_s",
        "min_wall_time_s",
        "max_wall_time_s",
        "peak_rss_mb",
        "avg_rss_mb",
        "feature_count",
        "note",
    ]
    per_feature_fieldnames = [
        "case_id",
        "label",
        "roi_name",
        "roi_voxels",
        "tool",
        "mode",
        "backend",
        "feature_class",
        "feature",
        "feature_key",
        "mean_execution_time_s",
        "median_execution_time_s",
        "std_execution_time_s",
        "min_execution_time_s",
        "max_execution_time_s",
        "peak_rss_mb",
        "avg_rss_mb",
        "value_kind",
        "value",
        "note",
    ]

    overall_csv = exp_dir / "pyradiomics_segment_case_timing.csv"
    per_feature_csv = exp_dir / "pyradiomics_segment_feature_timing.csv"
    feature_values_csv = exp_dir / "pyradiomics_segment_feature_values_wide.csv"
    summary_json = exp_dir / "pyradiomics_segment_benchmark_summary.json"

    feature_value_fieldnames, feature_value_rows = build_feature_value_matrix(per_feature_rows, "value")
    total_wall_mean, total_wall_std = sum_mean_and_std(
        overall_rows,
        "mean_wall_time_s",
        "std_wall_time_s",
    )
    total_exec_mean, total_exec_std = sum_mean_and_std(
        per_feature_rows,
        "mean_execution_time_s",
        "std_execution_time_s",
    )

    write_csv(overall_csv, overall_fieldnames, overall_rows)
    write_csv(per_feature_csv, per_feature_fieldnames, per_feature_rows)
    write_csv(feature_values_csv, feature_value_fieldnames, feature_value_rows)
    write_json(
        summary_json,
        {
            "tool": "pyradiomics",
            "mode": "segment",
            "backend": "cpu",
            "repeat": int(args.repeats),
            "number_of_cases": int(len(case_ids)),
            "number_of_rois": int(len(overall_rows)),
            "labels": [int(label) for label in labels],
            "case_ids": list(case_ids),
            "total_mean_wall_time_s": float(total_wall_mean),
            "std_wall_time_s": float(total_wall_std),
            "total_mean_execution_time_s": float(total_exec_mean),
            "std_execution_time_s": float(total_exec_std),
            "saved_feature_values_directory": relative_to_workspace(exp_dir),
            "feature_values_csv": relative_to_workspace(feature_values_csv),
            "overall_timing_csv": relative_to_workspace(overall_csv),
            "feature_timing_csv": relative_to_workspace(per_feature_csv),
        },
    )

    print(f"exp_dir={relative_to_workspace(exp_dir)}")
    print(f"summary_json={relative_to_workspace(summary_json)}")


def _voxel_run(args) -> None:
    SPATIAL_HELPERS._apply_runtime_patches()
    items, case_ids, labels, base_settings = _voxel_items(args)
    mode_name = "voxel_ibsi_validation" if args.dataset_suite == "ibsi_validation" else "voxel"
    exp_dir = make_experiment_dir("pyradiomics", mode_name, "cpu")
    voxel_map_dir = exp_dir / "voxel_maps"

    overall_rows: list[dict] = []
    per_feature_rows: list[dict] = []
    map_index_rows: list[dict] = []

    dataset_dir = Path(args.dataset_dir)
    for item in items:
        overall_row = {
            "case_id": item["case_id"],
            "study_id": str(item.get("study_id", "")),
            "modality": str(item.get("modality", "")),
            "label": int(item["label"]),
            "roi_name": item["roi_name"],
            "roi_voxels": int(item["roi_voxels"]),
            "tool": "pyradiomics",
            "mode": "voxel",
            "backend": "cpu",
            "mean_wall_time_s": float("nan"),
            "median_wall_time_s": float("nan"),
            "std_wall_time_s": float("nan"),
            "min_wall_time_s": float("nan"),
            "max_wall_time_s": float("nan"),
            "peak_rss_mb": float("nan"),
            "avg_rss_mb": float("nan"),
            "feature_count": 0,
            "note": str(item["note"]),
        }
        if not item["can_run"]:
            overall_rows.append(overall_row)
            continue

        image_path = Path(item["image_path"])
        mask_path = Path(item["mask_path"])
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))

        try:
            extractor_factory = lambda: SPATIAL_HELPERS._build_pyr_all(base_settings)
            extractor = extractor_factory()
            run_timings, result, rss_stats = SPATIAL_HELPERS._time_execute(
                extractor,
                image,
                mask,
                item["label"],
                args.repeats,
                0,
                voxel_based=True,
                repeat_workers=args.repeat_workers,
                build_extractor=extractor_factory,
                collect_result=True,
            )
            overall_stats = SPATIAL_HELPERS._timing_stats(run_timings, rss_stats=rss_stats)
            feature_stats = SPATIAL_HELPERS._extract_voxel_roi_stats(result, mask, item["label"])
            if not args.skip_map_save:
                SPATIAL_HELPERS._save_voxel_maps_from_result(
                    result_dict=result,
                    case_id=item["case_id"],
                    label=item["label"],
                    roi_name=item["roi_name"],
                    tool="pyradiomics",
                    out_dir_text=str(voxel_map_dir),
                    map_index_rows=map_index_rows,
                )
            if args.skip_feature_timings:
                feature_timings = {
                    feature_key: SPATIAL_HELPERS._nan_timing_stats("feature_timing_skipped")
                    for feature_key in feature_stats
                }
            else:
                feature_timings = _measure_feature_timings(
                    SPATIAL_HELPERS,
                    image=image,
                    mask=mask,
                    label=item["label"],
                    feature_keys=sorted(feature_stats.keys()),
                    repeats=args.repeats,
                    repeat_workers=args.repeat_workers,
                    voxel_based=True,
                    build_extractor=lambda feature_class, feature_name: SPATIAL_HELPERS._build_pyr_single_feature(
                        base_settings,
                        feature_class,
                        feature_name,
                    ),
                )
            overall_row.update(
                {
                    "mean_wall_time_s": float(overall_stats["mean_s"]),
                    "median_wall_time_s": float(overall_stats["median_s"]),
                    "std_wall_time_s": float(overall_stats["std_s"]),
                    "min_wall_time_s": float(overall_stats["min_s"]),
                    "max_wall_time_s": float(overall_stats["max_s"]),
                    "peak_rss_mb": float(overall_stats["peak_rss_mb"]),
                    "avg_rss_mb": float(overall_stats["avg_rss_mb"]),
                    "feature_count": int(len(feature_stats)),
                }
            )
            for feature_key in sorted(feature_stats):
                parts = SPATIAL_HELPERS._feature_parts(feature_key)
                if parts is None:
                    continue
                feature_class, feature_name = parts
                timing_stats = SPATIAL_HELPERS._coerce_timing_stats(feature_timings.get(feature_key))
                feature_entry = feature_stats[feature_key]
                per_feature_rows.append(
                    {
                        "case_id": item["case_id"],
                        "study_id": str(item.get("study_id", "")),
                        "modality": str(item.get("modality", "")),
                        "label": int(item["label"]),
                        "roi_name": item["roi_name"],
                        "roi_voxels": int(item["roi_voxels"]),
                        "tool": "pyradiomics",
                        "mode": "voxel",
                        "backend": "cpu",
                        "feature_class": feature_class,
                        "feature": feature_name,
                        "feature_key": feature_key,
                        "mean_execution_time_s": float(timing_stats["mean_s"]),
                        "median_execution_time_s": float(timing_stats["median_s"]),
                        "std_execution_time_s": float(timing_stats["std_s"]),
                        "min_execution_time_s": float(timing_stats["min_s"]),
                        "max_execution_time_s": float(timing_stats["max_s"]),
                        "peak_rss_mb": float(timing_stats["peak_rss_mb"]),
                        "avg_rss_mb": float(timing_stats["avg_rss_mb"]),
                        "value_kind": "roi_mean",
                        "value": float(feature_entry["roi_mean"]),
                        "valid_voxels": int(feature_entry["valid_voxels"]),
                        "note": SPATIAL_HELPERS._append_note(
                            str(timing_stats.get("note", "")),
                            str(feature_entry.get("note", "")),
                        ),
                    }
                )
        except Exception as exc:
            overall_row["note"] = SPATIAL_HELPERS._append_note(
                overall_row["note"],
                f"pyradiomics_voxel_error:{type(exc).__name__}",
            )

        overall_rows.append(overall_row)

    overall_fieldnames = [
        "case_id",
        "study_id",
        "modality",
        "label",
        "roi_name",
        "roi_voxels",
        "tool",
        "mode",
        "backend",
        "mean_wall_time_s",
        "median_wall_time_s",
        "std_wall_time_s",
        "min_wall_time_s",
        "max_wall_time_s",
        "peak_rss_mb",
        "avg_rss_mb",
        "feature_count",
        "note",
    ]
    per_feature_fieldnames = [
        "case_id",
        "study_id",
        "modality",
        "label",
        "roi_name",
        "roi_voxels",
        "tool",
        "mode",
        "backend",
        "feature_class",
        "feature",
        "feature_key",
        "mean_execution_time_s",
        "median_execution_time_s",
        "std_execution_time_s",
        "min_execution_time_s",
        "max_execution_time_s",
        "peak_rss_mb",
        "avg_rss_mb",
        "value_kind",
        "value",
        "valid_voxels",
        "note",
    ]
    map_index_fieldnames = [
        "case_id",
        "label",
        "roi_name",
        "tool",
        "mode",
        "feature_family",
        "feature_pyradiomics",
        "feature_tool",
        "source_key",
        "map_path",
    ]

    overall_csv = exp_dir / "pyradiomics_voxel_case_timing.csv"
    per_feature_csv = exp_dir / "pyradiomics_voxel_feature_timing.csv"
    feature_values_csv = exp_dir / "pyradiomics_voxel_feature_values_wide.csv"
    map_index_csv = exp_dir / "pyradiomics_voxel_map_index.csv"
    summary_json = exp_dir / "pyradiomics_voxel_benchmark_summary.json"

    feature_value_fieldnames, feature_value_rows = build_feature_value_matrix(per_feature_rows, "value")
    total_wall_mean, total_wall_std = sum_mean_and_std(
        overall_rows,
        "mean_wall_time_s",
        "std_wall_time_s",
    )
    total_exec_mean, total_exec_std = sum_mean_and_std(
        per_feature_rows,
        "mean_execution_time_s",
        "std_execution_time_s",
    )

    write_csv(overall_csv, overall_fieldnames, overall_rows)
    write_csv(per_feature_csv, per_feature_fieldnames, per_feature_rows)
    write_csv(feature_values_csv, feature_value_fieldnames, feature_value_rows)
    write_csv(map_index_csv, map_index_fieldnames, map_index_rows)
    write_json(
        summary_json,
        {
            "tool": "pyradiomics",
            "mode": "voxel",
            "backend": "cpu",
            "dataset_suite": args.dataset_suite,
            "repeat": int(args.repeats),
            "number_of_cases": int(len(case_ids)),
            "number_of_rois": int(len(overall_rows)),
            "labels": [int(label) for label in labels],
            "case_ids": list(case_ids),
            "modalities": sorted({str(item.get("modality", "")) for item in items if item.get("modality", "")}),
            "feature_timings_skipped": bool(args.skip_feature_timings),
            "voxel_map_save_skipped": bool(args.skip_map_save),
            "total_mean_wall_time_s": float(total_wall_mean),
            "std_wall_time_s": float(total_wall_std),
            "total_mean_execution_time_s": float(total_exec_mean),
            "std_execution_time_s": float(total_exec_std),
            "saved_feature_values_directory": relative_to_workspace(exp_dir),
            "feature_values_csv": relative_to_workspace(feature_values_csv),
            "overall_timing_csv": relative_to_workspace(overall_csv),
            "feature_timing_csv": relative_to_workspace(per_feature_csv),
            "voxel_map_directory": relative_to_workspace(voxel_map_dir),
            "voxel_map_index_csv": relative_to_workspace(map_index_csv),
        },
    )

    print(f"exp_dir={relative_to_workspace(exp_dir)}")
    print(f"summary_json={relative_to_workspace(summary_json)}")


def run_benchmark(mode: str, argv: list[str] | None = None) -> None:
    if mode == "segment":
        parser = _build_segment_parser()
        args = parser.parse_args(argv)
        _segment_run(args)
        return
    if mode == "voxel":
        parser = _build_voxel_parser()
        args = parser.parse_args(argv)
        _voxel_run(args)
        return
    raise ValueError(f"Unsupported mode: {mode}")
