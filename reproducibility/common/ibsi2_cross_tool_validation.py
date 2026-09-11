#!/usr/bin/env python3
"""Run scalar IBSI-2 reference verification across supported radiomics tools.

The runner evaluates the 18 IBSI-2 statistical features for every phase-2
configuration. Rows without finite standardized references are retained as
``not_standardized``. A tool configuration is marked ``unsupported`` when the
tool cannot represent the required response-map filter contract exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from reproducibility.common.official_benchmark_common import (
    make_experiment_dir,
    relative_to_workspace,
    write_csv,
    write_json,
)
from reproducibility.flash.benchmark_flash_ibsi2 import (
    FLASH_FIRSTORDER_BY_TAG,
    IBSI2_CONFIG_JSON,
    IBSI2_IMAGE,
    IBSI2_MASK,
    IBSI2_TEMPLATE,
    IBSI_HIGH_CONSENSUS_MAP,
    _load_165_mirp_tags,
    _load_template_stats,
    _phase2_configs,
    _prepare_case,
    _run_flash_config,
    _support_plan as flash_support_plan,
)


DEFAULT_MIRP_IMAGE = (
    WORKSPACE_ROOT
    / "reproducibility"
    / "data"
    / "ibsi"
    / "data_sets"
    / "ibsi_1_ct_radiomics_phantom"
    / "dicom"
    / "image"
)
DEFAULT_MIRP_MASK = (
    WORKSPACE_ROOT
    / "reproducibility"
    / "data"
    / "ibsi"
    / "data_sets"
    / "ibsi_1_ct_radiomics_phantom"
    / "dicom"
    / "mask"
)
DEFAULT_TOOLS = ("flash", "mirp", "pyradiomics")
DEFAULT_FLASH_BACKENDS = ("cpu",)
ALLOWED_TOOLS = frozenset(DEFAULT_TOOLS)
ALLOWED_FLASH_BACKENDS = frozenset(("cpu", "cuda"))


@dataclass
class ToolRun:
    supported: bool
    values: dict[str, float]
    sources: dict[str, str]
    note: str
    error: str = ""
    unsupported_tags: frozenset[str] = frozenset()


def _parse_csv_choices(raw: str, allowed: frozenset[str], argument: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    if not values:
        raise argparse.ArgumentTypeError(f"{argument} must contain at least one value")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"{argument} contains unsupported values: {', '.join(unknown)}"
        )
    return values


def _config_sort_key(config_id: str) -> tuple[int, str]:
    number, suffix = config_id.split(".", 1)
    return int(number), suffix


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _matching_scalar_values(frame: Any, feature_tags: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    columns = [str(column) for column in frame.columns]
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    for tag in feature_tags:
        matches = [column for column in columns if column == tag or column.endswith(f"_{tag}")]
        if not matches:
            continue
        source = min(matches, key=len)
        try:
            value = float(frame[source].iloc[0])
        except (IndexError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values[tag] = value
            sources[tag] = source
    return values, sources


def _run_flash(
    *,
    config_id: str,
    config: dict[str, Any],
    backend: str,
    num_threads: int,
) -> ToolRun:
    supported, _image_type, _custom_args, support_note = flash_support_plan(config_id, config)
    if not supported:
        return ToolRun(False, {}, {}, support_note)
    try:
        values, sources, note = _run_flash_config(
            image_path=IBSI2_IMAGE,
            mask_path=IBSI2_MASK,
            config_id=config_id,
            config=config,
            backend=backend,
            num_threads=num_threads,
        )
        return ToolRun(True, values, sources, note)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        return ToolRun(True, {}, {}, f"flash_runtime_error:{type(exc).__name__}", error)


def _mirp_transformation_settings(config: dict[str, Any]) -> Any:
    from mirp.settings.transformation_parameters import ImageTransformationSettingsClass

    filter_cfg = dict(config.get("filter") or {})
    filter_name = str(filter_cfg.get("name", "none")).lower()
    by_slice = str(config.get("image_processing", {}).get("extraction_mode", "")).upper() == "2D"
    common: dict[str, Any] = {
        "by_slice": by_slice,
        "response_map_feature_families": "statistics",
        "response_map_feature_settings": None,
        "boundary_condition": "reflect",
    }

    if filter_name == "none":
        return ImageTransformationSettingsClass(filter_kernels=None, **common)
    if filter_name == "mean":
        return ImageTransformationSettingsClass(
            filter_kernels="mean",
            mean_filter_kernel_size=int(filter_cfg.get("support_voxels", 5)),
            **common,
        )
    if filter_name == "laplacian_of_gaussian":
        return ImageTransformationSettingsClass(
            filter_kernels="laplacian_of_gaussian",
            laplacian_of_gaussian_sigma=float(filter_cfg.get("scale_sigma_star_mm", 1.5)),
            laplacian_of_gaussian_kernel_truncate=4.0,
            **common,
        )
    if filter_name == "laws":
        return ImageTransformationSettingsClass(
            filter_kernels="laws",
            laws_kernel=str(filter_cfg.get("response_map")),
            laws_compute_energy=bool(filter_cfg.get("energy_map", False)),
            laws_rotation_invariance=bool(filter_cfg.get("pseudo_rotational_invariance", False)),
            laws_delta=int(filter_cfg.get("energy_map_distance_delta_voxels", 7)),
            laws_pooling_method=str(filter_cfg.get("response_map_pooling", "max")),
            **common,
        )
    if filter_name == "gabor":
        theta_step = 180.0 / 8.0
        return ImageTransformationSettingsClass(
            filter_kernels="gabor",
            gabor_sigma=float(filter_cfg.get("scale_sigma_star_mm", 5.0)),
            gabor_gamma=float(filter_cfg.get("ellipticity_gamma", 1.5)),
            gabor_lambda=float(filter_cfg.get("wavelength_lambda_star_mm", 2.0)),
            gabor_theta=0.0,
            gabor_theta_step=theta_step,
            gabor_response=str(filter_cfg.get("response_map", "modulus")),
            gabor_rotation_invariance=bool(filter_cfg.get("pseudo_rotational_invariance", False)),
            gabor_pooling_method="mean",
            **common,
        )
    if filter_name == "daubechies_3_wavelet":
        return ImageTransformationSettingsClass(
            filter_kernels="separable_wavelet",
            separable_wavelet_families="db3",
            separable_wavelet_set=str(filter_cfg.get("wavelet_filter_combination")),
            separable_wavelet_rotation_invariance=bool(
                filter_cfg.get("pseudo_rotational_invariance", False)
            ),
            separable_wavelet_decomposition_level=int(
                filter_cfg.get("wavelet_decomposition_level", 1)
            ),
            separable_wavelet_pooling_method="mean",
            **common,
        )
    if filter_name == "simoncelli_wavelet":
        return ImageTransformationSettingsClass(
            filter_kernels="nonseparable_wavelet",
            nonseparable_wavelet_families="simoncelli",
            nonseparable_wavelet_decomposition_level=int(
                filter_cfg.get("wavelet_decomposition_level", 1)
            ),
            **common,
        )
    raise NotImplementedError(f"MIRP mapping is unavailable for filter {filter_name}")


def _run_mirp(
    *,
    config_id: str,
    config: dict[str, Any],
    feature_tags: list[str],
    image_path: Path,
    mask_path: Path,
    roi_name: str,
) -> ToolRun:
    filter_name = str((config.get("filter") or {}).get("name", "none")).lower()
    if filter_name == "riesz_transformed_simoncelli_wavelet":
        return ToolRun(False, {}, {}, "unsupported_mirp_riesz_configuration")

    try:
        from mirp.extract_features_and_images import extract_features
        from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
        from mirp.settings.general_parameters import GeneralSettingsClass
        from mirp.settings.generic import SettingsClass
        from mirp.settings.image_processing_parameters import ImagePostProcessingClass
        from mirp.settings.interpolation_parameters import (
            ImageInterpolationSettingsClass,
            MaskInterpolationSettingsClass,
        )
        from mirp.settings.perturbation_parameters import ImagePerturbationSettingsClass
        from mirp.settings.resegmentation_parameters import ResegmentationSettingsClass

        by_slice = str(config.get("image_processing", {}).get("extraction_mode", "")).upper() == "2D"
        general = GeneralSettingsClass(by_slice=by_slice, config_str=config_id)
        if by_slice:
            interpolation = ImageInterpolationSettingsClass(
                by_slice=True,
                anti_aliasing=False,
            )
        else:
            interpolation = ImageInterpolationSettingsClass(
                by_slice=False,
                spline_order=3,
                new_spacing=1.0,
                anti_aliasing=False,
            )
        base_families = "statistics" if filter_name == "none" else "none"
        feature_settings = FeatureExtractionSettingsClass(
            by_slice=by_slice,
            no_approximation=True,
            base_feature_families=base_families,
        )
        settings = SettingsClass(
            general_settings=general,
            post_process_settings=ImagePostProcessingClass(),
            img_interpolate_settings=interpolation,
            roi_interpolate_settings=MaskInterpolationSettingsClass(),
            roi_resegment_settings=ResegmentationSettingsClass(
                resegmentation_intensity_range=[-1000.0, 400.0]
            ),
            perturbation_settings=ImagePerturbationSettingsClass(crop_around_roi=False),
            img_transform_settings=_mirp_transformation_settings(config),
            feature_extr_settings=feature_settings,
        )
        extracted = extract_features(
            write_features=False,
            export_features=True,
            image=str(image_path),
            mask=str(mask_path),
            roi_name=roi_name,
            settings=settings,
            num_cpus=1,
        )
        if not extracted:
            raise RuntimeError("MIRP returned no feature table")
        values, sources = _matching_scalar_values(extracted[0], feature_tags)
        return ToolRun(True, values, sources, "mirp_ibsi2_reference_configuration")
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        return ToolRun(True, {}, {}, f"mirp_runtime_error:{type(exc).__name__}", error)


def _pyradiomics_support(config: dict[str, Any]) -> tuple[bool, str]:
    filter_cfg = dict(config.get("filter") or {})
    filter_name = str(filter_cfg.get("name", "none")).lower()
    if filter_name == "none":
        return True, "pyradiomics_original"
    if filter_name == "laplacian_of_gaussian":
        return False, "unsupported_pyradiomics_log_normalizes_across_scale_and_is_3d_only"
    if filter_name == "daubechies_3_wavelet":
        return False, "unsupported_pyradiomics_wavelet_lacks_required_rotational_pooling_contract"
    return False, f"unsupported_pyradiomics_filter:{filter_name}"


def _run_pyradiomics(
    *,
    config_id: str,
    config: dict[str, Any],
    feature_tags: list[str],
) -> ToolRun:
    supported, support_note = _pyradiomics_support(config)
    if not supported:
        return ToolRun(False, {}, {}, support_note)

    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor as PyRadiomicsFeatureExtractor
        from radiomics.firstorder import RadiomicsFirstOrder

        image, mask = _prepare_case(IBSI2_IMAGE, IBSI2_MASK, config)
        force_2d = str(config.get("image_processing", {}).get("extraction_mode", "")).upper() == "2D"
        available_features = set(RadiomicsFirstOrder.getFeatureNames())
        requested_features = [
            FLASH_FIRSTORDER_BY_TAG[tag]
            for tag in feature_tags
            if FLASH_FIRSTORDER_BY_TAG[tag] in available_features
        ]
        unsupported_tags = frozenset(
            tag
            for tag in feature_tags
            if FLASH_FIRSTORDER_BY_TAG[tag] not in available_features
        )
        extractor = PyRadiomicsFeatureExtractor(
            additionalInfo=False,
            voxelArrayShift=0,
            force2D=force_2d,
            force2Ddimension=0,
        )
        extractor.disableAllImageTypes()
        extractor.disableAllFeatures()
        extractor.enableImageTypeByName("Original")
        extractor.enableFeaturesByName(firstorder=requested_features)
        raw = extractor.execute(image, mask, label=1, voxelBased=False)

        values: dict[str, float] = {}
        sources: dict[str, str] = {}
        for tag in feature_tags:
            feature_name = FLASH_FIRSTORDER_BY_TAG[tag]
            suffix = f"_firstorder_{feature_name}"
            matches = [str(key) for key in raw if str(key).endswith(suffix)]
            if not matches:
                continue
            source = matches[0]
            value = _finite(raw[source])
            if value is not None:
                values[tag] = value
                sources[tag] = source
        return ToolRun(
            True,
            values,
            sources,
            support_note,
            unsupported_tags=unsupported_tags,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        return ToolRun(True, {}, {}, f"pyradiomics_runtime_error:{type(exc).__name__}", error)


def _evaluate_rows(
    *,
    tool: str,
    backend: str,
    config_id: str,
    config: dict[str, Any],
    template_stats: list[dict[str, str]],
    run: ToolRun,
    in_165: set[str],
    tolerance_floor: float,
) -> list[dict[str, Any]]:
    reference_values = config.get("reference_values")
    if not isinstance(reference_values, dict):
        reference_values = {}

    rows: list[dict[str, Any]] = []
    for stat in template_stats:
        tag = stat["feature_tag"]
        reference = reference_values.get(tag)
        reference_value = tolerance = None
        if isinstance(reference, dict):
            reference_value = _finite(reference.get("consensus_value"))
            tolerance = _finite(reference.get("tolerance"))

        value = _finite(run.values.get(tag))
        effective_tolerance = None
        absolute_difference = None
        passed = ""
        if reference_value is None or tolerance is None:
            status = "not_standardized"
        elif not run.supported:
            status = "unsupported"
        elif tag in run.unsupported_tags:
            status = "unsupported"
        elif run.error:
            status = "error"
        elif value is None:
            status = "error"
        else:
            effective_tolerance = max(tolerance, tolerance_floor)
            absolute_difference = abs(value - reference_value)
            status = "pass" if absolute_difference <= effective_tolerance else "fail"
            passed = "1" if status == "pass" else "0"

        rows.append(
            {
                "suite": "ibsi2",
                "tool": tool,
                "backend": backend,
                "config_id": config_id,
                "filter_name": str((config.get("filter") or {}).get("name", "none")),
                "feature_tag": tag,
                "feature_name": stat["feature_name"],
                "ibsi_identifier": stat["ibsi_identifier"],
                "in_165_feature_set": int(tag in in_165),
                "reference_value": "" if reference_value is None else reference_value,
                "reference_tolerance": "" if tolerance is None else tolerance,
                "effective_tolerance": "" if effective_tolerance is None else effective_tolerance,
                "value": "" if value is None else value,
                "absolute_difference": "" if absolute_difference is None else absolute_difference,
                "status": status,
                "passed": passed,
                "source_key": run.sources.get(tag, ""),
                "support_note": run.note,
                "error": run.error,
            }
        )
    return rows


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in rows)
    finite = [row for row in rows if row["reference_value"] != "" and row["reference_tolerance"] != ""]
    computed = counts["pass"] + counts["fail"]
    return {
        "rows_total": len(rows),
        "finite_reference_rows": len(finite),
        "pass_count": counts["pass"],
        "fail_count": counts["fail"],
        "unsupported_count": counts["unsupported"],
        "error_count": counts["error"],
        "not_standardized_count": counts["not_standardized"],
        "coverage_rate": computed / len(finite) if finite else float("nan"),
        "pass_rate_finite_including_unsupported": counts["pass"] / len(finite) if finite else float("nan"),
        "pass_rate_computed_only": counts["pass"] / computed if computed else float("nan"),
    }


def run_cross_tool_ibsi2_validation(
    *,
    tools: tuple[str, ...] = DEFAULT_TOOLS,
    flash_backends: tuple[str, ...] = DEFAULT_FLASH_BACKENDS,
    num_threads: int = 1,
    tolerance_floor: float = 1e-4,
    output_dir: Path | None = None,
    mirp_image: Path = DEFAULT_MIRP_IMAGE,
    mirp_mask: Path = DEFAULT_MIRP_MASK,
    mirp_roi_name: str = "GTV-1",
) -> tuple[Path, dict[str, Any]]:
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    if tolerance_floor < 0:
        raise ValueError("tolerance_floor must be non-negative")

    output_dir = output_dir or make_experiment_dir("common", "ibsi2_cross_tool", "mixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(IBSI2_CONFIG_JSON.read_text(encoding="utf-8"))
    configs = _phase2_configs(payload)
    template_stats = _load_template_stats(IBSI2_TEMPLATE)
    feature_tags = [row["feature_tag"] for row in template_stats]
    in_165 = _load_165_mirp_tags(IBSI_HIGH_CONSENSUS_MAP)

    from flash_radiomics import setVerbosity

    setVerbosity(60)
    validation_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []

    runners: list[tuple[str, str, Callable[[str, dict[str, Any]], ToolRun]]] = []
    if "flash" in tools:
        for backend in flash_backends:
            backend_hdf5 = output_dir / "hdf5" / f"flash_{backend}"
            backend_hdf5.mkdir(parents=True, exist_ok=True)

            def run_flash_config(
                config_id: str,
                config: dict[str, Any],
                selected_backend: str = backend,
                hdf5_dir: Path = backend_hdf5,
            ) -> ToolRun:
                os.environ["FLASH_RADIOMICS_HDF5_DIR"] = str(hdf5_dir)
                return _run_flash(
                    config_id=config_id,
                    config=config,
                    backend=selected_backend,
                    num_threads=num_threads,
                )

            runners.append(("flash", backend, run_flash_config))
    if "mirp" in tools:
        runners.append(
            (
                "mirp",
                "cpu",
                lambda config_id, config: _run_mirp(
                    config_id=config_id,
                    config=config,
                    feature_tags=feature_tags,
                    image_path=mirp_image,
                    mask_path=mirp_mask,
                    roi_name=mirp_roi_name,
                ),
            )
        )
    if "pyradiomics" in tools:
        runners.append(
            (
                "pyradiomics",
                "cpu",
                lambda config_id, config: _run_pyradiomics(
                    config_id=config_id,
                    config=config,
                    feature_tags=feature_tags,
                ),
            )
        )

    for tool, backend, runner in runners:
        for config_id in sorted(configs, key=_config_sort_key):
            config = configs[config_id]
            if config.get("reference_values_available"):
                run = runner(config_id, config)
            else:
                run = ToolRun(False, {}, {}, "no_finite_standardized_reference_values")
            rows = _evaluate_rows(
                tool=tool,
                backend=backend,
                config_id=config_id,
                config=config,
                template_stats=template_stats,
                run=run,
                in_165=in_165,
                tolerance_floor=tolerance_floor,
            )
            validation_rows.extend(rows)
            counts = Counter(str(row["status"]) for row in rows)
            config_rows.append(
                {
                    "tool": tool,
                    "backend": backend,
                    "config_id": config_id,
                    "filter_name": str((config.get("filter") or {}).get("name", "none")),
                    "reference_values_available": int(bool(config.get("reference_values_available"))),
                    "supported": int(run.supported),
                    "pass_count": counts["pass"],
                    "fail_count": counts["fail"],
                    "unsupported_count": counts["unsupported"],
                    "error_count": counts["error"],
                    "not_standardized_count": counts["not_standardized"],
                    "support_note": run.note,
                    "error": run.error,
                }
            )
            print(
                f"tool={tool} backend={backend} config={config_id} "
                f"pass={counts['pass']} fail={counts['fail']} "
                f"unsupported={counts['unsupported']} error={counts['error']}",
                flush=True,
            )

    validation_fields = [
        "suite",
        "tool",
        "backend",
        "config_id",
        "filter_name",
        "feature_tag",
        "feature_name",
        "ibsi_identifier",
        "in_165_feature_set",
        "reference_value",
        "reference_tolerance",
        "effective_tolerance",
        "value",
        "absolute_difference",
        "status",
        "passed",
        "source_key",
        "support_note",
        "error",
    ]
    config_fields = [
        "tool",
        "backend",
        "config_id",
        "filter_name",
        "reference_values_available",
        "supported",
        "pass_count",
        "fail_count",
        "unsupported_count",
        "error_count",
        "not_standardized_count",
        "support_note",
        "error",
    ]
    validation_csv = output_dir / "ibsi2_cross_tool_validation.csv"
    config_csv = output_dir / "ibsi2_cross_tool_config_status.csv"
    summary_json = output_dir / "ibsi2_cross_tool_validation_summary.json"
    write_csv(validation_csv, validation_fields, validation_rows)
    write_csv(config_csv, config_fields, config_rows)

    groups: dict[str, dict[str, Any]] = {}
    for tool, backend, _runner in runners:
        key = f"{tool}:{backend}"
        groups[key] = _summarize_group(
            [
                row
                for row in validation_rows
                if row["tool"] == tool and row["backend"] == backend
            ]
        )
    finite_reference_rows = sum(
        1
        for config in configs.values()
        for tag in feature_tags
        if isinstance(config.get("reference_values"), dict)
        and isinstance(config["reference_values"].get(tag), dict)
        and _finite(config["reference_values"][tag].get("consensus_value")) is not None
        and _finite(config["reference_values"][tag].get("tolerance")) is not None
    )
    summary = {
        "suite": "ibsi2",
        "mode": "scalar",
        "tools": list(tools),
        "flash_backends": list(flash_backends),
        "num_threads": num_threads,
        "tolerance_floor": tolerance_floor,
        "config_json": relative_to_workspace(IBSI2_CONFIG_JSON),
        "template": relative_to_workspace(IBSI2_TEMPLATE),
        "flash_image": relative_to_workspace(IBSI2_IMAGE),
        "flash_mask": relative_to_workspace(IBSI2_MASK),
        "mirp_image": str(mirp_image),
        "mirp_mask": str(mirp_mask),
        "mirp_roi_name": mirp_roi_name,
        "configuration_count": len(configs),
        "finite_reference_configuration_count": sum(
            bool(config.get("reference_values_available")) for config in configs.values()
        ),
        "statistical_feature_count": len(template_stats),
        "finite_reference_rows_per_tool_backend": finite_reference_rows,
        "all_template_stats_in_165": all(tag in in_165 for tag in feature_tags),
        "groups": groups,
        "runtime_error_count": sum(group["error_count"] for group in groups.values()),
        "validation_csv": relative_to_workspace(validation_csv),
        "config_status_csv": relative_to_workspace(config_csv),
    }
    write_json(summary_json, summary)
    print(f"out_dir={relative_to_workspace(output_dir)}")
    print(f"summary_json={relative_to_workspace(summary_json)}")
    return output_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scalar IBSI-2 verification for Flash-Radiomics, MIRP, and PyRadiomics."
    )
    parser.add_argument("--tools", default=",".join(DEFAULT_TOOLS))
    parser.add_argument("--flash-backends", default=",".join(DEFAULT_FLASH_BACKENDS))
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--tolerance-floor", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mirp-image", type=Path, default=DEFAULT_MIRP_IMAGE)
    parser.add_argument("--mirp-mask", type=Path, default=DEFAULT_MIRP_MASK)
    parser.add_argument("--mirp-roi-name", default="GTV-1")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        tools = _parse_csv_choices(args.tools, ALLOWED_TOOLS, "--tools")
        flash_backends = _parse_csv_choices(
            args.flash_backends,
            ALLOWED_FLASH_BACKENDS,
            "--flash-backends",
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    _output_dir, summary = run_cross_tool_ibsi2_validation(
        tools=tools,
        flash_backends=flash_backends,
        num_threads=args.num_threads,
        tolerance_floor=args.tolerance_floor,
        output_dir=args.out_dir,
        mirp_image=args.mirp_image,
        mirp_mask=args.mirp_mask,
        mirp_roi_name=args.mirp_roi_name,
    )
    if summary["runtime_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
