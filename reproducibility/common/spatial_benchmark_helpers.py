"""Shared helpers for spatial (voxel-wise) radiomics benchmarks."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import threading
import time
import types
from typing import Any

import numpy as np
import psutil
import SimpleITK as sitk

from flash_radiomics.featureextractor import RadiomicsFeatureExtractor as FlashExtractor


ROI_NAME_BY_LABEL = {
    1: "Segment-1",
    2: "Segment-2",
}

IBSI_VALIDATION_MODALITIES = ("CT", "PET", "MR_T1")
DEFAULT_IBSI_VALIDATION_STUDY_DIR = Path(
    "reproducibility/data/ibsi/data_sets/ibsi_validation/nifti/STS_001"
)

VOXEL_FEATURE_CLASSES = (
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)

TIMING_STAT_KEYS = (
    "mean_s",
    "median_s",
    "std_s",
    "min_s",
    "max_s",
    "peak_rss_mb",
    "avg_rss_mb",
)


def _clear_stale_radiomics_bytecode():
    """Clear stale bytecode caches for the radiomics package.

    When pyradiomics is installed in editable mode and the source tree is
    relocated, existing .pyc files may embed the old absolute path as their
    source reference.  Python's SourceFileLoader then attempts to read the
    source at that stale path and raises FileNotFoundError.  Deleting the
    __pycache__ directory forces Python to recompile from the actual source
    location on the next import.
    """
    import importlib
    import importlib.util
    import shutil

    # Clear cached modules and bytecode so editable installs reload cleanly.
    for key in [k for k in sys.modules if k == "radiomics" or k.startswith("radiomics.")]:
        del sys.modules[key]

    importlib.invalidate_caches()

    for entry in sys.path:
        pkg_dir = Path(entry) / "radiomics"
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").is_file():
            cache_dir = pkg_dir / "__pycache__"
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)

    # PEP 660 installs may be redirected outside visible sys.path entries.
    try:
        spec = importlib.util.find_spec("radiomics")
        if spec and spec.origin:
            pkg_dir = Path(spec.origin).parent
            if pkg_dir.is_dir():
                cache_dir = pkg_dir / "__pycache__"
                if cache_dir.is_dir():
                    shutil.rmtree(cache_dir, ignore_errors=True)
    except (FileNotFoundError, ImportError, ValueError):
        pass


def _patch_numpy_delete_for_pyradiomics():
    original_delete = np.delete

    def safe_delete(arr, obj, axis=None):
        return original_delete(np.asarray(arr), obj, axis=axis)

    np.delete = safe_delete


def _patch_numpy_sum_for_pyradiomics():
    original_sum = np.sum

    def safe_sum(arr, *args, **kwargs):
        return original_sum(np.asarray(arr), *args, **kwargs)

    np.sum = safe_sum


def _patch_numpy_ufunc_config_for_pyradiomics():
    if "numpy.core._ufunc_config" in sys.modules:
        return
    module = types.ModuleType("numpy.core._ufunc_config")

    @contextlib.contextmanager
    def _no_nep50_warning():
        yield

    module._no_nep50_warning = _no_nep50_warning
    sys.modules["numpy.core._ufunc_config"] = module


def _patch_pyradiomics_cmatrix_outputs():
    from radiomics import cMatrices

    if cMatrices is None:
        raise RuntimeError("PyRadiomics cMatrices extension is unavailable")

    def _coerce(value):
        if isinstance(value, np.ndarray):
            return np.asarray(value)
        if isinstance(value, (tuple, list)):
            return type(value)(_coerce(v) for v in value)
        return value

    for fn_name in (
        "calculate_glcm",
        "calculate_glrlm",
        "calculate_glszm",
        "calculate_gldm",
        "calculate_ngtdm",
    ):
        if not hasattr(cMatrices, fn_name):
            continue
        original = getattr(cMatrices, fn_name)

        def _wrap(func):
            def wrapped(*args, **kwargs):
                return _coerce(func(*args, **kwargs))

            return wrapped

        setattr(cMatrices, fn_name, _wrap(original))


def _pyradiomics_extractor_cls():
    from radiomics.featureextractor import RadiomicsFeatureExtractor

    return RadiomicsFeatureExtractor


def _apply_runtime_patches():
    _clear_stale_radiomics_bytecode()
    _patch_numpy_delete_for_pyradiomics()
    _patch_numpy_sum_for_pyradiomics()
    _patch_numpy_ufunc_config_for_pyradiomics()
    _patch_pyradiomics_cmatrix_outputs()
    try:
        from radiomics import setVerbosity

        setVerbosity(logging.ERROR)
    except Exception:
        pass
    logging.disable(logging.CRITICAL)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    logging.getLogger("radiomics.featureextractor").setLevel(logging.ERROR)
    logging.getLogger("radiomics.imageoperations").setLevel(logging.ERROR)


def _append_note(existing: str, note: str) -> str:
    if not note:
        return existing
    if not existing:
        return note
    return f"{existing}|{note}"


def _parse_comma_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_labels(labels_arg: str) -> list[int]:
    labels = []
    for label in _parse_comma_ints(labels_arg):
        if label in ROI_NAME_BY_LABEL:
            labels.append(label)
    labels = sorted(set(labels))
    if not labels:
        raise ValueError("No valid labels selected. Allowed labels: 1,2")
    return labels


def _parse_distances(value: str) -> list[int]:
    distances = [int(v) for v in _parse_comma_ints(value) if int(v) > 0]
    if not distances:
        raise ValueError("distances must contain at least one positive integer")
    return sorted(set(distances))


def _resolve_case_ids(dataset_dir: Path, case_ids_arg: str, max_cases: int) -> list[str]:
    if case_ids_arg.strip():
        case_ids = [v.strip() for v in case_ids_arg.split(",") if v.strip()]
    else:
        case_ids = []
        for case_dir in sorted(dataset_dir.glob("case_*")):
            if not case_dir.is_dir():
                continue
            image_path = case_dir / "imaging.nii.gz"
            mask_path = case_dir / "segmentation.nii.gz"
            if image_path.exists() and mask_path.exists():
                case_ids.append(case_dir.name)
    if max_cases > 0:
        case_ids = case_ids[:max_cases]
    return case_ids


def _resolve_ibsi_validation_voxel_cases(
    study_dir: Path,
    modalities_arg: str = "",
    max_cases: int = 0,
) -> list[dict]:
    if not study_dir.exists():
        raise FileNotFoundError(f"IBSI validation study directory not found: {study_dir}")

    if modalities_arg.strip():
        requested = [value.strip() for value in modalities_arg.split(",") if value.strip()]
    else:
        requested = list(IBSI_VALIDATION_MODALITIES)

    modality_lookup = {value.upper(): value for value in IBSI_VALIDATION_MODALITIES}
    modalities: list[str] = []
    for raw_modality in requested:
        modality = modality_lookup.get(raw_modality.upper())
        if modality is None:
            raise ValueError(
                "Unsupported IBSI validation modality "
                f"{raw_modality!r}. Allowed: {','.join(IBSI_VALIDATION_MODALITIES)}"
            )
        if modality not in modalities:
            modalities.append(modality)

    if max_cases > 0:
        modalities = modalities[: int(max_cases)]

    cases: list[dict] = []
    study_id = study_dir.name
    for modality in modalities:
        image_path = study_dir / f"{modality}_image.nii.gz"
        mask_path = study_dir / f"{modality}_mask.nii.gz"
        cases.append(
            {
                "case_id": f"{study_id}_{modality}",
                "study_id": study_id,
                "modality": modality,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    return cases


def _base_settings(
    kernel_radius: int,
    masked_kernel: bool,
    bin_width: float,
    distances: list[int],
    voxel_batch: int,
) -> dict:
    return {
        "additionalInfo": False,
        "kernelRadius": int(kernel_radius),
        "maskedKernel": bool(masked_kernel),
        "binWidth": float(bin_width),
        "distances": list(distances),
        "gldm_a": 0,
        "initValue": float("nan"),
        "voxelBatch": int(voxel_batch),
    }


def _build_flash_all(settings: dict, backend: str, num_threads: int = 0) -> FlashExtractor:
    kwargs = {"backend": backend}
    if int(num_threads) > 0:
        kwargs["num_threads"] = int(num_threads)
    extractor = FlashExtractor(**kwargs)
    extractor.disableAllFeatures()
    for feature_class in VOXEL_FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class, True)
    extractor.settings.update(settings)
    return extractor


def _build_pyr_all(settings: dict) -> Any:
    extractor_cls = _pyradiomics_extractor_cls()
    extractor = extractor_cls()
    extractor.disableAllFeatures()
    for feature_class in VOXEL_FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class, True)
    extractor.settings.update(settings)
    return extractor


def _build_flash_single_feature(
    settings: dict,
    feature_class: str,
    feature_name: str,
    backend: str,
    num_threads: int = 0,
) -> FlashExtractor:
    kwargs = {"backend": backend}
    if int(num_threads) > 0:
        kwargs["num_threads"] = int(num_threads)
    extractor = FlashExtractor(**kwargs)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**{feature_class: [feature_name]})
    extractor.settings.update(settings)
    return extractor


def _build_pyr_single_feature(settings: dict, feature_class: str, feature_name: str) -> Any:
    extractor_cls = _pyradiomics_extractor_cls()
    extractor = extractor_cls()
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**{feature_class: [feature_name]})
    extractor.settings.update(settings)
    return extractor


class _RssSampler:
    def __init__(self, sample_interval_s: float = 0.01):
        self.sample_interval_s = float(max(0.001, sample_interval_s))
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(int(self._proc.memory_info().rss))
            except Exception:
                pass
            self._stop.wait(self.sample_interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()
        if not self._samples:
            try:
                self._samples.append(int(self._proc.memory_info().rss))
            except Exception:
                self._samples.append(0)
        mb = 1024.0 * 1024.0
        peak_rss_mb = float(max(self._samples) / mb)
        avg_rss_mb = float(sum(self._samples) / len(self._samples) / mb)
        return {"peak_rss_mb": peak_rss_mb, "avg_rss_mb": avg_rss_mb}


def _time_execute(
    extractor,
    image,
    mask,
    label: int,
    repeats: int,
    warmup: int,
    voxel_based: bool,
    repeat_workers: int = 0,
    build_extractor=None,
    log_repeat_progress: bool = False,
    progress_prefix: str = "",
    collect_result: bool = True,
):
    n_repeats = max(1, int(repeats))
    workers = int(repeat_workers)
    if workers <= 0:
        workers = n_repeats
    workers = max(1, min(workers, n_repeats))
    prefix = str(progress_prefix).strip()
    if prefix:
        prefix = f"{prefix} "

    def _execute(local_extractor):
        mode_option = (
            {"spatialMode": bool(voxel_based)}
            if isinstance(local_extractor, FlashExtractor)
            else {"voxelBased": bool(voxel_based)}
        )
        return local_extractor.execute(
            image,
            mask,
            label=int(label),
            **mode_option,
        )

    def _run_once(local_extractor):
        start = time.perf_counter()
        local_result = _execute(local_extractor)
        end = time.perf_counter()
        return (end - start), local_result

    def _run_once_duration(local_extractor):
        start = time.perf_counter()
        _execute(local_extractor)
        end = time.perf_counter()
        return end - start

    timings: list[float] = []
    result = None
    rss_sampler = _RssSampler()
    rss_sampler.start()
    try:
        if workers == 1 or n_repeats == 1:
            for _ in range(max(0, int(warmup))):
                _execute(extractor)
            if log_repeat_progress and n_repeats > 1:
                print(f"{prefix}repeat_mode=sequential repeats={n_repeats} workers=1")
            for repeat_idx in range(n_repeats):
                local_extractor = extractor
                if repeat_idx > 0 and callable(build_extractor):
                    local_extractor = build_extractor()
                duration, local_result = _run_once(local_extractor)
                timings.append(float(duration))
                if result is None:
                    result = local_result
        else:
            if not callable(build_extractor):
                raise ValueError(
                    "build_extractor is required when repeat_workers > 1 to run repeats concurrently."
                )
            if log_repeat_progress:
                print(f"{prefix}repeat_mode=multiprocess repeats={n_repeats} workers={workers}")
            try:
                mp_context = mp.get_context("fork")
            except ValueError:
                if log_repeat_progress:
                    print(
                        f"{prefix}repeat_mode_fallback=sequential "
                        "reason=fork_context_unavailable"
                    )
                for repeat_idx in range(n_repeats):
                    local_extractor = build_extractor()
                    for _ in range(max(0, int(warmup))):
                        _execute(local_extractor)
                    duration, local_result = _run_once(local_extractor)
                    timings.append(float(duration))
                    if log_repeat_progress:
                        print(
                            f"{prefix}repeat_done={repeat_idx + 1}/{n_repeats} "
                            f"duration_s={duration:.6f}"
                        )
                    if result is None and collect_result:
                        result = local_result
            else:
                queue = mp_context.Queue()
                pending = list(range(1, n_repeats + 1))
                active: dict[int, mp.Process] = {}

                def _worker(repeat_idx: int):
                    try:
                        local_extractor = build_extractor()
                        for _ in range(max(0, int(warmup))):
                            _execute(local_extractor)
                        duration = _run_once_duration(local_extractor)
                        queue.put((repeat_idx, float(duration), ""))
                    except Exception as exc:
                        queue.put((repeat_idx, float("nan"), f"{type(exc).__name__}: {exc}"))

                try:
                    while pending or active:
                        while pending and len(active) < workers:
                            repeat_idx = pending.pop(0)
                            process = mp_context.Process(target=_worker, args=(repeat_idx,))
                            process.start()
                            active[repeat_idx] = process

                        repeat_idx, duration, error_text = queue.get()
                        process = active.pop(int(repeat_idx), None)
                        if process is not None:
                            process.join()

                        if error_text:
                            raise RuntimeError(
                                f"Repeat worker {repeat_idx}/{n_repeats} failed: {error_text}"
                            )

                        timings.append(float(duration))
                        if log_repeat_progress:
                            print(
                                f"{prefix}repeat_done={repeat_idx}/{n_repeats} "
                                f"duration_s={duration:.6f}"
                            )
                finally:
                    for process in active.values():
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=5)
                    queue.close()
                    queue.join_thread()

                if collect_result and result is None:
                    local_extractor = build_extractor()
                    _, result = _run_once(local_extractor)
    finally:
        rss_stats = rss_sampler.stop()

    return timings, result, rss_stats


def _timing_stats(values: list[float], rss_stats: dict | None = None) -> dict[str, float]:
    rss_stats = rss_stats or {}
    return {
        "mean_s": float(statistics.mean(values)),
        "median_s": float(statistics.median(values)),
        "std_s": float(statistics.pstdev(values)),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
        "peak_rss_mb": float(rss_stats.get("peak_rss_mb", float("nan"))),
        "avg_rss_mb": float(rss_stats.get("avg_rss_mb", float("nan"))),
    }


def _nan_timing_stats(note: str = "") -> dict:
    return {
        "mean_s": float("nan"),
        "median_s": float("nan"),
        "std_s": float("nan"),
        "min_s": float("nan"),
        "max_s": float("nan"),
        "peak_rss_mb": float("nan"),
        "avg_rss_mb": float("nan"),
        "note": str(note),
    }


def _coerce_timing_stats(raw: dict | None) -> dict:
    stats = _nan_timing_stats()
    if not isinstance(raw, dict):
        return stats
    for key in TIMING_STAT_KEYS:
        stats[key] = float(raw.get(key, float("nan")))
    stats["note"] = str(raw.get("note", ""))
    return stats


def _feature_parts(feature_key: str) -> tuple[str, str] | None:
    if not feature_key.startswith("original_"):
        return None
    suffix = feature_key[len("original_") :]
    for feature_class in VOXEL_FEATURE_CLASSES:
        prefix = f"{feature_class}_"
        if suffix.startswith(prefix):
            return feature_class, suffix[len(prefix) :]
    return None


def _roi_mask_in_reference(mask_image: sitk.Image, label: int, reference: sitk.Image) -> np.ndarray:
    roi = sitk.Equal(mask_image, int(label))
    roi_resampled = sitk.Resample(
        roi,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return sitk.GetArrayFromImage(roi_resampled).astype(bool, copy=False)


def _extract_voxel_roi_stats(result_dict: dict, mask: sitk.Image, label: int) -> dict[str, dict[str, float]]:
    feature_stats: dict[str, dict[str, float]] = {}
    for key, value in result_dict.items():
        if not key.startswith("original_"):
            continue
        if not isinstance(value, sitk.Image):
            continue

        arr = sitk.GetArrayFromImage(value).astype(np.float64, copy=False)
        roi_mask = _roi_mask_in_reference(mask, label, value)
        if roi_mask.shape != arr.shape:
            feature_stats[key] = {
                "roi_mean": float("nan"),
                "valid_voxels": 0,
                "note": "roi_shape_mismatch",
            }
            continue

        valid = roi_mask & np.isfinite(arr)
        valid_voxels = int(np.sum(valid))
        if valid_voxels <= 0:
            feature_stats[key] = {
                "roi_mean": float("nan"),
                "valid_voxels": 0,
                "note": "no_finite_roi_voxels",
            }
            continue

        feature_stats[key] = {
            "roi_mean": float(np.mean(arr[valid])),
            "valid_voxels": int(valid_voxels),
            "note": "",
        }
    return feature_stats


def _sanitize_filename(raw: str) -> str:
    kept: list[str] = []
    for ch in str(raw):
        if ch.isalnum() or ch in {"_", "-"}:
            kept.append(ch)
        else:
            kept.append("_")
    name = "".join(kept).strip("_")
    return name or "feature_map"


def _save_voxel_maps_from_result(
    *,
    result_dict: dict,
    case_id: str,
    label: int,
    roi_name: str,
    tool: str,
    out_dir_text: str,
    map_index_rows: list[dict],
) -> None:
    if not isinstance(result_dict, dict):
        return
    if not str(out_dir_text).strip():
        return
    out_root = Path(out_dir_text) / str(case_id) / f"label_{int(label)}"
    out_root.mkdir(parents=True, exist_ok=True)

    for key, value in result_dict.items():
        if not key.startswith("original_"):
            continue
        if not isinstance(value, sitk.Image):
            continue
        parts = _feature_parts(key)
        if parts is None:
            continue
        feature_class, feature_name = parts
        safe_name = _sanitize_filename(f"{feature_class}__{feature_name}")
        map_path = out_root / f"{safe_name}.nii.gz"
        sitk.WriteImage(value, str(map_path), True)
        map_index_rows.append(
            {
                "case_id": str(case_id),
                "label": int(label),
                "roi_name": str(roi_name),
                "tool": str(tool),
                "mode": "voxel",
                "feature_family": str(feature_class),
                "feature_pyradiomics": str(feature_name),
                "feature_tool": str(feature_name),
                "source_key": str(key),
                "map_path": str(map_path),
            }
        )
