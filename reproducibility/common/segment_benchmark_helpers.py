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

SEGMENT_FEATURE_CLASSES = (
    "firstorder",
    "shape",
    "shape2D",
    "glcm",
    "glrlm",
    "glszm",
    "gldzm",
    "gldm",
    "ngtdm",
    "ih",
    "ivh",
)

SEGMENT_BASE_FEATURE_CLASSES = tuple(
    feature_class for feature_class in SEGMENT_FEATURE_CLASSES if feature_class != "shape2D"
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


def _base_settings(
    bin_width: float,
    distances: list[int],
    force2d: bool,
    force2d_dimension: int,
    segment_class_workers: int,
) -> dict:
    return {
        "additionalInfo": False,
        "binWidth": float(bin_width),
        "distances": list(distances),
        "gldm_a": 0,
        "force2D": bool(force2d),
        "force2Ddimension": int(force2d_dimension),
        "segmentClassWorkers": int(segment_class_workers),
    }


def _shape2d_compatibility(
    mask: sitk.Image,
    label_stats: sitk.LabelShapeStatisticsImageFilter,
    label: int,
    force2d: bool,
    force2d_dimension: int,
) -> tuple[bool, str]:
    ndim = int(mask.GetDimension())
    if ndim == 2:
        return True, ""
    if ndim != 3:
        return False, "shape2D_unsupported_ndim"
    if not force2d:
        return False, "shape2D_requires_force2D"
    if force2d_dimension < 0 or force2d_dimension > 2:
        return False, "shape2D_invalid_force2Ddimension"
    if not label_stats.HasLabel(int(label)):
        return False, "shape2D_label_missing"

    bbox = label_stats.GetBoundingBox(int(label))
    out_of_plane_size = int(bbox[3 + int(force2d_dimension)])
    if out_of_plane_size != 1:
        return False, "shape2D_out_of_plane_size_gt1"
    return True, ""


def _feature_classes_for_roi(
    mask: sitk.Image,
    label_stats: sitk.LabelShapeStatisticsImageFilter,
    label: int,
    include_shape2d: bool,
    force2d: bool,
    force2d_dimension: int,
) -> tuple[tuple[str, ...], str]:
    feature_classes = list(SEGMENT_BASE_FEATURE_CLASSES)
    note = ""
    if include_shape2d:
        shape2d_ok, shape2d_note = _shape2d_compatibility(
            mask, label_stats, label, force2d, force2d_dimension
        )
        if shape2d_ok:
            feature_classes.append("shape2D")
        else:
            note = shape2d_note
    else:
        note = "shape2D_not_requested"
    return tuple(feature_classes), note


def _build_flash_all(
    settings: dict,
    feature_classes: tuple[str, ...],
    backend: str,
    num_threads: int = 0,
) -> FlashExtractor:
    kwargs = {"backend": backend}
    if int(num_threads) > 0:
        kwargs["num_threads"] = int(num_threads)
    extractor = FlashExtractor(**kwargs)
    extractor.disableAllFeatures()
    for feature_class in feature_classes:
        extractor.enableFeatureClassByName(feature_class, True)
    extractor.settings.update(settings)
    return extractor


def _build_pyr_all(settings: dict, feature_classes: tuple[str, ...]) -> Any:
    extractor_cls = _pyradiomics_extractor_cls()
    extractor = extractor_cls()
    extractor.disableAllFeatures()
    for feature_class in feature_classes:
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


def _to_scalar(value) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    return None


def _extract_scalar_original_features(result_dict: dict) -> dict[str, float]:
    features = {}
    for key, value in result_dict.items():
        if not key.startswith("original_"):
            continue
        scalar = _to_scalar(value)
        if scalar is not None:
            features[key] = scalar
    return features


def _feature_parts(feature_key: str) -> tuple[str, str] | None:
    if not feature_key.startswith("original_"):
        return None
    suffix = feature_key[len("original_") :]
    for feature_class in SEGMENT_FEATURE_CLASSES:
        prefix = f"{feature_class}_"
        if suffix.startswith(prefix):
            return feature_class, suffix[len(prefix) :]
    return None
