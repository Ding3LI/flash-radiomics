"""Persistence for the case-level ``flashrad_hdf5`` container.

The layout follows the feature-cache files produced by
``spatial_phenotyping.write_flashrad_hdf5``. Scalar and spatial extractions
are two optional payloads in the same ROI group; feature selection changes
dataset lengths, not paths or nesting.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import h5py
import numpy as np
import SimpleITK as sitk


SCHEMA_NAME = "flashrad_hdf5"
SCHEMA_VERSION = "1.0"
FORMAT_NAME = "Flash-Radiomics HDF5"
FORMAT_VERSION = SCHEMA_VERSION


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _numeric_feature_items(feature_vector: Mapping[str, Any]) -> list[tuple[str, float]]:
    return [
        (str(name), float(value))
        for name, value in feature_vector.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and not str(name).startswith("diagnostics_")
    ]


def _spatial_feature_items(
    feature_vector: Mapping[str, Any],
) -> list[tuple[str, sitk.Image]]:
    return [
        (str(name), value)
        for name, value in feature_vector.items()
        if isinstance(value, sitk.Image)
        and not str(name).startswith("diagnostics_")
    ]


def _metadata_outputs(feature_vector: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(name): _jsonable(value)
        for name, value in feature_vector.items()
        if str(name).startswith("diagnostics_")
        or not isinstance(value, (int, float, np.integer, np.floating, sitk.Image))
    }


def _replace_dataset(group: h5py.Group, name: str, **kwargs) -> h5py.Dataset:
    if name in group:
        del group[name]
    return group.create_dataset(name, **kwargs)


def _compressed_dataset_options(
    array: np.ndarray,
    compression: str | None,
) -> dict[str, Any]:
    if compression is None or array.size == 0:
        return {}
    return {"compression": compression, "shuffle": True}


def _set_geometry_attributes(
    dataset: h5py.Dataset,
    reference: sitk.Image,
    *,
    shape_convention: str,
) -> None:
    dataset.attrs["shape_convention"] = shape_convention
    dataset.attrs["spatial_grid"] = "roi_crop_grid"
    dataset.attrs["array_axis_order"] = (
        "z,y,x" if reference.GetDimension() == 3 else "y,x"
    )
    dataset.attrs["spacing_xyz"] = np.asarray(reference.GetSpacing(), dtype=np.float64)
    dataset.attrs["origin_xyz"] = np.asarray(reference.GetOrigin(), dtype=np.float64)
    dataset.attrs["direction"] = np.asarray(reference.GetDirection(), dtype=np.float64)
    dataset.attrs["size_xyz"] = np.asarray(reference.GetSize(), dtype=np.int64)


def _same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetDimension() == right.GetDimension()
        and left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), rtol=0.0, atol=1e-8)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), rtol=0.0, atol=1e-8)
        and np.allclose(
            left.GetDirection(),
            right.GetDirection(),
            rtol=0.0,
            atol=1e-8,
        )
    )


def _resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    interpolator: int,
    output_pixel_type: int,
) -> sitk.Image:
    if _same_geometry(image, reference):
        return sitk.Cast(image, output_pixel_type)
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        0.0,
        output_pixel_type,
    )


def _channel_first_array(image: sitk.Image, dtype: np.dtype) -> np.ndarray:
    array = sitk.GetArrayFromImage(image).astype(dtype, copy=False)
    return array[np.newaxis, ...]


def _roi_crop_reference(image: sitk.Image, feature_mask: sitk.Image) -> sitk.Image:
    binary_mask = sitk.Cast(sitk.NotEqual(feature_mask, 0), sitk.sitkUInt8)
    binary_mask = _resample_to_reference(
        binary_mask,
        image,
        interpolator=sitk.sitkNearestNeighbor,
        output_pixel_type=sitk.sitkUInt8,
    )
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(binary_mask)
    if not statistics.HasLabel(1):
        return image
    bounding_box = statistics.GetBoundingBox(1)
    dimension = image.GetDimension()
    index = [int(value) for value in bounding_box[:dimension]]
    size = [int(value) for value in bounding_box[dimension:]]
    return sitk.RegionOfInterest(image, size=size, index=index)


def _write_common_grid(
    roi_group: h5py.Group,
    *,
    image: sitk.Image,
    feature_mask: sitk.Image,
    reference: sitk.Image,
    compression: str | None,
) -> None:
    aligned_image = _resample_to_reference(
        image,
        reference,
        interpolator=sitk.sitkLinear,
        output_pixel_type=sitk.sitkFloat32,
    )
    binary_mask = sitk.Cast(sitk.NotEqual(feature_mask, 0), sitk.sitkUInt8)
    aligned_mask = _resample_to_reference(
        binary_mask,
        reference,
        interpolator=sitk.sitkNearestNeighbor,
        output_pixel_type=sitk.sitkUInt8,
    )
    image_array = _channel_first_array(aligned_image, np.float32)
    mask_array = _channel_first_array(aligned_mask, np.uint8)
    convention = (
        "channel_first_3d_crop"
        if reference.GetDimension() == 3
        else "channel_first_2d_crop"
    )

    image_dataset = _replace_dataset(
        roi_group,
        "image",
        data=image_array,
        **_compressed_dataset_options(image_array, compression),
    )
    mask_dataset = _replace_dataset(
        roi_group,
        "mask",
        data=mask_array,
        **_compressed_dataset_options(mask_array, compression),
    )
    _set_geometry_attributes(image_dataset, reference, shape_convention=convention)
    _set_geometry_attributes(mask_dataset, reference, shape_convention=convention)
    roi_group.attrs["crop_shape"] = json.dumps(list(image_array.shape[1:]))


def _write_scalar_payload(
    roi_group: h5py.Group,
    feature_vector: Mapping[str, Any],
    *,
    string_dtype: np.dtype,
) -> None:
    if "scalar_features" in roi_group:
        del roi_group["scalar_features"]
    scalar_group = roi_group.require_group("scalar_features")
    numeric_items = _numeric_feature_items(feature_vector)
    scalar_group.create_dataset(
        "names",
        data=np.asarray([name for name, _ in numeric_items], dtype=object),
        dtype=string_dtype,
    )
    scalar_group.create_dataset(
        "values",
        data=np.asarray([value for _, value in numeric_items], dtype=np.float64),
    )
    scalar_group.attrs["feature_count"] = len(numeric_items)
    scalar_group.attrs["name_dataset"] = scalar_group["names"].name
    scalar_group.attrs["value_dataset"] = scalar_group["values"].name


def _write_spatial_payload(
    roi_group: h5py.Group,
    feature_items: list[tuple[str, sitk.Image]],
    *,
    reference: sitk.Image,
    string_dtype: np.dtype,
    compression: str | None,
) -> None:
    feature_arrays = []
    for _, feature_map in feature_items:
        aligned_map = _resample_to_reference(
            feature_map,
            reference,
            interpolator=sitk.sitkLinear,
            output_pixel_type=sitk.sitkFloat32,
        )
        feature_arrays.append(
            sitk.GetArrayFromImage(aligned_map).astype(np.float32, copy=False)
        )

    spatial_shape = tuple(reversed(reference.GetSize()))
    feature_maps = (
        np.stack(feature_arrays, axis=0).astype(np.float32, copy=False)
        if feature_arrays
        else np.empty((0, *spatial_shape), dtype=np.float32)
    )
    feature_names = np.asarray([name for name, _ in feature_items], dtype=object)
    maps_dataset = _replace_dataset(
        roi_group,
        "feature_maps",
        data=feature_maps,
        **_compressed_dataset_options(feature_maps, compression),
    )
    names_dataset = _replace_dataset(
        roi_group,
        "feature_names",
        data=feature_names,
        dtype=string_dtype,
    )
    convention = (
        "channel_first_3d_crop"
        if reference.GetDimension() == 3
        else "channel_first_2d_crop"
    )
    _set_geometry_attributes(maps_dataset, reference, shape_convention=convention)
    maps_dataset.attrs["feature_axis"] = 0
    maps_dataset.attrs["feature_count"] = len(feature_items)
    maps_dataset.attrs["feature_name_dataset"] = names_dataset.name
    roi_group.attrs["feature_channel_count"] = len(feature_items)


def _read_string_list_attribute(handle: h5py.File, name: str) -> set[str]:
    raw_value = handle.attrs.get(name)
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    if raw_value is not None:
        try:
            decoded = json.loads(str(raw_value))
            if isinstance(decoded, list):
                return {str(value) for value in decoded}
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return set()


def _discover_stored_modes(handle: h5py.File) -> set[str]:
    modes = _read_string_list_attribute(handle, "modes_json")
    legacy_value = str(handle.attrs.get("mode", "")).strip().lower()
    if legacy_value in {"scalar", "spatial"}:
        modes.add(legacy_value)
    rois = handle.get("rois")
    if isinstance(rois, h5py.Group):
        for roi_group in rois.values():
            if "scalar_features" in roi_group:
                modes.add("scalar")
            if "feature_maps" in roi_group and "feature_names" in roi_group:
                modes.add("spatial")
    return modes


def _validate_existing_container(
    handle: h5py.File,
    output_path: Path,
    case_id: str,
) -> None:
    schema_name = handle.attrs.get("schema_name")
    if isinstance(schema_name, bytes):
        schema_name = schema_name.decode("utf-8")
    if schema_name != SCHEMA_NAME:
        raise ValueError(
            f"Cannot merge into {output_path}: expected schema_name={SCHEMA_NAME!r}, "
            f"found {schema_name!r}"
        )
    stored_case_id = str(handle.attrs.get("case_id", ""))
    if stored_case_id and stored_case_id != str(case_id):
        raise ValueError(
            f"Cannot store case {case_id!r} in the case-level container for "
            f"{stored_case_id!r}: {output_path}"
        )


def write_extraction_hdf5(
    output_path: str | Path,
    *,
    mode: str,
    case_id: str,
    roi_id: str,
    roi_name: str,
    image: sitk.Image,
    feature_mask: sitk.Image,
    feature_vector: Mapping[str, Any],
    settings: Mapping[str, Any],
    backend: str,
    enabled_image_types: Mapping[str, Any],
    enabled_features: Mapping[str, Any],
    compression: str | None = "lzf",
) -> Path:
    """Atomically create or update one case-level Flash-Radiomics HDF5 file.

    The scalar payload is ``scalar_features/{names,values}``. The spatial
    payload uses the established cache paths ``feature_maps`` and
    ``feature_names`` with a channel-first common grid. Rewriting one mode
    replaces only that mode, so the other payload and phenotype groups remain
    intact.
    """
    mode = str(mode).strip().lower()
    if mode not in {"scalar", "spatial"}:
        raise ValueError("mode must be 'scalar' or 'spatial'")
    if compression not in {"gzip", "lzf", None}:
        raise ValueError("compression must be 'gzip', 'lzf', or None")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(fd)
    temporary_path = Path(tmp_name)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    try:
        output_exists = output_path.is_file()
        if output_exists:
            shutil.copyfile(output_path, temporary_path)

        with h5py.File(temporary_path, "a") as handle:
            if output_exists:
                _validate_existing_container(handle, output_path, str(case_id))

            timestamp = datetime.now(timezone.utc).isoformat()
            handle.attrs["schema_name"] = SCHEMA_NAME
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["format_name"] = FORMAT_NAME
            handle.attrs["format_version"] = FORMAT_VERSION
            handle.attrs["case_id"] = str(case_id)
            handle.attrs["created_by"] = "flash_radiomics"
            if "created_utc" not in handle.attrs:
                handle.attrs["created_utc"] = timestamp
            handle.attrs["updated_utc"] = timestamp
            handle.attrs["compression"] = (
                "none" if compression is None else compression
            )

            modes = _discover_stored_modes(handle)
            modes.add(mode)
            handle.attrs["modes_json"] = json.dumps(sorted(modes))
            handle.attrs["mode"] = (
                next(iter(modes)) if len(modes) == 1 else "combined"
            )

            backends = _read_string_list_attribute(handle, "backends_json")
            legacy_backend = str(handle.attrs.get("backend", "")).strip()
            if legacy_backend and legacy_backend != "mixed":
                backends.add(legacy_backend)
            backends.add(str(backend))
            handle.attrs["backends_json"] = json.dumps(sorted(backends))
            handle.attrs["backend"] = (
                next(iter(backends)) if len(backends) == 1 else "mixed"
            )

            roi_group = handle.require_group(f"rois/{roi_id}")
            roi_group.attrs["roi_id"] = str(roi_id)
            roi_group.attrs["roi_name"] = str(roi_name)

            feature_items = _spatial_feature_items(feature_vector)
            if mode == "spatial" and feature_items:
                reference = feature_items[0][1]
            else:
                reference = _roi_crop_reference(image, feature_mask)

            # Preserve the spatial grid when merging scalar outputs.
            if mode == "spatial" or "feature_maps" not in roi_group:
                _write_common_grid(
                    roi_group,
                    image=image,
                    feature_mask=feature_mask,
                    reference=reference,
                    compression=compression,
                )

            metadata_group = roi_group.require_group("metadata")
            extractions_group = metadata_group.require_group("extractions")
            if mode in extractions_group:
                del extractions_group[mode]
            mode_metadata = extractions_group.require_group(mode)
            mode_metadata.create_dataset(
                "settings_json",
                data=json.dumps(_jsonable(settings), sort_keys=True),
                dtype=string_dtype,
            )
            mode_metadata.create_dataset(
                "enabled_image_types_json",
                data=json.dumps(_jsonable(enabled_image_types), sort_keys=True),
                dtype=string_dtype,
            )
            mode_metadata.create_dataset(
                "enabled_features_json",
                data=json.dumps(_jsonable(enabled_features), sort_keys=True),
                dtype=string_dtype,
            )
            mode_metadata.create_dataset(
                "non_numeric_outputs_json",
                data=json.dumps(_metadata_outputs(feature_vector), sort_keys=True),
                dtype=string_dtype,
            )
            mode_metadata.attrs["backend"] = str(backend)

            if mode == "scalar":
                _write_scalar_payload(
                    roi_group,
                    feature_vector,
                    string_dtype=string_dtype,
                )
            else:
                _write_spatial_payload(
                    roi_group,
                    feature_items,
                    reference=reference,
                    string_dtype=string_dtype,
                    compression=compression,
                )

            handle.flush()
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path
