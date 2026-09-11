# Flash-Radiomics HDF5 schema

Flash-Radiomics writes one case-level `*.flashrad.h5` file. The schema name is `flashrad_hdf5`.

The structure is fixed across all supported extraction choices:

```text
/
  attrs: schema_name, schema_version, case_id, mode, modes_json
  rois/
    <roi_id>/
      attrs: roi_id, roi_name, crop_shape
      image
      mask
      metadata/
        extractions/
          scalar/
            settings_json
            enabled_image_types_json
            enabled_features_json
            non_numeric_outputs_json
          spatial/
            settings_json
            enabled_image_types_json
            enabled_features_json
            non_numeric_outputs_json
      scalar_features/
        names
        values
      feature_names
      feature_maps
```

`scalar_features` is present when scalar extraction has been run.
`feature_names` and `feature_maps` are present when spatial extraction has been
run. Their absence means that mode was not requested; it does not indicate a
different schema.

## Dataset contracts

- `image` and `mask` use shape `(1, Z, Y, X)` for 3D data or `(1, Y, X)` for
  2D data.
- `feature_maps` uses shape `(C, Z, Y, X)` for 3D data or `(C, Y, X)` for 2D
  data.
- `feature_names` has length `C`, and item `i` names channel `i` in
  `feature_maps`.
- `scalar_features/names` and `scalar_features/values` have equal lengths.
  Item `i` in `names` identifies item `i` in `values`.
- Spatial maps, the image, and the mask share one physical grid. Geometry is
  recorded on the image-like datasets as spacing, origin, direction, size,
  and array-axis-order attributes.
- Partial and full feature selections only change `C` or the scalar feature
  count. Dataset paths and dimensional conventions remain unchanged.
- Diagnostic outputs are stored in the mode-specific metadata and are not
  mixed into radiomics feature arrays.

## Mode and merge behavior

The root `mode` attribute is `scalar`, `spatial`, or `combined`.
`modes_json` is the machine-readable list of payloads currently stored.

Scalar and spatial calls using the same case ID, ROI ID, and output path are
merged atomically. Re-running a mode replaces only that mode. Existing data
from the other mode and existing `phenotypes` groups are preserved. A writer
refuses to merge a different case ID into an existing case-level file.

When an output directory is configured instead of an exact file path, both
modes resolve to:

```text
<output_directory>/<case_id>_<roi_id>.flashrad.h5
```
