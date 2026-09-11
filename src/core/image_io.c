#include "image_io.h"
#include "nifti1_io.h"
#include "utils.h"
#include "dicom_converter.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Helper function to check if path is a directory */
static int is_directory(const char* path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) {
        return 0;
    }
    return S_ISDIR(statbuf.st_mode);
}

/* Helper function to detect format from path */
static FlashRadiomicsImageFormat detect_format(const char* path) {
    if (!path) {
        return FLASH_RADIOMICS_FORMAT_NIFTI;
    }
    
    /* If path is a directory, assume DICOM series */
    if (is_directory(path)) {
        return FLASH_RADIOMICS_FORMAT_DICOM_SERIES;
    }
    
    /* Check file extension */
    const char* ext = strrchr(path, '.');
    if (ext) {
        /* NIfTI extensions */
        if (strcmp(ext, ".nii") == 0 || strcmp(ext, ".gz") == 0 ||
            strcmp(ext, ".hdr") == 0 || strcmp(ext, ".img") == 0) {
            return FLASH_RADIOMICS_FORMAT_NIFTI;
        }
        /* DICOM extensions */
        if (strcmp(ext, ".dcm") == 0 || strcmp(ext, ".DCM") == 0 ||
            strcmp(ext, ".dicom") == 0 || strcmp(ext, ".DICOM") == 0) {
            return FLASH_RADIOMICS_FORMAT_DICOM_SERIES;
        }
    }
    
    /* For files without clear extension, check if it looks like DICOM */
    /* DICOM files often have no extension or numeric extensions */
    /* Default to DICOM for single files without NIfTI extension */
    struct stat statbuf;
    if (stat(path, &statbuf) == 0 && S_ISREG(statbuf.st_mode)) {
        /* If it's a regular file without NIfTI extension, try DICOM */
        return FLASH_RADIOMICS_FORMAT_DICOM_SERIES;
    }
    
    /* Default to NIfTI */
    return FLASH_RADIOMICS_FORMAT_NIFTI;
}

/* Convert nifti_image data to float array */
static float* convert_nifti_to_float(nifti_image* nim, int* error_code) {
    if (!nim || !nim->data) {
        if (error_code) *error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        return NULL;
    }
    
    size_t nvox = nim->nvox;
    float* float_data = (float*)flash_radiomics_malloc_safe(nvox * sizeof(float), error_code);
    if (!float_data) {
        return NULL;
    }
    
    /* Apply scaling if present */
    float slope = (nim->scl_slope != 0.0f) ? nim->scl_slope : 1.0f;
    float inter = nim->scl_inter;
    
    /* Convert based on datatype */
    switch (nim->datatype) {
        case DT_UNSIGNED_CHAR:
        case DT_INT8: {
            uint8_t* data = (uint8_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                float_data[i] = (float)data[i] * slope + inter;
            }
            break;
        }
        case DT_SIGNED_SHORT: {
            int16_t* data = (int16_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                float_data[i] = (float)data[i] * slope + inter;
            }
            break;
        }
        case DT_UINT16: {
            uint16_t* data = (uint16_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                float_data[i] = (float)data[i] * slope + inter;
            }
            break;
        }
        case DT_SIGNED_INT: {
            int32_t* data = (int32_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                float_data[i] = (float)data[i] * slope + inter;
            }
            break;
        }
        case DT_UINT32: {
            uint32_t* data = (uint32_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                float_data[i] = (float)data[i] * slope + inter;
            }
            break;
        }
        case DT_FLOAT: {
            float* data = (float*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                float_data[i] = data[i] * slope + inter;
            }
            break;
        }
        case DT_DOUBLE: {
            double* data = (double*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                float_data[i] = (float)(data[i] * slope + inter);
            }
            break;
        }
        default:
            flash_radiomics_free(float_data);
            if (error_code) *error_code = FLASH_RADIOMICS_ERROR_INVALID_FORMAT;
            return NULL;
    }
    
    if (error_code) *error_code = FLASH_RADIOMICS_SUCCESS;
    return float_data;
}

/* Convert nifti_image mask data to uint8 array */
static uint8_t* convert_nifti_to_mask(nifti_image* nim, int label, int* error_code) {
    if (!nim || !nim->data) {
        if (error_code) *error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        return NULL;
    }
    
    size_t nvox = nim->nvox;
    uint8_t* mask_data = (uint8_t*)flash_radiomics_calloc_safe(nvox, sizeof(uint8_t), error_code);
    if (!mask_data) {
        return NULL;
    }
    
    /* Convert based on datatype - set to 1 where value equals label */
    switch (nim->datatype) {
        case DT_UNSIGNED_CHAR:
        case DT_INT8: {
            uint8_t* data = (uint8_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                mask_data[i] = (data[i] == label) ? 1 : 0;
            }
            break;
        }
        case DT_SIGNED_SHORT: {
            int16_t* data = (int16_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                mask_data[i] = (data[i] == label) ? 1 : 0;
            }
            break;
        }
        case DT_UINT16: {
            uint16_t* data = (uint16_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                mask_data[i] = (data[i] == label) ? 1 : 0;
            }
            break;
        }
        case DT_SIGNED_INT: {
            int32_t* data = (int32_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                mask_data[i] = (data[i] == label) ? 1 : 0;
            }
            break;
        }
        case DT_UINT32: {
            uint32_t* data = (uint32_t*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                mask_data[i] = (data[i] == label) ? 1 : 0;
            }
            break;
        }
        case DT_FLOAT: {
            float* data = (float*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                mask_data[i] = ((int)data[i] == label) ? 1 : 0;
            }
            break;
        }
        case DT_DOUBLE: {
            double* data = (double*)nim->data;
            for (size_t i = 0; i < nvox; i++) {
                mask_data[i] = ((int)data[i] == label) ? 1 : 0;
            }
            break;
        }
        default:
            flash_radiomics_free(mask_data);
            if (error_code) *error_code = FLASH_RADIOMICS_ERROR_INVALID_FORMAT;
            return NULL;
    }
    
    if (error_code) *error_code = FLASH_RADIOMICS_SUCCESS;
    return mask_data;
}

/* Load NIfTI image */
static FlashRadiomicsImage* load_nifti_image(const char* path) {
    FlashRadiomicsImage* img = NULL;
    nifti_image* nim = NULL;
    
    /* Allocate result structure */
    img = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
    if (!img) {
        return NULL;
    }
    
    /* Initialize error state */
    flash_radiomics_clear_error(&img->error_code, img->error_msg);
    
    /* Check if file exists and is NIfTI format */
    if (!is_nifti_file(path)) {
        flash_radiomics_set_error_formatted(&img->error_code, img->error_msg,
                                      FLASH_RADIOMICS_ERROR_INVALID_FORMAT,
                                      "Not a valid NIfTI file: %s", path);
        return img;
    }
    
    /* Read NIfTI file */
    nim = nifti_image_read(path, 1);
    if (!nim) {
        flash_radiomics_set_error_formatted(&img->error_code, img->error_msg,
                                      FLASH_RADIOMICS_ERROR_IO_FAILED,
                                      "Failed to read NIfTI file: %s", path);
        return img;
    }
    
    /* Validate dimensions (support 2D and 3D images) */
    if (nim->ndim < 2 || nim->nx <= 0 || nim->ny <= 0) {
        flash_radiomics_set_error_formatted(&img->error_code, img->error_msg,
                                      FLASH_RADIOMICS_ERROR_INVALID_FORMAT,
                                      "Invalid dimensions in NIfTI file: ndim=%d, dims=%dx%d",
                                      nim->ndim, nim->nx, nim->ny);
        nifti_image_free(nim);
        return img;
    }
    
    /* Detect dimensionality: 2D if nz <= 1, otherwise 3D */
    if (nim->nz <= 1) {
        /* 2D image */
        img->ndim = 2;
        img->dims[0] = nim->nx;
        img->dims[1] = nim->ny;
        img->dims[2] = 1;
    } else {
        /* 3D image */
        img->ndim = 3;
        img->dims[0] = nim->nx;
        img->dims[1] = nim->ny;
        img->dims[2] = nim->nz;
        
        /* For 4D+ images, only use first volume */
        if (nim->ndim > 3) {
            fprintf(stderr, "::WARN:: NIfTI file has %d dimensions, using first 3D volume only\n", nim->ndim);
        }
    }
    
    /* Fill in spacing */
    img->spacing[0] = (nim->dx > 0) ? nim->dx : 1.0f;
    img->spacing[1] = (nim->dy > 0) ? nim->dy : 1.0f;
    img->spacing[2] = (nim->dz > 0) ? nim->dz : 1.0f;
    
    /* Fill in origin from qform or sform */
    if (nim->qform_code > 0) {
        img->origin[0] = nim->qoffset_x;
        img->origin[1] = nim->qoffset_y;
        img->origin[2] = nim->qoffset_z;
    } else if (nim->sform_code > 0) {
        img->origin[0] = nim->sto_xyz[0][3];
        img->origin[1] = nim->sto_xyz[1][3];
        img->origin[2] = nim->sto_xyz[2][3];
    } else {
        img->origin[0] = img->origin[1] = img->origin[2] = 0.0f;
    }
    
    /* Convert data to float */
    img->data = convert_nifti_to_float(nim, &img->error_code);
    if (!img->data) {
        if (img->error_code == FLASH_RADIOMICS_SUCCESS) {
            flash_radiomics_set_error(&img->error_code, img->error_msg,
                               FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                               "Failed to allocate memory for image data");
        }
        nifti_image_free(nim);
        return img;
    }
    
    /* Clean up */
    nifti_image_free(nim);
    
    return img;
}

/* Load NIfTI mask */
static FlashRadiomicsMask* load_nifti_mask(const char* path, int label) {
    FlashRadiomicsMask* mask = NULL;
    nifti_image* nim = NULL;
    
    /* Allocate result structure */
    mask = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
    if (!mask) {
        return NULL;
    }
    
    /* Initialize error state */
    flash_radiomics_clear_error(&mask->error_code, mask->error_msg);
    mask->label = label;
    
    /* Check if file exists and is NIfTI format */
    if (!is_nifti_file(path)) {
        flash_radiomics_set_error_formatted(&mask->error_code, mask->error_msg,
                                      FLASH_RADIOMICS_ERROR_INVALID_FORMAT,
                                      "Not a valid NIfTI file: %s", path);
        return mask;
    }
    
    /* Read NIfTI file */
    nim = nifti_image_read(path, 1);
    if (!nim) {
        flash_radiomics_set_error_formatted(&mask->error_code, mask->error_msg,
                                      FLASH_RADIOMICS_ERROR_IO_FAILED,
                                      "Failed to read NIfTI file: %s", path);
        return mask;
    }
    
    /* Validate dimensions (support 2D and 3D masks) */
    if (nim->ndim < 2 || nim->nx <= 0 || nim->ny <= 0) {
        flash_radiomics_set_error_formatted(&mask->error_code, mask->error_msg,
                                      FLASH_RADIOMICS_ERROR_INVALID_FORMAT,
                                      "Invalid dimensions in NIfTI file: ndim=%d, dims=%dx%d",
                                      nim->ndim, nim->nx, nim->ny);
        nifti_image_free(nim);
        return mask;
    }
    
    /* Detect dimensionality: 2D if nz <= 1, otherwise 3D */
    if (nim->nz <= 1) {
        /* 2D mask */
        mask->ndim = 2;
        mask->dims[0] = nim->nx;
        mask->dims[1] = nim->ny;
        mask->dims[2] = 1;
    } else {
        /* 3D mask */
        mask->ndim = 3;
        mask->dims[0] = nim->nx;
        mask->dims[1] = nim->ny;
        mask->dims[2] = nim->nz;
    }
    
    /* Convert data to binary mask */
    mask->data = convert_nifti_to_mask(nim, label, &mask->error_code);
    if (!mask->data) {
        if (mask->error_code == FLASH_RADIOMICS_SUCCESS) {
            flash_radiomics_set_error(&mask->error_code, mask->error_msg,
                               FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                               "Failed to allocate memory for mask data");
        }
        nifti_image_free(nim);
        return mask;
    }
    
    /* Clean up */
    nifti_image_free(nim);
    
    return mask;
}

/* Public API: Load image */
FlashRadiomicsImage* flash_radiomics_load_image(const char* path, FlashRadiomicsImageFormat format) {
    if (!path) {
        FlashRadiomicsImage* img = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
        if (img) {
            flash_radiomics_set_error(&img->error_code, img->error_msg,
                               FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                               "Path cannot be NULL");
        }
        return img;
    }
    
    /* Auto-detect format if requested */
    if (format == FLASH_RADIOMICS_FORMAT_AUTO) {
        format = detect_format(path);
    }
    
    /* Load based on format */
    switch (format) {
        case FLASH_RADIOMICS_FORMAT_NIFTI:
            return load_nifti_image(path);
            
        case FLASH_RADIOMICS_FORMAT_DICOM_SERIES: {
            /* Convert DICOM to temporary NIfTI file */
            char temp_nifti[] = "/tmp/flash_radiomics_dicom_XXXXXX.nii";
            int fd = mkstemps(temp_nifti, 4);
            if (fd == -1) {
                FlashRadiomicsImage* img = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
                if (img) {
                    flash_radiomics_set_error(&img->error_code, img->error_msg,
                                       FLASH_RADIOMICS_ERROR_IO_FAILED,
                                       "Failed to create temporary file for DICOM conversion");
                }
                return img;
            }
            close(fd);
            
            /* Convert DICOM series to NIfTI */
            int result = flash_radiomics_dicom_to_nifti(path, temp_nifti);
            if (result != FLASH_RADIOMICS_SUCCESS) {
                unlink(temp_nifti);
                FlashRadiomicsImage* img = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
                if (img) {
                    flash_radiomics_set_error_formatted(&img->error_code, img->error_msg,
                                                  result,
                                                  "Failed to convert DICOM series: %s", path);
                }
                return img;
            }
            
            /* Load the converted NIfTI file */
            FlashRadiomicsImage* img = load_nifti_image(temp_nifti);
            
            /* Clean up temporary file */
            unlink(temp_nifti);
            
            return img;
        }
            
        default: {
            FlashRadiomicsImage* img = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
            if (img) {
                flash_radiomics_set_error(&img->error_code, img->error_msg,
                                   FLASH_RADIOMICS_ERROR_INVALID_FORMAT,
                                   "Unknown image format");
            }
            return img;
        }
    }
}

/* Public API: Load mask */
FlashRadiomicsMask* flash_radiomics_load_mask(const char* path, int label, FlashRadiomicsImageFormat format) {
    if (!path) {
        FlashRadiomicsMask* mask = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
        if (mask) {
            flash_radiomics_set_error(&mask->error_code, mask->error_msg,
                               FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                               "Path cannot be NULL");
        }
        return mask;
    }
    
    /* Auto-detect format if requested */
    if (format == FLASH_RADIOMICS_FORMAT_AUTO) {
        format = detect_format(path);
    }
    
    /* Load based on format */
    switch (format) {
        case FLASH_RADIOMICS_FORMAT_NIFTI:
            return load_nifti_mask(path, label);
            
        case FLASH_RADIOMICS_FORMAT_DICOM_SERIES: {
            /* Convert DICOM to temporary NIfTI file */
            char temp_nifti[] = "/tmp/flash_radiomics_dicom_mask_XXXXXX.nii";
            int fd = mkstemps(temp_nifti, 4);
            if (fd == -1) {
                FlashRadiomicsMask* mask = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
                if (mask) {
                    mask->label = label;
                    flash_radiomics_set_error(&mask->error_code, mask->error_msg,
                                       FLASH_RADIOMICS_ERROR_IO_FAILED,
                                       "Failed to create temporary file for DICOM conversion");
                }
                return mask;
            }
            close(fd);
            
            /* Convert DICOM series to NIfTI */
            int result = flash_radiomics_dicom_to_nifti(path, temp_nifti);
            if (result != FLASH_RADIOMICS_SUCCESS) {
                unlink(temp_nifti);
                FlashRadiomicsMask* mask = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
                if (mask) {
                    mask->label = label;
                    flash_radiomics_set_error_formatted(&mask->error_code, mask->error_msg,
                                                  result,
                                                  "Failed to convert DICOM series: %s", path);
                }
                return mask;
            }
            
            /* Load the converted NIfTI file */
            FlashRadiomicsMask* mask = load_nifti_mask(temp_nifti, label);
            
            /* Clean up temporary file */
            unlink(temp_nifti);
            
            return mask;
        }
            
        default: {
            FlashRadiomicsMask* mask = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
            if (mask) {
                mask->label = label;
                flash_radiomics_set_error(&mask->error_code, mask->error_msg,
                                   FLASH_RADIOMICS_ERROR_INVALID_FORMAT,
                                   "Unknown mask format");
            }
            return mask;
        }
    }
}

/* Public API: Validate inputs */
int flash_radiomics_validate_inputs(FlashRadiomicsImage* img, FlashRadiomicsMask* mask) {
    if (!img || !mask) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    
    /* Check if image and mask were loaded successfully */
    if (img->error_code != FLASH_RADIOMICS_SUCCESS) {
        return img->error_code;
    }
    
    if (mask->error_code != FLASH_RADIOMICS_SUCCESS) {
        return mask->error_code;
    }
    
    /* Check if data is present */
    if (!img->data || !mask->data) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    
    /* Validate ndim */
    if (!flash_radiomics_validate_ndim(img->ndim) || !flash_radiomics_validate_ndim(mask->ndim)) {
        return FLASH_RADIOMICS_ERROR_INVALID_FORMAT;
    }
    
    /* Validate dimensions based on ndim */
    if (!flash_radiomics_validate_dims_with_ndim(img->ndim, img->dims) || 
        !flash_radiomics_validate_dims_with_ndim(mask->ndim, mask->dims)) {
        return FLASH_RADIOMICS_ERROR_INVALID_FORMAT;
    }
    
    /* Check dimension and ndim compatibility */
    if (!flash_radiomics_dims_and_ndim_match(img->ndim, img->dims, mask->ndim, mask->dims)) {
        return FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH;
    }
    
    return FLASH_RADIOMICS_SUCCESS;
}

/* Public API: Free image */
void flash_radiomics_free_image(FlashRadiomicsImage* img) {
    if (img) {
        if (img->data) {
            flash_radiomics_free(img->data);
        }
        flash_radiomics_free(img);
    }
}

/* Public API: Extract 2D slice from 3D image */
FlashRadiomicsImage* flash_radiomics_extract_slice(FlashRadiomicsImage* img, int slice_index, int axis) {
    if (!img || !img->data) {
        FlashRadiomicsImage* result = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
        if (result) {
            flash_radiomics_set_error(&result->error_code, result->error_msg,
                               FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                               "Invalid image parameter");
        }
        return result;
    }
    
    /* Check if image is 3D */
    if (img->ndim != 3) {
        FlashRadiomicsImage* result = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
        if (result) {
            flash_radiomics_set_error(&result->error_code, result->error_msg,
                               FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                               "Image must be 3D for slice extraction");
        }
        return result;
    }
    
    /* Validate axis */
    if (axis < 0 || axis > 2) {
        FlashRadiomicsImage* result = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
        if (result) {
            flash_radiomics_set_error_formatted(&result->error_code, result->error_msg,
                                          FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                                          "Invalid axis: %d (must be 0, 1, or 2)", axis);
        }
        return result;
    }
    
    /* Validate slice index */
    if (slice_index < 0 || slice_index >= img->dims[axis]) {
        FlashRadiomicsImage* result = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
        if (result) {
            flash_radiomics_set_error_formatted(&result->error_code, result->error_msg,
                                          FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                                          "Slice index %d out of range [0, %d) for axis %d",
                                          slice_index, img->dims[axis], axis);
        }
        return result;
    }
    
    /* Allocate result structure */
    FlashRadiomicsImage* slice = (FlashRadiomicsImage*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsImage));
    if (!slice) {
        return NULL;
    }
    
    flash_radiomics_clear_error(&slice->error_code, slice->error_msg);
    slice->ndim = 2;
    
    /* Determine slice dimensions based on axis */
    int slice_width, slice_height;
    float spacing_x, spacing_y;
    
    switch (axis) {
        case 0: /* Sagittal (YZ plane) */
            slice_width = img->dims[1];
            slice_height = img->dims[2];
            spacing_x = img->spacing[1];
            spacing_y = img->spacing[2];
            break;
        case 1: /* Coronal (XZ plane) */
            slice_width = img->dims[0];
            slice_height = img->dims[2];
            spacing_x = img->spacing[0];
            spacing_y = img->spacing[2];
            break;
        case 2: /* Axial (XY plane) */
            slice_width = img->dims[0];
            slice_height = img->dims[1];
            spacing_x = img->spacing[0];
            spacing_y = img->spacing[1];
            break;
        default:
            flash_radiomics_free(slice);
            return NULL;
    }
    
    slice->dims[0] = slice_width;
    slice->dims[1] = slice_height;
    slice->dims[2] = 1;
    slice->spacing[0] = spacing_x;
    slice->spacing[1] = spacing_y;
    slice->spacing[2] = 1.0f;
    
    /* Copy origin (simplified - just use original origin) */
    slice->origin[0] = img->origin[0];
    slice->origin[1] = img->origin[1];
    slice->origin[2] = img->origin[2];
    
    /* Allocate slice data */
    size_t slice_size = slice_width * slice_height;
    slice->data = (float*)flash_radiomics_malloc_safe(slice_size * sizeof(float), &slice->error_code);
    if (!slice->data) {
        flash_radiomics_free(slice);
        return NULL;
    }
    
    /* Extract slice data */
    int nx = img->dims[0];
    int ny = img->dims[1];
    int nz = img->dims[2];
    
    switch (axis) {
        case 0: /* Sagittal - extract X=slice_index plane */
            for (int z = 0; z < nz; z++) {
                for (int y = 0; y < ny; y++) {
                    int src_idx = slice_index + y * nx + z * nx * ny;
                    int dst_idx = y + z * ny;
                    slice->data[dst_idx] = img->data[src_idx];
                }
            }
            break;
            
        case 1: /* Coronal - extract Y=slice_index plane */
            for (int z = 0; z < nz; z++) {
                for (int x = 0; x < nx; x++) {
                    int src_idx = x + slice_index * nx + z * nx * ny;
                    int dst_idx = x + z * nx;
                    slice->data[dst_idx] = img->data[src_idx];
                }
            }
            break;
            
        case 2: /* Axial - extract Z=slice_index plane */
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    int src_idx = x + y * nx + slice_index * nx * ny;
                    int dst_idx = x + y * nx;
                    slice->data[dst_idx] = img->data[src_idx];
                }
            }
            break;
    }
    
    return slice;
}

/* Public API: Extract 2D mask slice from 3D mask */
FlashRadiomicsMask* flash_radiomics_extract_mask_slice(FlashRadiomicsMask* mask, int slice_index, int axis) {
    if (!mask || !mask->data) {
        FlashRadiomicsMask* result = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
        if (result) {
            flash_radiomics_set_error(&result->error_code, result->error_msg,
                               FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                               "Invalid mask parameter");
        }
        return result;
    }
    
    /* Check if mask is 3D */
    if (mask->ndim != 3) {
        FlashRadiomicsMask* result = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
        if (result) {
            flash_radiomics_set_error(&result->error_code, result->error_msg,
                               FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                               "Mask must be 3D for slice extraction");
        }
        return result;
    }
    
    /* Validate axis */
    if (axis < 0 || axis > 2) {
        FlashRadiomicsMask* result = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
        if (result) {
            flash_radiomics_set_error_formatted(&result->error_code, result->error_msg,
                                          FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                                          "Invalid axis: %d (must be 0, 1, or 2)", axis);
        }
        return result;
    }
    
    /* Validate slice index */
    if (slice_index < 0 || slice_index >= mask->dims[axis]) {
        FlashRadiomicsMask* result = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
        if (result) {
            flash_radiomics_set_error_formatted(&result->error_code, result->error_msg,
                                          FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                                          "Slice index %d out of range [0, %d) for axis %d",
                                          slice_index, mask->dims[axis], axis);
        }
        return result;
    }
    
    /* Allocate result structure */
    FlashRadiomicsMask* slice = (FlashRadiomicsMask*)flash_radiomics_calloc(1, sizeof(FlashRadiomicsMask));
    if (!slice) {
        return NULL;
    }
    
    flash_radiomics_clear_error(&slice->error_code, slice->error_msg);
    slice->ndim = 2;
    slice->label = mask->label;
    
    /* Determine slice dimensions based on axis */
    int slice_width, slice_height;
    
    switch (axis) {
        case 0: /* Sagittal (YZ plane) */
            slice_width = mask->dims[1];
            slice_height = mask->dims[2];
            break;
        case 1: /* Coronal (XZ plane) */
            slice_width = mask->dims[0];
            slice_height = mask->dims[2];
            break;
        case 2: /* Axial (XY plane) */
            slice_width = mask->dims[0];
            slice_height = mask->dims[1];
            break;
        default:
            flash_radiomics_free(slice);
            return NULL;
    }
    
    slice->dims[0] = slice_width;
    slice->dims[1] = slice_height;
    slice->dims[2] = 1;
    
    /* Allocate slice data */
    size_t slice_size = slice_width * slice_height;
    slice->data = (uint8_t*)flash_radiomics_calloc_safe(slice_size, sizeof(uint8_t), &slice->error_code);
    if (!slice->data) {
        flash_radiomics_free(slice);
        return NULL;
    }
    
    /* Extract slice data */
    int nx = mask->dims[0];
    int ny = mask->dims[1];
    int nz = mask->dims[2];
    
    switch (axis) {
        case 0: /* Sagittal - extract X=slice_index plane */
            for (int z = 0; z < nz; z++) {
                for (int y = 0; y < ny; y++) {
                    int src_idx = slice_index + y * nx + z * nx * ny;
                    int dst_idx = y + z * ny;
                    slice->data[dst_idx] = mask->data[src_idx];
                }
            }
            break;
            
        case 1: /* Coronal - extract Y=slice_index plane */
            for (int z = 0; z < nz; z++) {
                for (int x = 0; x < nx; x++) {
                    int src_idx = x + slice_index * nx + z * nx * ny;
                    int dst_idx = x + z * nx;
                    slice->data[dst_idx] = mask->data[src_idx];
                }
            }
            break;
            
        case 2: /* Axial - extract Z=slice_index plane */
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    int src_idx = x + y * nx + slice_index * nx * ny;
                    int dst_idx = x + y * nx;
                    slice->data[dst_idx] = mask->data[src_idx];
                }
            }
            break;
    }
    
    return slice;
}

/* Public API: Free mask */
void flash_radiomics_free_mask(FlashRadiomicsMask* mask) {
    if (mask) {
        if (mask->data) {
            flash_radiomics_free(mask->data);
        }
        flash_radiomics_free(mask);
    }
}
