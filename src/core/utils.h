#ifndef FLASH_RADIOMICS_UTILS_H
#define FLASH_RADIOMICS_UTILS_H

#include "types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory allocation wrappers with error checking
// Returns NULL on failure and logs error to stderr
void* flash_radiomics_malloc(size_t size);
void* flash_radiomics_calloc(size_t nmemb, size_t size);
void* flash_radiomics_realloc(void* ptr, size_t size);
void flash_radiomics_free(void* ptr);

// Safe memory allocation with error code reporting
// Returns NULL on failure and sets error_code
void* flash_radiomics_malloc_safe(size_t size, int* error_code);
void* flash_radiomics_calloc_safe(size_t nmemb, size_t size, int* error_code);

// Array indexing helpers for 2D data
// Assumes [y][x] memory layout (y-major order)
static inline size_t flash_radiomics_index_2d(int x, int y, const int dims[3]) {
    return (size_t)y * dims[0] + (size_t)x;
}

// Array indexing helpers for 3D data
// Assumes [z][y][x] memory layout (z-major order)
static inline size_t flash_radiomics_index_3d(int x, int y, int z, const int dims[3]) {
    return (size_t)z * dims[1] * dims[0] + (size_t)y * dims[0] + (size_t)x;
}

// Generic indexing helper that works for both 2D and 3D
// For 2D: z should be 0
static inline size_t flash_radiomics_index(int x, int y, int z, int ndim, const int dims[3]) {
    if (ndim == 2) {
        return flash_radiomics_index_2d(x, y, dims);
    } else {
        return flash_radiomics_index_3d(x, y, z, dims);
    }
}

// Check if indices are within bounds for 2D
static inline int flash_radiomics_is_valid_index_2d(int x, int y, const int dims[3]) {
    return (x >= 0 && x < dims[0] && 
            y >= 0 && y < dims[1]);
}

// Check if indices are within bounds for 3D
static inline int flash_radiomics_is_valid_index_3d(int x, int y, int z, const int dims[3]) {
    return (x >= 0 && x < dims[0] && 
            y >= 0 && y < dims[1] && 
            z >= 0 && z < dims[2]);
}

// Generic bounds checking that works for both 2D and 3D
static inline int flash_radiomics_is_valid_index(int x, int y, int z, int ndim, const int dims[3]) {
    if (ndim == 2) {
        return flash_radiomics_is_valid_index_2d(x, y, dims);
    } else {
        return flash_radiomics_is_valid_index_3d(x, y, z, dims);
    }
}

// Get total number of voxels/pixels
static inline size_t flash_radiomics_get_num_voxels(const int dims[3]) {
    return (size_t)dims[0] * dims[1] * dims[2];
}

// Get total number of voxels/pixels based on dimensionality
static inline size_t flash_radiomics_get_num_elements(int ndim, const int dims[3]) {
    if (ndim == 2) {
        return (size_t)dims[0] * dims[1];
    } else {
        return (size_t)dims[0] * dims[1] * dims[2];
    }
}

// Thread-safe error reporting functions
// These functions are safe to call from multiple threads
void flash_radiomics_set_error(int* error_code, char* error_msg, int code, const char* msg);
void flash_radiomics_set_error_formatted(int* error_code, char* error_msg, int code, 
                                    const char* format, ...);

// Clear error state
void flash_radiomics_clear_error(int* error_code, char* error_msg);

// Copy error from one structure to another
void flash_radiomics_copy_error(int* dest_code, char* dest_msg, 
                          int src_code, const char* src_msg);

// Dimension validation helpers
int flash_radiomics_validate_ndim(int ndim);
int flash_radiomics_validate_dims(const int dims[3]);
int flash_radiomics_validate_dims_2d(const int dims[3]);
int flash_radiomics_validate_dims_3d(const int dims[3]);
int flash_radiomics_validate_dims_with_ndim(int ndim, const int dims[3]);
int flash_radiomics_dims_match(const int dims1[3], const int dims2[3]);
int flash_radiomics_dims_and_ndim_match(int ndim1, const int dims1[3], 
                                   int ndim2, const int dims2[3]);

// Discretize image within mask using pyradiomics-style binning.
// Returns a newly allocated int array of size N (flattened) with values:
// 0 for outside mask, and 1..Ng for ROI voxels.
// Caller must free the returned array.
int* flash_radiomics_discretize_image(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    double bin_width,
    int bin_count,
    int* out_ng
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_UTILS_H
