#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Memory allocation wrappers with error checking

void* flash_radiomics_malloc(size_t size) {
    if (size == 0) {
        return NULL;
    }
    
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "::ERROR:: Memory allocation failed: %zu bytes\n", size);
    }
    return ptr;
}

void* flash_radiomics_calloc(size_t nmemb, size_t size) {
    if (nmemb == 0 || size == 0) {
        return NULL;
    }
    
    void* ptr = calloc(nmemb, size);
    if (!ptr) {
        fprintf(stderr, "::ERROR:: Memory allocation failed: %zu x %zu bytes\n", nmemb, size);
    }
    return ptr;
}

void* flash_radiomics_realloc(void* ptr, size_t size) {
    if (size == 0) {
        flash_radiomics_free(ptr);
        return NULL;
    }
    
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        fprintf(stderr, "::ERROR:: Memory reallocation failed: %zu bytes\n", size);
    }
    return new_ptr;
}

void flash_radiomics_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// Safe memory allocation with error code reporting

void* flash_radiomics_malloc_safe(size_t size, int* error_code) {
    if (size == 0) {
        if (error_code) {
            *error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        }
        return NULL;
    }
    
    void* ptr = malloc(size);
    if (!ptr) {
        if (error_code) {
            *error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        }
        fprintf(stderr, "::ERROR:: Memory allocation failed: %zu bytes\n", size);
    } else {
        if (error_code) {
            *error_code = FLASH_RADIOMICS_SUCCESS;
        }
    }
    return ptr;
}

void* flash_radiomics_calloc_safe(size_t nmemb, size_t size, int* error_code) {
    if (nmemb == 0 || size == 0) {
        if (error_code) {
            *error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        }
        return NULL;
    }
    
    void* ptr = calloc(nmemb, size);
    if (!ptr) {
        if (error_code) {
            *error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        }
        fprintf(stderr, "::ERROR:: Memory allocation failed: %zu x %zu bytes\n", nmemb, size);
    } else {
        if (error_code) {
            *error_code = FLASH_RADIOMICS_SUCCESS;
        }
    }
    return ptr;
}

// Thread-safe error reporting functions

void flash_radiomics_set_error(int* error_code, char* error_msg, int code, const char* msg) {
    if (error_code) {
        #ifdef _OPENMP
        #pragma omp atomic write
        #endif
        *error_code = code;
    }
    
    if (error_msg && msg) {
        // Thread-safe string copy
        #ifdef _OPENMP
        #pragma omp critical(error_msg_write)
        #endif
        {
            strncpy(error_msg, msg, 255);
            error_msg[255] = '\0';
        }
    }
}

void flash_radiomics_set_error_formatted(int* error_code, char* error_msg, int code, 
                                    const char* format, ...) {
    if (error_code) {
        #ifdef _OPENMP
        #pragma omp atomic write
        #endif
        *error_code = code;
    }
    
    if (error_msg && format) {
        char buffer[256];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        
        // Thread-safe string copy
        #ifdef _OPENMP
        #pragma omp critical(error_msg_write)
        #endif
        {
            strncpy(error_msg, buffer, 255);
            error_msg[255] = '\0';
        }
    }
}

void flash_radiomics_clear_error(int* error_code, char* error_msg) {
    if (error_code) {
        #ifdef _OPENMP
        #pragma omp atomic write
        #endif
        *error_code = FLASH_RADIOMICS_SUCCESS;
    }
    
    if (error_msg) {
        #ifdef _OPENMP
        #pragma omp critical(error_msg_write)
        #endif
        {
            error_msg[0] = '\0';
        }
    }
}

void flash_radiomics_copy_error(int* dest_code, char* dest_msg, 
                          int src_code, const char* src_msg) {
    if (dest_code) {
        #ifdef _OPENMP
        #pragma omp atomic write
        #endif
        *dest_code = src_code;
    }
    
    if (dest_msg && src_msg) {
        #ifdef _OPENMP
        #pragma omp critical(error_msg_write)
        #endif
        {
            strncpy(dest_msg, src_msg, 255);
            dest_msg[255] = '\0';
        }
    }
}

// Dimension validation helpers

int flash_radiomics_validate_ndim(int ndim) {
    return (ndim == 2 || ndim == 3);
}

int flash_radiomics_validate_dims(const int dims[3]) {
    if (!dims) {
        return 0;
    }
    
    // Check that all dimensions are positive
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        return 0;
    }
    
    // Check for reasonable upper bounds to prevent overflow
    // Max dimension: 10000 voxels per axis (1 trillion voxels total max)
    if (dims[0] > 10000 || dims[1] > 10000 || dims[2] > 10000) {
        return 0;
    }
    
    return 1;
}

int flash_radiomics_validate_dims_2d(const int dims[3]) {
    if (!dims) {
        return 0;
    }
    
    // For 2D, x and y must be positive, z should be 1
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] != 1) {
        return 0;
    }
    
    // Check for reasonable upper bounds
    if (dims[0] > 10000 || dims[1] > 10000) {
        return 0;
    }
    
    return 1;
}

int flash_radiomics_validate_dims_3d(const int dims[3]) {
    if (!dims) {
        return 0;
    }
    
    // For 3D, all dimensions must be positive
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        return 0;
    }
    
    // Check for reasonable upper bounds
    if (dims[0] > 10000 || dims[1] > 10000 || dims[2] > 10000) {
        return 0;
    }
    
    return 1;
}

int flash_radiomics_validate_dims_with_ndim(int ndim, const int dims[3]) {
    if (!flash_radiomics_validate_ndim(ndim)) {
        return 0;
    }
    
    if (ndim == 2) {
        return flash_radiomics_validate_dims_2d(dims);
    } else {
        return flash_radiomics_validate_dims_3d(dims);
    }
}

int flash_radiomics_dims_match(const int dims1[3], const int dims2[3]) {
    if (!dims1 || !dims2) {
        return 0;
    }
    
    return (dims1[0] == dims2[0] && 
            dims1[1] == dims2[1] && 
            dims1[2] == dims2[2]);
}

int flash_radiomics_dims_and_ndim_match(int ndim1, const int dims1[3], 
                                   int ndim2, const int dims2[3]) {
    if (!dims1 || !dims2) {
        return 0;
    }
    
    // Dimensionality must match
    if (ndim1 != ndim2) {
        return 0;
    }
    
    // Dimensions must match
    return flash_radiomics_dims_match(dims1, dims2);
}

int* flash_radiomics_discretize_image(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    double bin_width,
    int bin_count,
    int* out_ng
) {
    if (!img || !mask || !out_ng) {
        return NULL;
    }

    size_t total_voxels = flash_radiomics_get_num_elements(img->ndim, img->dims);
    int* discretized = (int*)flash_radiomics_malloc(total_voxels * sizeof(int));
    if (!discretized) {
        return NULL;
    }

    double min_val = 0.0;
    double max_val = 0.0;
    int first = 1;

    for (size_t i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            float val = img->data[i];
            if (first) {
                min_val = max_val = val;
                first = 0;
            } else {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
    }

    if (first) {
        flash_radiomics_free(discretized);
        return NULL;
    }

    if (bin_count <= 0 && bin_width <= 0.0) {
        bin_width = 25.0;
    }

    int max_bin = 1;

    if (bin_count > 0) {
        double denom = max_val - min_val;
        for (size_t i = 0; i < total_voxels; i++) {
            if (mask->data[i] != 0) {
                float val = img->data[i];
                int bin = 1;
                if (denom > 0.0) {
                    if (val < max_val) {
                        bin = (int)floor((double)bin_count * (val - min_val) / denom) + 1;
                    } else {
                        bin = bin_count;
                    }
                }
                discretized[i] = bin;
                if (bin > max_bin) max_bin = bin;
            } else {
                discretized[i] = 0;
            }
        }
    } else {
        double low_bound = floor(min_val / bin_width) * bin_width;
        for (size_t i = 0; i < total_voxels; i++) {
            if (mask->data[i] != 0) {
                float val = img->data[i];
                int bin = (int)floor((val - low_bound) / bin_width) + 1;
                if (bin < 1) bin = 1;
                discretized[i] = bin;
                if (bin > max_bin) max_bin = bin;
            } else {
                discretized[i] = 0;
            }
        }
    }

    int* present = (int*)flash_radiomics_calloc(max_bin + 1, sizeof(int));
    if (!present) {
        flash_radiomics_free(discretized);
        return NULL;
    }
    for (size_t i = 0; i < total_voxels; i++) {
        int v = discretized[i];
        if (v > 0) present[v] = 1;
    }

    int* map = (int*)flash_radiomics_calloc(max_bin + 1, sizeof(int));
    if (!map) {
        flash_radiomics_free(present);
        flash_radiomics_free(discretized);
        return NULL;
    }

    int ng = 0;
    for (int v = 1; v <= max_bin; v++) {
        if (present[v]) {
            ng++;
            map[v] = ng;
        }
    }

    for (size_t i = 0; i < total_voxels; i++) {
        int v = discretized[i];
        if (v > 0) {
            discretized[i] = map[v];
        }
    }

    flash_radiomics_free(present);
    flash_radiomics_free(map);

    *out_ng = ng;
    return discretized;
}
