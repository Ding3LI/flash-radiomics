#ifndef FLASH_RADIOMICS_API_H
#define FLASH_RADIOMICS_API_H

#include "../core/types.h"

// Initialize execution context
FlashRadiomicsContext* flash_radiomics_init_context(FlashRadiomicsBackend backend);

// Detect available hardware backends
int flash_radiomics_detect_backends();

// Free context
void flash_radiomics_free_context(FlashRadiomicsContext* ctx);

// Segment-mode CUDA workspace controls (no-op on non-CUDA builds).
void flash_radiomics_cuda_reset_segment_workspace(void);
size_t flash_radiomics_cuda_segment_workspace_required_bytes(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask
);
void flash_radiomics_cuda_set_deterministic_mode(int enabled);
int flash_radiomics_cuda_get_deterministic_mode(void);
void flash_radiomics_cuda_configure_glszm_determinism(
    int tie_break_lowest_label,
    int deterministic_reduction,
    int allow_early_exit,
    int ccl_sync_interval,
    int ccl_flatten_interval
);
void flash_radiomics_cuda_set_bin_minimum(int enabled, double bin_minimum);

// Extract first-order features
FlashRadiomicsResult* flash_radiomics_firstorder(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    double bin_width,
    int bin_count,
    double voxel_array_shift
);

// Extract GLCM features
FlashRadiomicsResult* flash_radiomics_glcm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
);

// Extract GLRLM features
FlashRadiomicsResult* flash_radiomics_glrlm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    double bin_width,
    int bin_count
);

// Extract GLSZM features
FlashRadiomicsResult* flash_radiomics_glszm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    double bin_width,
    int bin_count
);

// Extract GLDM features
FlashRadiomicsResult* flash_radiomics_gldm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    double gldm_a,
    int force_2d
);

// Extract NGTDM features
FlashRadiomicsResult* flash_radiomics_ngtdm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    int force_2d
);

// Extract shape features
FlashRadiomicsResult* flash_radiomics_shape(
    FlashRadiomicsMask* mask,
    float* spacing,
    FlashRadiomicsBackend backend
);

// Extract shape2D features
FlashRadiomicsResult* flash_radiomics_shape2d(
    FlashRadiomicsMask* mask,
    float* spacing,
    FlashRadiomicsBackend backend
);

// Free result
void flash_radiomics_free_result(FlashRadiomicsResult* result);

#endif // FLASH_RADIOMICS_API_H
