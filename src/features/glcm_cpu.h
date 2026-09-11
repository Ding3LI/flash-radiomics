#ifndef FLASH_RADIOMICS_GLCM_CPU_H
#define FLASH_RADIOMICS_GLCM_CPU_H

#include "../core/types.h"

// Extract GLCM features using CPU
FlashRadiomicsResult* flash_radiomics_glcm_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
);

FlashRadiomicsResult* flash_radiomics_glcm_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    const int* discretized,
    int ng
);

FlashRadiomicsResult* flash_radiomics_glcm_cpu_discretized_selective(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    const int* discretized,
    int ng,
    int include_mcc
);

#define FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT 24

// Compute voxel GLCM features for a batch of discretized kernel windows.
// Inputs are flattened C-order windows (y,x for 2D; z,y,x for 3D), grouped
// by batch. window_dims follow [x, y, z] (z=1 for 2D).
// out_features must provide at least batch_size * out_feature_stride doubles.
// Returns FlashRadiomicsErrorCode.
int flash_radiomics_glcm_cpu_discretized_voxel_batch(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    int* distances,
    int num_distances,
    const uint8_t* include_features,
    int ng,
    double* out_features,
    int out_feature_stride
);

#endif // FLASH_RADIOMICS_GLCM_CPU_H
