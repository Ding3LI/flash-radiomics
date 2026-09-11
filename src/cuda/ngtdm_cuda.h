#ifndef FLASH_RADIOMICS_NGTDM_CUDA_H
#define FLASH_RADIOMICS_NGTDM_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../core/types.h"
#include "cuda_common.h"

// CUDA implementation of NGTDM features
FlashRadiomicsResult* flash_radiomics_ngtdm_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    int force_2d
);

// CUDA voxel-batch API for discretized windows.
int flash_radiomics_ngtdm_cuda_discretized_voxel_batch(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    const int* distances,
    int num_distances,
    const uint8_t* include_features,
    int ng,
    double* out_features,
    int out_feature_stride
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_NGTDM_CUDA_H
