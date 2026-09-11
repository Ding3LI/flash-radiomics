#ifndef FLASH_RADIOMICS_FIRSTORDER_CUDA_H
#define FLASH_RADIOMICS_FIRSTORDER_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../core/types.h"
#include "cuda_common.h"

// CUDA implementation of first-order features
FlashRadiomicsResult* flash_radiomics_firstorder_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    double bin_width,
    int bin_count,
    double voxel_array_shift
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_FIRSTORDER_CUDA_H
