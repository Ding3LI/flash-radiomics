#ifndef FLASH_RADIOMICS_SHAPE_CUDA_H
#define FLASH_RADIOMICS_SHAPE_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../core/types.h"
#include "cuda_common.h"

// CUDA implementation of shape features
FlashRadiomicsResult* flash_radiomics_shape_cuda(
    FlashRadiomicsMask* mask,
    float* spacing,
    CudaContext* ctx
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_SHAPE_CUDA_H
