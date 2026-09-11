#ifndef FLASH_RADIOMICS_GLCM_MPS_H
#define FLASH_RADIOMICS_GLCM_MPS_H

#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// MPS implementation for GLCM features
FlashRadiomicsResult* flash_radiomics_glcm_mps(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    void* device_handle,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_GLCM_MPS_H
