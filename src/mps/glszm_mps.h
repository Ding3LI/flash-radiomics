#ifndef FLASH_RADIOMICS_GLSZM_MPS_H
#define FLASH_RADIOMICS_GLSZM_MPS_H

#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// MPS implementation for GLSZM features
FlashRadiomicsResult* flash_radiomics_glszm_mps(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    void* device_handle,
    double bin_width,
    int bin_count
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_GLSZM_MPS_H
