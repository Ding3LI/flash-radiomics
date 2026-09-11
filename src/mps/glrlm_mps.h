#ifndef FLASH_RADIOMICS_GLRLM_MPS_H
#define FLASH_RADIOMICS_GLRLM_MPS_H

#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// MPS implementation for GLRLM features
FlashRadiomicsResult* flash_radiomics_glrlm_mps(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    void* device_handle,
    double bin_width,
    int bin_count
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_GLRLM_MPS_H
