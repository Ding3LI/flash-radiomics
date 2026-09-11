#ifndef FLASH_RADIOMICS_FIRSTORDER_MPS_H
#define FLASH_RADIOMICS_FIRSTORDER_MPS_H

#include "../core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Extract first-order features using MPS (Metal Performance Shaders)
 * 
 * @param img Input image
 * @param mask Input mask
 * @param device_handle Metal device handle (MTLDevice*)
 * @return Result structure with computed features
 */
FlashRadiomicsResult* flash_radiomics_firstorder_mps(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    void* device_handle,
    double bin_width,
    int bin_count,
    double voxel_array_shift
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_FIRSTORDER_MPS_H
