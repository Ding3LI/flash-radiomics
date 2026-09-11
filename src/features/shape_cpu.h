#ifndef FLASH_RADIOMICS_SHAPE_CPU_H
#define FLASH_RADIOMICS_SHAPE_CPU_H

#include "../core/types.h"

// Extract shape features using CPU
FlashRadiomicsResult* flash_radiomics_shape_cpu(
    FlashRadiomicsMask* mask,
    float* spacing
);

// Extract shape2D features using CPU
FlashRadiomicsResult* flash_radiomics_shape2d_cpu(
    FlashRadiomicsMask* mask,
    float* spacing
);

#endif // FLASH_RADIOMICS_SHAPE_CPU_H
