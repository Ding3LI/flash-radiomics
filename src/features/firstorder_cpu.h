#ifndef FLASH_RADIOMICS_FIRSTORDER_CPU_H
#define FLASH_RADIOMICS_FIRSTORDER_CPU_H

#include "../core/types.h"

// Extract first-order features using CPU
FlashRadiomicsResult* flash_radiomics_firstorder_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    double bin_width,
    int bin_count,
    double voxel_array_shift
);

FlashRadiomicsResult* flash_radiomics_firstorder_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    const int* discretized,
    int ng,
    double voxel_array_shift
);

#endif // FLASH_RADIOMICS_FIRSTORDER_CPU_H
