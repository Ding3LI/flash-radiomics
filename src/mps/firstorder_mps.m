#include "firstorder_mps.h"

#include "../features/firstorder_cpu.h"

FlashRadiomicsResult* flash_radiomics_firstorder_mps(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    void* device_handle,
    double bin_width,
    int bin_count,
    double voxel_array_shift
) {
    // CPU bridge keeps MPS API behavior aligned with the latest CPU formulas.
    (void)device_handle;
    return flash_radiomics_firstorder_cpu(
        img,
        mask,
        1,
        bin_width,
        bin_count,
        voxel_array_shift
    );
}
