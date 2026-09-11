#include "glcm_mps.h"

#include "../features/glcm_cpu.h"

FlashRadiomicsResult* flash_radiomics_glcm_mps(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    void* device_handle,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
) {
    // CPU bridge keeps MPS API behavior aligned with the latest CPU formulas.
    (void)device_handle;
    return flash_radiomics_glcm_cpu(
        img,
        mask,
        1,
        distances,
        num_distances,
        bin_width,
        bin_count
    );
}
