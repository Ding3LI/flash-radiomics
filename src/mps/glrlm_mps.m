#include "glrlm_mps.h"

#include "../features/glrlm_cpu.h"

FlashRadiomicsResult* flash_radiomics_glrlm_mps(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    void* device_handle,
    double bin_width,
    int bin_count
) {
    // CPU bridge keeps MPS API behavior aligned with the latest CPU formulas.
    (void)device_handle;
    return flash_radiomics_glrlm_cpu(
        img,
        mask,
        1,
        bin_width,
        bin_count
    );
}
