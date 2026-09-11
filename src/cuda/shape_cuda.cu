#include "shape_cuda.h"
extern "C" {
#include "../features/shape_cpu.h"
}

#include <cstdio>
#include <cstdlib>

FlashRadiomicsResult* flash_radiomics_shape_cuda(
    FlashRadiomicsMask* mask,
    float* spacing,
    CudaContext* ctx
) {
    (void)ctx;
    // Keep CUDA backend shape numerically identical to CPU cShape definitions.
    return flash_radiomics_shape_cpu(mask, spacing);
}
