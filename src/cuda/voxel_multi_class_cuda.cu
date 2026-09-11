/**
 * voxel_multi_class_cuda.cu
 *
 * This file exists solely so CMake picks up the compilation unit.
 * The actual implementation of flash_radiomics_voxel_multi_class_cuda_batch
 * lives in voxel_batch_texture_cuda.cu, which has direct access to the
 * anonymous-namespace kernel functions.
 */
#include "voxel_multi_class_cuda.h"
