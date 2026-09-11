#ifndef FLASH_RADIOMICS_VOXEL_MULTI_CLASS_CUDA_H
#define FLASH_RADIOMICS_VOXEL_MULTI_CLASS_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Class enable bitmask values for flash_radiomics_voxel_multi_class_cuda_batch.
#define FLASH_VOXEL_CLASS_GLCM   (1u << 0)
#define FLASH_VOXEL_CLASS_GLRLM  (1u << 1)
#define FLASH_VOXEL_CLASS_GLSZM  (1u << 2)
#define FLASH_VOXEL_CLASS_GLDM   (1u << 3)
#define FLASH_VOXEL_CLASS_NGTDM  (1u << 4)
#define FLASH_VOXEL_CLASS_ALL    (FLASH_VOXEL_CLASS_GLCM | FLASH_VOXEL_CLASS_GLRLM | \
                                  FLASH_VOXEL_CLASS_GLSZM | FLASH_VOXEL_CLASS_GLDM | \
                                  FLASH_VOXEL_CLASS_NGTDM)

/**
 * Unified multi-class voxel feature extraction on CUDA.
 *
 * Uploads discretized_windows and mask_windows to the GPU ONCE,
 * then launches all enabled feature class kernels concurrently on
 * separate CUDA streams with CUDA-event synchronization.
 *
 * Common parameters:
 *   discretized_windows  - host pointer to [batch_size * window_voxels] int32 values
 *   mask_windows         - host pointer to [batch_size * window_voxels] uint8 values
 *   batch_size           - number of patches in this batch
 *   ndim                 - 2 or 3
 *   window_dims          - [depth, height, width]
 *   ng                   - number of gray levels (bins)
 *   class_mask           - bitmask of FLASH_VOXEL_CLASS_* flags
 *
 * Per-class parameters (ignored if the class is not enabled):
 *   GLCM: distances, num_distances, glcm_include_features, glcm_out, glcm_out_stride
 *   GLRLM: glrlm_include_features, glrlm_out, glrlm_out_stride
 *   GLSZM: glszm_include_features, glszm_out, glszm_out_stride
 *   GLDM: distances, num_distances (shared with GLCM), gldm_a,
 *          gldm_include_features, gldm_out, gldm_out_stride
 *   NGTDM: distances, num_distances (shared with GLCM),
 *           ngtdm_include_features, ngtdm_out, ngtdm_out_stride
 *
 * Returns 0 (FLASH_RADIOMICS_SUCCESS) on success, negative error code on failure.
 */
int flash_radiomics_voxel_multi_class_cuda_batch(
    /* Shared input */
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    int ng,
    unsigned int class_mask,

    /* Shared distance array (GLCM, GLDM, NGTDM) */
    const int* distances,
    int num_distances,

    /* GLDM-specific */
    double gldm_a,

    /* GLCM output */
    const uint8_t* glcm_include_features,
    double* glcm_out_features,
    int glcm_out_feature_stride,

    /* GLRLM output */
    const uint8_t* glrlm_include_features,
    double* glrlm_out_features,
    int glrlm_out_feature_stride,

    /* GLSZM output */
    const uint8_t* glszm_include_features,
    double* glszm_out_features,
    int glszm_out_feature_stride,

    /* GLDM output */
    const uint8_t* gldm_include_features,
    double* gldm_out_features,
    int gldm_out_feature_stride,

    /* NGTDM output */
    const uint8_t* ngtdm_include_features,
    double* ngtdm_out_features,
    int ngtdm_out_feature_stride
);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_VOXEL_MULTI_CLASS_CUDA_H
