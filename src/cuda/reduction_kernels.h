#ifndef FLASH_RADIOMICS_REDUCTION_KERNELS_H
#define FLASH_RADIOMICS_REDUCTION_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cuda_common.h"

// Parallel reduction operations for CUDA
// These kernels provide efficient parallel reduction primitives
// using warp shuffle instructions and shared memory

// Sum reduction: compute sum of all elements
int cuda_reduce_sum_float(const float* d_input, float* d_output, int n, CudaContext* ctx);
int cuda_reduce_sum_double(const double* d_input, double* d_output, int n, CudaContext* ctx);

// Min/Max reduction: find minimum and maximum values
int cuda_reduce_min_float(const float* d_input, float* d_output, int n, CudaContext* ctx);
int cuda_reduce_max_float(const float* d_input, float* d_output, int n, CudaContext* ctx);
int cuda_reduce_min_max_float(const float* d_input, float* d_min, float* d_max, int n, CudaContext* ctx);

// Sum of squares reduction: compute sum of squared elements
int cuda_reduce_sum_squares_float(const float* d_input, double* d_output, int n, CudaContext* ctx);
int cuda_reduce_sum_squares_double(const double* d_input, double* d_output, int n, CudaContext* ctx);

// Kahan summation for high-precision accumulation
int cuda_reduce_sum_kahan_float(const float* d_input, double* d_output, int n, CudaContext* ctx);
int cuda_reduce_sum_kahan_double(const double* d_input, double* d_output, int n, CudaContext* ctx);

// Combined statistical moments reduction (mean, variance, skewness, kurtosis)
// Computes multiple moments in a single pass for efficiency
int cuda_reduce_moments(const float* d_input, const uint8_t* d_mask, 
                        double* d_moments, int n, CudaContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_REDUCTION_KERNELS_H
