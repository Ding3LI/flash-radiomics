#include "reduction_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

// Warp size constant
#define WARP_SIZE 32

// Block size for reduction kernels (must be power of 2)
#define REDUCTION_BLOCK_SIZE 256

// ============================================================================
// Warp-level reduction primitives using shuffle instructions
// ============================================================================

// Warp-level sum reduction using shuffle down
__device__ __forceinline__ float warp_reduce_sum_float(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_sum_double(double val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level min reduction using shuffle down
__device__ __forceinline__ float warp_reduce_min_float(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fminf(val, other);
    }
    return val;
}

// Warp-level max reduction using shuffle down
__device__ __forceinline__ float warp_reduce_max_float(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// ============================================================================
// Block-level reduction using shared memory
// ============================================================================

// Block-level sum reduction for float
__device__ void block_reduce_sum_float(float* shared, float val, int tid) {
    // Warp-level reduction first
    val = warp_reduce_sum_float(val);
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        shared[tid / WARP_SIZE] = val;
    }
    __syncthreads();
    
    // First warp reduces the partial sums
    if (tid < WARP_SIZE) {
        val = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[tid] : 0.0f;
        val = warp_reduce_sum_float(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
}

// Block-level sum reduction for double
__device__ void block_reduce_sum_double(double* shared, double val, int tid) {
    // Warp-level reduction first
    val = warp_reduce_sum_double(val);
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        shared[tid / WARP_SIZE] = val;
    }
    __syncthreads();
    
    // First warp reduces the partial sums
    if (tid < WARP_SIZE) {
        val = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[tid] : 0.0;
        val = warp_reduce_sum_double(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
}

// ============================================================================
// Kernel: Parallel sum reduction (float)
// ============================================================================

__global__ void reduce_sum_float_kernel(const float* input, float* output, int n) {
    __shared__ float shared[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to handle large arrays
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    
    // Block-level reduction
    block_reduce_sum_float(shared, sum, tid);
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// ============================================================================
// Kernel: Parallel sum reduction (double)
// ============================================================================

__global__ void reduce_sum_double_kernel(const double* input, double* output, int n) {
    __shared__ double shared[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to handle large arrays
    double sum = 0.0;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    
    // Block-level reduction
    block_reduce_sum_double(shared, sum, tid);
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// ============================================================================
// Kernel: Parallel min reduction (float)
// ============================================================================

__global__ void reduce_min_float_kernel(const float* input, float* output, int n) {
    __shared__ float shared[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to find minimum
    float min_val = FLT_MAX;
    for (int i = idx; i < n; i += stride) {
        min_val = fminf(min_val, input[i]);
    }
    
    // Warp-level reduction
    min_val = warp_reduce_min_float(min_val);
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        shared[tid / WARP_SIZE] = min_val;
    }
    __syncthreads();
    
    // First warp reduces the partial mins
    if (tid < WARP_SIZE) {
        min_val = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[tid] : FLT_MAX;
        min_val = warp_reduce_min_float(min_val);
        if (tid == 0) {
            atomicMin((int*)output, __float_as_int(min_val));
        }
    }
}

// ============================================================================
// Kernel: Parallel max reduction (float)
// ============================================================================

__global__ void reduce_max_float_kernel(const float* input, float* output, int n) {
    __shared__ float shared[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to find maximum
    float max_val = -FLT_MAX;
    for (int i = idx; i < n; i += stride) {
        max_val = fmaxf(max_val, input[i]);
    }
    
    // Warp-level reduction
    max_val = warp_reduce_max_float(max_val);
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        shared[tid / WARP_SIZE] = max_val;
    }
    __syncthreads();
    
    // First warp reduces the partial maxs
    if (tid < WARP_SIZE) {
        max_val = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[tid] : -FLT_MAX;
        max_val = warp_reduce_max_float(max_val);
        if (tid == 0) {
            atomicMax((int*)output, __float_as_int(max_val));
        }
    }
}

// ============================================================================
// Kernel: Parallel min/max reduction (float)
// ============================================================================

__global__ void reduce_min_max_float_kernel(const float* input, float* min_out, 
                                             float* max_out, int n) {
    __shared__ float shared_min[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    __shared__ float shared_max[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to find min and max
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (int i = idx; i < n; i += stride) {
        float val = input[i];
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
    }
    
    // Warp-level reduction
    min_val = warp_reduce_min_float(min_val);
    max_val = warp_reduce_max_float(max_val);
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        int warp_id = tid / WARP_SIZE;
        shared_min[warp_id] = min_val;
        shared_max[warp_id] = max_val;
    }
    __syncthreads();
    
    // First warp reduces the partial results
    if (tid < WARP_SIZE) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        min_val = (tid < num_warps) ? shared_min[tid] : FLT_MAX;
        max_val = (tid < num_warps) ? shared_max[tid] : -FLT_MAX;
        min_val = warp_reduce_min_float(min_val);
        max_val = warp_reduce_max_float(max_val);
        if (tid == 0) {
            atomicMin((int*)min_out, __float_as_int(min_val));
            atomicMax((int*)max_out, __float_as_int(max_val));
        }
    }
}

// ============================================================================
// Kernel: Parallel sum of squares reduction (float to double)
// ============================================================================

__global__ void reduce_sum_squares_float_kernel(const float* input, double* output, int n) {
    __shared__ double shared[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to compute sum of squares
    double sum_sq = 0.0;
    for (int i = idx; i < n; i += stride) {
        double val = (double)input[i];
        sum_sq += val * val;
    }
    
    // Block-level reduction
    block_reduce_sum_double(shared, sum_sq, tid);
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// ============================================================================
// Kernel: Parallel sum of squares reduction (double)
// ============================================================================

__global__ void reduce_sum_squares_double_kernel(const double* input, double* output, int n) {
    __shared__ double shared[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to compute sum of squares
    double sum_sq = 0.0;
    for (int i = idx; i < n; i += stride) {
        double val = input[i];
        sum_sq += val * val;
    }
    
    // Block-level reduction
    block_reduce_sum_double(shared, sum_sq, tid);
    
    // First thread writes block result
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// ============================================================================
// Host functions
// ============================================================================

int cuda_reduce_sum_float(const float* d_input, float* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to zero
    cudaMemset(d_output, 0, sizeof(float));
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    // Limit grid size to avoid excessive atomics
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_sum_float_kernel<<<grid_size, block_size, 0, ctx->stream>>>(d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_sum_double(const double* d_input, double* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to zero
    cudaMemset(d_output, 0, sizeof(double));
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_sum_double_kernel<<<grid_size, block_size, 0, ctx->stream>>>(d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_min_float(const float* d_input, float* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to FLT_MAX
    float init_val = FLT_MAX;
    cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_min_float_kernel<<<grid_size, block_size, 0, ctx->stream>>>(d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_max_float(const float* d_input, float* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to -FLT_MAX
    float init_val = -FLT_MAX;
    cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_max_float_kernel<<<grid_size, block_size, 0, ctx->stream>>>(d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_min_max_float(const float* d_input, float* d_min, float* d_max, 
                               int n, CudaContext* ctx) {
    if (!d_input || !d_min || !d_max || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize outputs
    float init_min = FLT_MAX;
    float init_max = -FLT_MAX;
    cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_min_max_float_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_input, d_min, d_max, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_sum_squares_float(const float* d_input, double* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to zero
    cudaMemset(d_output, 0, sizeof(double));
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_sum_squares_float_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_sum_squares_double(const double* d_input, double* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to zero
    cudaMemset(d_output, 0, sizeof(double));
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_sum_squares_double_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

// ============================================================================
// Kahan Summation for High-Precision Accumulation
// ============================================================================
// Kahan summation (compensated summation) reduces numerical error when
// summing a sequence of finite-precision floating-point numbers.
// This is critical for statistical moment calculations where precision matters.

// Warp-level Kahan sum reduction
__device__ __forceinline__ void warp_reduce_sum_kahan(double& sum, double& compensation) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        double other_sum = __shfl_down_sync(0xffffffff, sum, offset);
        double other_comp = __shfl_down_sync(0xffffffff, compensation, offset);
        
        // Kahan summation algorithm
        double y = other_sum - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
        
        // Add the other thread's compensation
        compensation += other_comp;
    }
}

// ============================================================================
// Kernel: Kahan summation reduction (float to double)
// ============================================================================

__global__ void reduce_sum_kahan_float_kernel(const float* input, double* output, int n) {
    __shared__ double shared_sum[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    __shared__ double shared_comp[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop with Kahan summation
    double sum = 0.0;
    double compensation = 0.0;
    
    for (int i = idx; i < n; i += stride) {
        double y = (double)input[i] - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    // Warp-level Kahan reduction
    warp_reduce_sum_kahan(sum, compensation);
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        int warp_id = tid / WARP_SIZE;
        shared_sum[warp_id] = sum;
        shared_comp[warp_id] = compensation;
    }
    __syncthreads();
    
    // First warp reduces the partial sums with Kahan
    if (tid < WARP_SIZE) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum = (tid < num_warps) ? shared_sum[tid] : 0.0;
        compensation = (tid < num_warps) ? shared_comp[tid] : 0.0;
        
        warp_reduce_sum_kahan(sum, compensation);
        
        if (tid == 0) {
            // Final Kahan addition to global result
            // Note: This uses atomic which may lose some precision,
            // but it's necessary for multi-block reduction
            atomicAdd(output, sum);
        }
    }
}

// ============================================================================
// Kernel: Kahan summation reduction (double)
// ============================================================================

__global__ void reduce_sum_kahan_double_kernel(const double* input, double* output, int n) {
    __shared__ double shared_sum[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    __shared__ double shared_comp[REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop with Kahan summation
    double sum = 0.0;
    double compensation = 0.0;
    
    for (int i = idx; i < n; i += stride) {
        double y = input[i] - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    // Warp-level Kahan reduction
    warp_reduce_sum_kahan(sum, compensation);
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        int warp_id = tid / WARP_SIZE;
        shared_sum[warp_id] = sum;
        shared_comp[warp_id] = compensation;
    }
    __syncthreads();
    
    // First warp reduces the partial sums with Kahan
    if (tid < WARP_SIZE) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        sum = (tid < num_warps) ? shared_sum[tid] : 0.0;
        compensation = (tid < num_warps) ? shared_comp[tid] : 0.0;
        
        warp_reduce_sum_kahan(sum, compensation);
        
        if (tid == 0) {
            atomicAdd(output, sum);
        }
    }
}

// ============================================================================
// Kernel: Combined statistical moments with Kahan summation
// ============================================================================
// Computes mean, variance, skewness, and kurtosis in a single pass
// Uses Kahan summation for numerical stability

__global__ void reduce_moments_kernel(const float* input, const uint8_t* mask,
                                      double* moments, int n) {
    // moments[0] = sum (for mean)
    // moments[1] = sum of squares (for variance)
    // moments[2] = sum of cubes (for skewness)
    // moments[3] = sum of fourth powers (for kurtosis)
    // moments[4] = count of valid voxels
    
    __shared__ double shared_sums[5][REDUCTION_BLOCK_SIZE / WARP_SIZE];
    __shared__ double shared_comps[5][REDUCTION_BLOCK_SIZE / WARP_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize Kahan accumulators for each moment
    double sums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double comps[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    // Grid-stride loop with Kahan summation for each moment
    for (int i = idx; i < n; i += stride) {
        if (mask[i] > 0) {
            double val = (double)input[i];
            double val2 = val * val;
            double val3 = val2 * val;
            double val4 = val2 * val2;
            
            // Kahan summation for each moment
            double values[5] = {val, val2, val3, val4, 1.0};
            for (int m = 0; m < 5; m++) {
                double y = values[m] - comps[m];
                double t = sums[m] + y;
                comps[m] = (t - sums[m]) - y;
                sums[m] = t;
            }
        }
    }
    
    // Warp-level reduction for each moment
    for (int m = 0; m < 5; m++) {
        warp_reduce_sum_kahan(sums[m], comps[m]);
    }
    
    // First thread in each warp writes to shared memory
    if (tid % WARP_SIZE == 0) {
        int warp_id = tid / WARP_SIZE;
        for (int m = 0; m < 5; m++) {
            shared_sums[m][warp_id] = sums[m];
            shared_comps[m][warp_id] = comps[m];
        }
    }
    __syncthreads();
    
    // First warp reduces the partial sums
    if (tid < WARP_SIZE) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        
        for (int m = 0; m < 5; m++) {
            sums[m] = (tid < num_warps) ? shared_sums[m][tid] : 0.0;
            comps[m] = (tid < num_warps) ? shared_comps[m][tid] : 0.0;
            warp_reduce_sum_kahan(sums[m], comps[m]);
        }
        
        if (tid == 0) {
            for (int m = 0; m < 5; m++) {
                atomicAdd(&moments[m], sums[m]);
            }
        }
    }
}

// ============================================================================
// Host functions for Kahan summation
// ============================================================================

int cuda_reduce_sum_kahan_float(const float* d_input, double* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to zero
    cudaMemset(d_output, 0, sizeof(double));
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_sum_kahan_float_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_sum_kahan_double(const double* d_input, double* d_output, int n, CudaContext* ctx) {
    if (!d_input || !d_output || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize output to zero
    cudaMemset(d_output, 0, sizeof(double));
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_sum_kahan_double_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_input, d_output, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

int cuda_reduce_moments(const float* d_input, const uint8_t* d_mask,
                        double* d_moments, int n, CudaContext* ctx) {
    if (!d_input || !d_mask || !d_moments || n <= 0) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Initialize moments array to zero (5 elements)
    cudaMemset(d_moments, 0, 5 * sizeof(double));
    
    // Calculate grid and block dimensions
    int block_size = REDUCTION_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    grid_size = (grid_size > 1024) ? 1024 : grid_size;
    
    // Launch kernel
    reduce_moments_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_input, d_mask, d_moments, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}
