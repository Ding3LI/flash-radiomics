#include "cuda_common.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef FLASH_RADIOMICS_CUDA_ENABLED
#include <cuda_runtime.h>

#define FLASH_CUDA_DEFAULT_NUM_BINS 256
#define FLASH_CUDA_DEFAULT_SEGMENT_WORKSPACE_MB 512
#define FLASH_CUDA_DEFAULT_GLSZM_CCL_SYNC_INTERVAL 32
#define FLASH_CUDA_DEFAULT_GLSZM_CCL_FLATTEN_INTERVAL 16

enum {
    FLASH_CUDA_DISCRETIZE_RANGE = 0,
    FLASH_CUDA_DISCRETIZE_FIXED_WIDTH = 1
};

__global__ static void cuda_discretize_segment_kernel(
    const float* image,
    const uint8_t* mask,
    int* discretized,
    int n_voxels,
    int mode,
    int num_bins,
    float min_val,
    float max_val,
    float bin_width,
    int min_bin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n_voxels; i += stride) {
        if (mask[i] == 0) {
            discretized[i] = -1;
            continue;
        }

        const float val = image[i];
        int bin_idx = 0;
        if (mode == FLASH_CUDA_DISCRETIZE_FIXED_WIDTH) {
            if (bin_width > 0.0f) {
                if (min_bin == INT_MIN) {
                    bin_idx = (int)floorf((val - min_val) / bin_width);
                } else {
                    bin_idx = (int)floorf(val / bin_width) - min_bin;
                }
            }
        } else {
            float range = max_val - min_val;
            if (range > 0.0f) {
                float scaled = ((val - min_val) / range) * (float)num_bins;
                bin_idx = (int)floorf(scaled);
                if (val >= max_val) {
                    bin_idx = num_bins - 1;
                }
            }
        }

        if (bin_idx < 0) {
            bin_idx = 0;
        } else if (bin_idx >= num_bins) {
            bin_idx = num_bins - 1;
        }
        discretized[i] = bin_idx;
    }
}

static size_t cuda_default_segment_workspace_budget_bytes(const CudaDeviceInfo* device) {
    size_t default_budget = (size_t)FLASH_CUDA_DEFAULT_SEGMENT_WORKSPACE_MB * 1024u * 1024u;
    if (!device || device->free_memory == 0) {
        return default_budget;
    }
    size_t half_free = device->free_memory / 2u;
    if (half_free == 0) {
        return default_budget;
    }
    return (half_free < default_budget) ? half_free : default_budget;
}

static int cuda_workspace_error_from_runtime(cudaError_t err) {
    switch (err) {
        case cudaErrorMemoryAllocation:
            return FLASH_CUDA_ERROR_OUT_OF_MEMORY;
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevicePointer:
        case cudaErrorInvalidMemcpyDirection:
            return FLASH_CUDA_ERROR_INVALID_VALUE;
        default:
            return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
}

static void cuda_set_default_determinism_config(CudaContext* ctx) {
    if (!ctx) {
        return;
    }
    ctx->deterministic_mode = 0;
    ctx->glszm_deterministic_reduction = 0;
    ctx->glszm_ccl_tie_break_lowest_label = 0;
    ctx->glszm_ccl_allow_early_exit = 1;
    ctx->glszm_ccl_sync_interval = FLASH_CUDA_DEFAULT_GLSZM_CCL_SYNC_INTERVAL;
    ctx->glszm_ccl_flatten_interval = FLASH_CUDA_DEFAULT_GLSZM_CCL_FLATTEN_INTERVAL;
}

static int cuda_compute_mask_min_max_host(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    float* min_val,
    float* max_val
) {
    if (!img || !mask || !min_val || !max_val) {
        return 0;
    }
    int n_voxels = img->dims[0] * img->dims[1] * img->dims[2];
    int found = 0;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    for (int i = 0; i < n_voxels; i++) {
        if (mask->data[i] == 0) {
            continue;
        }
        float v = img->data[i];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
        found = 1;
    }
    if (!found) {
        return 0;
    }
    *min_val = local_min;
    *max_val = local_max;
    return 1;
}

static CudaDiscretizationDescriptor cuda_resolve_discretization_descriptor(
    float min_val,
    float max_val,
    double bin_width,
    int bin_count,
    int has_bin_minimum,
    double bin_minimum
) {
    CudaDiscretizationDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.min_val = min_val;
    desc.max_val = max_val;

    if (bin_count > 0) {
        desc.mode = FLASH_CUDA_DISCRETIZE_RANGE;
        desc.num_bins = bin_count;
        return desc;
    }

    if (bin_width > 0.0) {
        desc.mode = FLASH_CUDA_DISCRETIZE_FIXED_WIDTH;
        desc.bin_width = (float)bin_width;
        if (has_bin_minimum) {
            desc.min_val = (float)bin_minimum;
            desc.min_bin = INT_MIN;
            desc.num_bins = (int)floor(((double)max_val - bin_minimum) / bin_width) + 1;
        } else {
            desc.min_bin = (int)floor((double)min_val / bin_width);
            {
                int max_bin = (int)floor((double)max_val / bin_width);
                desc.num_bins = max_bin - desc.min_bin + 1;
            }
        }
        {
            if (desc.num_bins < 1) {
                desc.num_bins = 1;
            }
        }
        return desc;
    }

    desc.mode = FLASH_CUDA_DISCRETIZE_RANGE;
    desc.num_bins = FLASH_CUDA_DEFAULT_NUM_BINS;
    return desc;
}

static int cuda_segment_workspace_matches(
    const CudaSegmentWorkspace* ws,
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    double bin_width,
    int bin_count,
    int has_bin_minimum,
    double bin_minimum
) {
    if (!ws || !ws->valid || !img || !mask) {
        return 0;
    }
    if (ws->host_image_ptr != (uintptr_t)img->data) {
        return 0;
    }
    if (ws->host_mask_ptr != (uintptr_t)mask->data) {
        return 0;
    }
    if (ws->ndim != img->ndim || ws->label != mask->label) {
        return 0;
    }
    if (ws->dims[0] != img->dims[0] || ws->dims[1] != img->dims[1] || ws->dims[2] != img->dims[2]) {
        return 0;
    }
    if (ws->bin_count != bin_count || ws->bin_width != bin_width) {
        return 0;
    }
    if (ws->has_bin_minimum != ((has_bin_minimum != 0) ? 1 : 0)) {
        return 0;
    }
    if (ws->has_bin_minimum && ws->bin_minimum != bin_minimum) {
        return 0;
    }
    if (!ws->d_image || !ws->d_mask || !ws->d_discretized || ws->n_voxels <= 0) {
        return 0;
    }
    return 1;
}

// Device detection
int cuda_detect_device(CudaDeviceInfo* info) {
    if (!info) return FLASH_CUDA_ERROR_INVALID_VALUE;
    
    // Initialize structure
    memset(info, 0, sizeof(CudaDeviceInfo));
    
    // Check CUDA availability using cudaGetDeviceCount()
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        info->is_available = 0;
        return FLASH_CUDA_ERROR_NO_DEVICE;
    }
    
    // Get properties of first device (device 0)
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        info->is_available = 0;
        return FLASH_CUDA_ERROR_NO_DEVICE;
    }
    
    // Extract device information
    info->device_id = 0;
    strncpy(info->device_name, prop.name, sizeof(info->device_name) - 1);
    info->device_name[sizeof(info->device_name) - 1] = '\0';
    
    // Extract compute capability
    info->compute_capability_major = prop.major;
    info->compute_capability_minor = prop.minor;
    
    // Verify compute capability >= 7.0 (Volta architecture or newer)
    if (prop.major < 7) {
        info->is_available = 0;
        return FLASH_CUDA_ERROR_INSUFFICIENT_CC;
    }
    
    // Extract memory information
    info->total_memory = prop.totalGlobalMem;
    
    // Get free memory
    size_t free_mem, total_mem;
    err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err == cudaSuccess) {
        info->free_memory = free_mem;
    } else {
        info->free_memory = 0;
    }
    
    // Extract multiprocessor count
    info->multiprocessor_count = prop.multiProcessorCount;
    
    // Extract max threads per block
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    
    // Mark as available
    info->is_available = 1;
    
    return FLASH_CUDA_SUCCESS;
}

// Context initialization
CudaContext* cuda_init_context(int device_id) {
    // Allocate context structure
    CudaContext* ctx = (CudaContext*)malloc(sizeof(CudaContext));
    if (!ctx) {
        return NULL;
    }
    
    // Initialize structure
    memset(ctx, 0, sizeof(CudaContext));
    ctx->error_code = FLASH_CUDA_SUCCESS;
    strcpy(ctx->error_msg, "No error");
    ctx->segment_workspace = NULL;
    ctx->segment_workspace_budget_bytes = 0;
    cuda_set_default_determinism_config(ctx);
    
    // Detect and verify device
    int result = cuda_detect_device(&ctx->device);
    if (result != FLASH_CUDA_SUCCESS) {
        ctx->error_code = result;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                "Failed to detect CUDA device: %s", cuda_get_error_string(result));
        return ctx;  // Return context with error state
    }
    
    // Set the device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        ctx->error_code = FLASH_CUDA_ERROR_NO_DEVICE;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                "Failed to set CUDA device %d: %s", device_id, cudaGetErrorString(err));
        return ctx;
    }
    
    // Create CUDA stream for asynchronous operations
    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        ctx->error_code = FLASH_CUDA_ERROR_KERNEL_FAILED;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                "Failed to create CUDA stream: %s", cudaGetErrorString(err));
        return ctx;
    }
    
    // Allocate pinned memory buffer for faster transfers (16 MB default)
    ctx->pinned_buffer_size = 16 * 1024 * 1024;  // 16 MB
    err = cudaMallocHost(&ctx->pinned_buffer, ctx->pinned_buffer_size);
    if (err != cudaSuccess) {
        // Non-fatal error, continue without pinned buffer
        ctx->pinned_buffer = NULL;
        ctx->pinned_buffer_size = 0;
        // Don't set error code, just log warning
        fprintf(stderr, "Warning: Failed to allocate pinned memory buffer: %s\n", 
                cudaGetErrorString(err));
    }

    // Initialize segment workspace memory budget.
    ctx->segment_workspace_budget_bytes =
        cuda_default_segment_workspace_budget_bytes(&ctx->device);

    // Initialize auxiliary streams to NULL.
    ctx->num_aux_streams = 0;
    for (int i = 0; i < FLASH_CUDA_MAX_AUX_STREAMS; i++) {
        ctx->aux_streams[i] = NULL;
    }
    
    return ctx;
}

// Context cleanup
void cuda_free_context(CudaContext* ctx) {
    if (!ctx) return;

    cuda_segment_workspace_reset(ctx);

    // Destroy auxiliary streams.
    cuda_destroy_aux_streams(ctx);
    
    // Free pinned memory buffer
    if (ctx->pinned_buffer) {
        cudaFreeHost(ctx->pinned_buffer);
        ctx->pinned_buffer = NULL;
    }
    
    // Destroy CUDA stream
    if (ctx->stream) {
        cudaStreamDestroy(ctx->stream);
        ctx->stream = NULL;
    }
    
    // Free context structure
    free(ctx);
}

cudaStream_t cuda_get_aux_stream(CudaContext* ctx, int index) {
    if (!ctx || index < 0 || index >= ctx->num_aux_streams) {
        return ctx ? ctx->stream : NULL;
    }
    return ctx->aux_streams[index];
}

int cuda_create_aux_streams(CudaContext* ctx, int count) {
    if (!ctx) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    if (count <= 0) {
        return FLASH_CUDA_SUCCESS;
    }
    if (count > FLASH_CUDA_MAX_AUX_STREAMS) {
        count = FLASH_CUDA_MAX_AUX_STREAMS;
    }
    // Destroy any existing auxiliary streams first.
    cuda_destroy_aux_streams(ctx);
    for (int i = 0; i < count; i++) {
        cudaError_t err = cudaStreamCreate(&ctx->aux_streams[i]);
        if (err != cudaSuccess) {
            // Destroy any streams we already created.
            for (int j = 0; j < i; j++) {
                cudaStreamDestroy(ctx->aux_streams[j]);
                ctx->aux_streams[j] = NULL;
            }
            ctx->num_aux_streams = 0;
            return FLASH_CUDA_ERROR_KERNEL_FAILED;
        }
    }
    ctx->num_aux_streams = count;
    return FLASH_CUDA_SUCCESS;
}

void cuda_destroy_aux_streams(CudaContext* ctx) {
    if (!ctx) {
        return;
    }
    for (int i = 0; i < ctx->num_aux_streams; i++) {
        if (ctx->aux_streams[i]) {
            cudaStreamDestroy(ctx->aux_streams[i]);
            ctx->aux_streams[i] = NULL;
        }
    }
    ctx->num_aux_streams = 0;
}

// Memory management functions

// Allocate device memory
void* cuda_malloc(size_t size, CudaContext* ctx) {
    if (!ctx || size == 0) {
        if (ctx) {
            ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
            strcpy(ctx->error_msg, "Invalid parameters for cuda_malloc");
        }
        return NULL;
    }
    
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size);
    
    if (err != cudaSuccess) {
        ctx->error_code = FLASH_CUDA_ERROR_OUT_OF_MEMORY;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                "Failed to allocate %zu bytes of GPU memory: %s", 
                size, cudaGetErrorString(err));
        return NULL;
    }
    
    return ptr;
}

// Free device memory
void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// Copy data from host to device
int cuda_memcpy_h2d(void* dst, const void* src, size_t size, CudaContext* ctx) {
    if (!ctx || !dst || !src || size == 0) {
        if (ctx) {
            ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
            strcpy(ctx->error_msg, "Invalid parameters for cuda_memcpy_h2d");
        }
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    cudaError_t err;
    
    // Use asynchronous copy with stream if available
    if (ctx->stream) {
        err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, ctx->stream);
    } else {
        err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    
    if (err != cudaSuccess) {
        ctx->error_code = FLASH_CUDA_ERROR_KERNEL_FAILED;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                "Failed to copy %zu bytes from host to device: %s", 
                size, cudaGetErrorString(err));
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

// Copy data from device to host
int cuda_memcpy_d2h(void* dst, const void* src, size_t size, CudaContext* ctx) {
    if (!ctx || !dst || !src || size == 0) {
        if (ctx) {
            ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
            strcpy(ctx->error_msg, "Invalid parameters for cuda_memcpy_d2h");
        }
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    cudaError_t err;
    
    // Use asynchronous copy with stream if available
    if (ctx->stream) {
        err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, ctx->stream);
        // Synchronize to ensure data is available on host
        cudaStreamSynchronize(ctx->stream);
    } else {
        err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    
    if (err != cudaSuccess) {
        ctx->error_code = FLASH_CUDA_ERROR_KERNEL_FAILED;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                "Failed to copy %zu bytes from device to host: %s", 
                size, cudaGetErrorString(err));
        return FLASH_CUDA_ERROR_KERNEL_FAILED;
    }
    
    return FLASH_CUDA_SUCCESS;
}

// Allocate pinned host memory for faster transfers
void* cuda_malloc_pinned(size_t size) {
    if (size == 0) return NULL;
    
    void* ptr = NULL;
    cudaError_t err = cudaMallocHost(&ptr, size);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate %zu bytes of pinned memory: %s\n", 
                size, cudaGetErrorString(err));
        return NULL;
    }
    
    return ptr;
}

// Free pinned host memory
void cuda_free_pinned(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

// Error handling utilities

// Convert error codes to human-readable messages
const char* cuda_get_error_string(int error_code) {
    switch (error_code) {
        case FLASH_CUDA_SUCCESS: 
            return "Success";
        case FLASH_CUDA_ERROR_NO_DEVICE: 
            return "No CUDA device found";
        case FLASH_CUDA_ERROR_INSUFFICIENT_CC: 
            return "Insufficient compute capability (requires >= 7.0)";
        case FLASH_CUDA_ERROR_OUT_OF_MEMORY: 
            return "Out of GPU memory";
        case FLASH_CUDA_ERROR_KERNEL_FAILED: 
            return "Kernel execution failed";
        case FLASH_CUDA_ERROR_INVALID_VALUE: 
            return "Invalid value or parameter";
        default: 
            return "Unknown error";
    }
}

// Check for kernel launch errors
int cuda_check_last_error(CudaContext* ctx) {
    if (!ctx) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    
    // Get last error from CUDA runtime
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        // Map CUDA error to our error codes
        int error_code;
        switch (err) {
            case cudaErrorMemoryAllocation:
                error_code = FLASH_CUDA_ERROR_OUT_OF_MEMORY;
                break;
            case cudaErrorInvalidValue:
            case cudaErrorInvalidDevicePointer:
            case cudaErrorInvalidMemcpyDirection:
                error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
                break;
            case cudaErrorNoDevice:
            case cudaErrorInsufficientDriver:
            case cudaErrorDeviceUninitialized:
                error_code = FLASH_CUDA_ERROR_NO_DEVICE;
                break;
            default:
                error_code = FLASH_CUDA_ERROR_KERNEL_FAILED;
                break;
        }
        
        // Store error information in context
        ctx->error_code = error_code;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                "CUDA error: %s", cudaGetErrorString(err));
        
        return error_code;
    }
    
    // No error
    ctx->error_code = FLASH_CUDA_SUCCESS;
    return FLASH_CUDA_SUCCESS;
}

// Kernel launch helpers

// Determine optimal thread block size based on number of elements
int cuda_get_optimal_block_size(int n_elements) {
    // For very small workloads, use smaller block sizes
    if (n_elements < 64) {
        return 32;
    } else if (n_elements < 256) {
        return 64;
    } else if (n_elements < 1024) {
        return 128;
    }
    
    // For larger workloads, use standard block sizes
    // 256 is a good default that balances occupancy and resource usage
    // It's a multiple of warp size (32) and works well for most kernels
    return 256;
}

// Calculate grid dimensions based on number of elements and block size
int cuda_get_grid_size(int n_elements, int block_size) {
    if (block_size <= 0) {
        return 0;
    }
    
    // Calculate number of blocks needed to cover all elements
    // Using ceiling division: (n + block_size - 1) / block_size
    return (n_elements + block_size - 1) / block_size;
}

void cuda_set_deterministic_mode(CudaContext* ctx, int enabled) {
    if (!ctx) {
        return;
    }
    ctx->deterministic_mode = (enabled != 0) ? 1 : 0;
}

int cuda_get_deterministic_mode(const CudaContext* ctx) {
    if (!ctx) {
        return 0;
    }
    return (ctx->deterministic_mode != 0) ? 1 : 0;
}

void cuda_configure_glszm_determinism(
    CudaContext* ctx,
    int tie_break_lowest_label,
    int deterministic_reduction,
    int allow_early_exit,
    int ccl_sync_interval,
    int ccl_flatten_interval
) {
    if (!ctx) {
        return;
    }

    if (tie_break_lowest_label >= 0) {
        ctx->glszm_ccl_tie_break_lowest_label = (tie_break_lowest_label != 0) ? 1 : 0;
    }
    if (deterministic_reduction >= 0) {
        ctx->glszm_deterministic_reduction = (deterministic_reduction != 0) ? 1 : 0;
    }
    if (allow_early_exit >= 0) {
        ctx->glszm_ccl_allow_early_exit = (allow_early_exit != 0) ? 1 : 0;
    }
    if (ccl_sync_interval > 0) {
        ctx->glszm_ccl_sync_interval = ccl_sync_interval;
    }
    if (ccl_flatten_interval > 0) {
        ctx->glszm_ccl_flatten_interval = ccl_flatten_interval;
    }

    if (ctx->glszm_ccl_sync_interval < 1) {
        ctx->glszm_ccl_sync_interval = 1;
    }
    if (ctx->glszm_ccl_flatten_interval < 1) {
        ctx->glszm_ccl_flatten_interval = 1;
    }
}

void cuda_get_glszm_determinism(
    const CudaContext* ctx,
    int* tie_break_lowest_label,
    int* deterministic_reduction,
    int* allow_early_exit,
    int* ccl_sync_interval,
    int* ccl_flatten_interval
) {
    if (tie_break_lowest_label) {
        *tie_break_lowest_label = 0;
    }
    if (deterministic_reduction) {
        *deterministic_reduction = 0;
    }
    if (allow_early_exit) {
        *allow_early_exit = 1;
    }
    if (ccl_sync_interval) {
        *ccl_sync_interval = FLASH_CUDA_DEFAULT_GLSZM_CCL_SYNC_INTERVAL;
    }
    if (ccl_flatten_interval) {
        *ccl_flatten_interval = FLASH_CUDA_DEFAULT_GLSZM_CCL_FLATTEN_INTERVAL;
    }
    if (!ctx) {
        return;
    }

    if (tie_break_lowest_label) {
        *tie_break_lowest_label = (ctx->glszm_ccl_tie_break_lowest_label != 0) ? 1 : 0;
    }
    if (deterministic_reduction) {
        *deterministic_reduction = (ctx->glszm_deterministic_reduction != 0) ? 1 : 0;
    }
    if (allow_early_exit) {
        *allow_early_exit = (ctx->glszm_ccl_allow_early_exit != 0) ? 1 : 0;
    }
    if (ccl_sync_interval) {
        *ccl_sync_interval = (ctx->glszm_ccl_sync_interval > 0)
            ? ctx->glszm_ccl_sync_interval
            : FLASH_CUDA_DEFAULT_GLSZM_CCL_SYNC_INTERVAL;
    }
    if (ccl_flatten_interval) {
        *ccl_flatten_interval = (ctx->glszm_ccl_flatten_interval > 0)
            ? ctx->glszm_ccl_flatten_interval
            : FLASH_CUDA_DEFAULT_GLSZM_CCL_FLATTEN_INTERVAL;
    }
}

size_t cuda_segment_workspace_required_bytes(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask
) {
    if (!img || !mask) {
        return 0;
    }
    if (img->ndim != mask->ndim ||
        img->dims[0] != mask->dims[0] ||
        img->dims[1] != mask->dims[1] ||
        img->dims[2] != mask->dims[2]) {
        return 0;
    }

    size_t n_voxels = (size_t)img->dims[0] * (size_t)img->dims[1] * (size_t)img->dims[2];
    if (n_voxels == 0) {
        return 0;
    }

    return n_voxels * (sizeof(float) + sizeof(uint8_t) + sizeof(int));
}

int cuda_segment_workspace_configure_budget(CudaContext* ctx, size_t budget_bytes) {
    if (!ctx) {
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }
    ctx->segment_workspace_budget_bytes = budget_bytes;
    if (ctx->segment_workspace) {
        ctx->segment_workspace->workspace_budget_bytes = budget_bytes;
    }
    return FLASH_CUDA_SUCCESS;
}

size_t cuda_segment_workspace_get_budget(const CudaContext* ctx) {
    if (!ctx) {
        return 0;
    }
    return ctx->segment_workspace_budget_bytes;
}

int cuda_segment_workspace_prepare(
    CudaContext* ctx,
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    double bin_width,
    int bin_count,
    int has_bin_minimum,
    double bin_minimum
) {
    CudaSegmentWorkspace* ws = NULL;
    size_t required_bytes = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    cudaError_t err = cudaSuccess;
    int n_voxels = 0;
    int block_size = 256;
    int grid_size = 0;

    if (!ctx || !img || !mask) {
        if (ctx) {
            ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Invalid parameters for cuda_segment_workspace_prepare");
        }
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }

    required_bytes = cuda_segment_workspace_required_bytes(img, mask);
    if (required_bytes == 0) {
        ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Invalid image/mask dimensions for segment workspace");
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }

    if (ctx->segment_workspace_budget_bytes > 0 &&
        required_bytes > ctx->segment_workspace_budget_bytes) {
        ctx->error_code = FLASH_CUDA_ERROR_OUT_OF_MEMORY;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Segment workspace needs %zu bytes but budget is %zu bytes",
                 required_bytes, ctx->segment_workspace_budget_bytes);
        return FLASH_CUDA_ERROR_OUT_OF_MEMORY;
    }

    ws = ctx->segment_workspace;
    if (!ws) {
        ws = (CudaSegmentWorkspace*)calloc(1, sizeof(CudaSegmentWorkspace));
        if (!ws) {
            ctx->error_code = FLASH_CUDA_ERROR_OUT_OF_MEMORY;
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate host segment workspace metadata");
            return FLASH_CUDA_ERROR_OUT_OF_MEMORY;
        }
        ctx->segment_workspace = ws;
    }

    if (cuda_segment_workspace_matches(
            ws, img, mask, bin_width, bin_count, has_bin_minimum, bin_minimum
        )) {
        return FLASH_CUDA_SUCCESS;
    }

    n_voxels = img->dims[0] * img->dims[1] * img->dims[2];
    if (n_voxels <= 0) {
        ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Segment workspace received empty volume");
        ws->valid = 0;
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }

    if ((size_t)n_voxels > ws->allocated_voxels ||
        !ws->d_image ||
        !ws->d_mask ||
        !ws->d_discretized) {
        if (ws->d_image) {
            cudaFree(ws->d_image);
            ws->d_image = NULL;
        }
        if (ws->d_mask) {
            cudaFree(ws->d_mask);
            ws->d_mask = NULL;
        }
        if (ws->d_discretized) {
            cudaFree(ws->d_discretized);
            ws->d_discretized = NULL;
        }

        err = cudaMalloc((void**)&ws->d_image, (size_t)n_voxels * sizeof(float));
        if (err != cudaSuccess) goto cuda_fail;
        err = cudaMalloc((void**)&ws->d_mask, (size_t)n_voxels * sizeof(uint8_t));
        if (err != cudaSuccess) goto cuda_fail;
        err = cudaMalloc((void**)&ws->d_discretized, (size_t)n_voxels * sizeof(int));
        if (err != cudaSuccess) goto cuda_fail;

        ws->allocated_voxels = (size_t)n_voxels;
        ws->allocated_bytes =
            ws->allocated_voxels * (sizeof(float) + sizeof(uint8_t) + sizeof(int));
    }

    err = cudaMemcpyAsync(
        ws->d_image,
        img->data,
        (size_t)n_voxels * sizeof(float),
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (err != cudaSuccess) goto cuda_fail;

    err = cudaMemcpyAsync(
        ws->d_mask,
        mask->data,
        (size_t)n_voxels * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (err != cudaSuccess) goto cuda_fail;

    if (!cuda_compute_mask_min_max_host(img, mask, &min_val, &max_val)) {
        ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Mask contains no ROI voxels for discretization");
        ws->valid = 0;
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }

    ws->discretization =
        cuda_resolve_discretization_descriptor(
            min_val, max_val, bin_width, bin_count, has_bin_minimum, bin_minimum
        );
    if (ws->discretization.num_bins <= 0) {
        ctx->error_code = FLASH_CUDA_ERROR_INVALID_VALUE;
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Invalid discretization configuration");
        ws->valid = 0;
        return FLASH_CUDA_ERROR_INVALID_VALUE;
    }

    grid_size = cuda_get_grid_size(n_voxels, block_size);
    if (grid_size > 1024) {
        grid_size = 1024;
    }
    cuda_discretize_segment_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        ws->d_image,
        ws->d_mask,
        ws->d_discretized,
        n_voxels,
        ws->discretization.mode,
        ws->discretization.num_bins,
        ws->discretization.min_val,
        ws->discretization.max_val,
        ws->discretization.bin_width,
        ws->discretization.min_bin
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cuda_fail;

    ws->ndim = img->ndim;
    ws->dims[0] = img->dims[0];
    ws->dims[1] = img->dims[1];
    ws->dims[2] = img->dims[2];
    ws->label = mask->label;
    ws->n_voxels = n_voxels;
    ws->required_bytes = required_bytes;
    ws->workspace_budget_bytes = ctx->segment_workspace_budget_bytes;
    ws->host_image_ptr = (uintptr_t)img->data;
    ws->host_mask_ptr = (uintptr_t)mask->data;
    ws->bin_width = bin_width;
    ws->bin_count = bin_count;
    ws->has_bin_minimum = (has_bin_minimum != 0) ? 1 : 0;
    ws->bin_minimum = bin_minimum;
    ws->valid = 1;

    ctx->error_code = FLASH_CUDA_SUCCESS;
    strcpy(ctx->error_msg, "No error");
    return FLASH_CUDA_SUCCESS;

cuda_fail:
    ws->valid = 0;
    ctx->error_code = cuda_workspace_error_from_runtime(err);
    snprintf(ctx->error_msg, sizeof(ctx->error_msg),
             "Failed to prepare segment workspace: %s", cudaGetErrorString(err));
    return ctx->error_code;
}

void cuda_segment_workspace_reset(CudaContext* ctx) {
    if (!ctx || !ctx->segment_workspace) {
        return;
    }

    if (ctx->segment_workspace->d_image) {
        cudaFree(ctx->segment_workspace->d_image);
    }
    if (ctx->segment_workspace->d_mask) {
        cudaFree(ctx->segment_workspace->d_mask);
    }
    if (ctx->segment_workspace->d_discretized) {
        cudaFree(ctx->segment_workspace->d_discretized);
    }
    free(ctx->segment_workspace);
    ctx->segment_workspace = NULL;
}

const float* cuda_segment_workspace_image(const CudaContext* ctx) {
    if (!ctx || !ctx->segment_workspace || !ctx->segment_workspace->valid) {
        return NULL;
    }
    return ctx->segment_workspace->d_image;
}

const uint8_t* cuda_segment_workspace_mask(const CudaContext* ctx) {
    if (!ctx || !ctx->segment_workspace || !ctx->segment_workspace->valid) {
        return NULL;
    }
    return ctx->segment_workspace->d_mask;
}

const int* cuda_segment_workspace_discretized(const CudaContext* ctx) {
    if (!ctx || !ctx->segment_workspace || !ctx->segment_workspace->valid) {
        return NULL;
    }
    return ctx->segment_workspace->d_discretized;
}

int cuda_segment_workspace_num_bins(const CudaContext* ctx) {
    if (!ctx || !ctx->segment_workspace || !ctx->segment_workspace->valid) {
        return 0;
    }
    return ctx->segment_workspace->discretization.num_bins;
}

int cuda_segment_workspace_voxel_count(const CudaContext* ctx) {
    if (!ctx || !ctx->segment_workspace || !ctx->segment_workspace->valid) {
        return 0;
    }
    return ctx->segment_workspace->n_voxels;
}

size_t cuda_segment_workspace_bytes(const CudaContext* ctx) {
    if (!ctx || !ctx->segment_workspace) {
        return 0;
    }
    return ctx->segment_workspace->required_bytes;
}

#else
// Stub implementations when CUDA is not available

int cuda_detect_device(CudaDeviceInfo* info) {
    if (info) {
        info->is_available = 0;
    }
    return FLASH_CUDA_ERROR_NO_DEVICE;
}

CudaContext* cuda_init_context(int device_id) {
    return NULL;
}

void cuda_free_context(CudaContext* ctx) {
}

void* cuda_malloc(size_t size, CudaContext* ctx) {
    return NULL;
}

void cuda_free(void* ptr) {
}

int cuda_memcpy_h2d(void* dst, const void* src, size_t size, CudaContext* ctx) {
    return FLASH_CUDA_ERROR_NO_DEVICE;
}

int cuda_memcpy_d2h(void* dst, const void* src, size_t size, CudaContext* ctx) {
    return FLASH_CUDA_ERROR_NO_DEVICE;
}

void* cuda_malloc_pinned(size_t size) {
    return NULL;
}

void cuda_free_pinned(void* ptr) {
}

const char* cuda_get_error_string(int error_code) {
    return "CUDA not available";
}

int cuda_check_last_error(CudaContext* ctx) {
    return FLASH_CUDA_ERROR_NO_DEVICE;
}

int cuda_get_optimal_block_size(int n_elements) {
    return 0;
}

int cuda_get_grid_size(int n_elements, int block_size) {
    return 0;
}

void cuda_set_deterministic_mode(CudaContext* ctx, int enabled) {
    (void)ctx;
    (void)enabled;
}

int cuda_get_deterministic_mode(const CudaContext* ctx) {
    (void)ctx;
    return 0;
}

void cuda_configure_glszm_determinism(
    CudaContext* ctx,
    int tie_break_lowest_label,
    int deterministic_reduction,
    int allow_early_exit,
    int ccl_sync_interval,
    int ccl_flatten_interval
) {
    (void)ctx;
    (void)tie_break_lowest_label;
    (void)deterministic_reduction;
    (void)allow_early_exit;
    (void)ccl_sync_interval;
    (void)ccl_flatten_interval;
}

void cuda_get_glszm_determinism(
    const CudaContext* ctx,
    int* tie_break_lowest_label,
    int* deterministic_reduction,
    int* allow_early_exit,
    int* ccl_sync_interval,
    int* ccl_flatten_interval
) {
    (void)ctx;
    if (tie_break_lowest_label) *tie_break_lowest_label = 0;
    if (deterministic_reduction) *deterministic_reduction = 0;
    if (allow_early_exit) *allow_early_exit = 1;
    if (ccl_sync_interval) *ccl_sync_interval = 16;
    if (ccl_flatten_interval) *ccl_flatten_interval = 8;
}

size_t cuda_segment_workspace_required_bytes(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask
) {
    return 0;
}

int cuda_segment_workspace_configure_budget(CudaContext* ctx, size_t budget_bytes) {
    return FLASH_CUDA_ERROR_NO_DEVICE;
}

size_t cuda_segment_workspace_get_budget(const CudaContext* ctx) {
    return 0;
}

int cuda_segment_workspace_prepare(
    CudaContext* ctx,
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    double bin_width,
    int bin_count,
    int has_bin_minimum,
    double bin_minimum
) {
    (void)has_bin_minimum;
    (void)bin_minimum;
    return FLASH_CUDA_ERROR_NO_DEVICE;
}

void cuda_segment_workspace_reset(CudaContext* ctx) {
}

const float* cuda_segment_workspace_image(const CudaContext* ctx) {
    return NULL;
}

const uint8_t* cuda_segment_workspace_mask(const CudaContext* ctx) {
    return NULL;
}

const int* cuda_segment_workspace_discretized(const CudaContext* ctx) {
    return NULL;
}

int cuda_segment_workspace_num_bins(const CudaContext* ctx) {
    return 0;
}

int cuda_segment_workspace_voxel_count(const CudaContext* ctx) {
    return 0;
}

size_t cuda_segment_workspace_bytes(const CudaContext* ctx) {
    return 0;
}

#endif // FLASH_RADIOMICS_CUDA_ENABLED
