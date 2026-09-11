#ifndef FLASH_RADIOMICS_CUDA_COMMON_H
#define FLASH_RADIOMICS_CUDA_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "../core/types.h"

// CUDA error codes
#define FLASH_CUDA_SUCCESS                0
#define FLASH_CUDA_ERROR_NO_DEVICE       -1
#define FLASH_CUDA_ERROR_INSUFFICIENT_CC -2
#define FLASH_CUDA_ERROR_OUT_OF_MEMORY   -3
#define FLASH_CUDA_ERROR_KERNEL_FAILED   -4
#define FLASH_CUDA_ERROR_INVALID_VALUE   -5

// Forward declarations for CUDA types
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
#include <cuda_runtime.h>
#else
// Dummy types when CUDA is not available
typedef void* cudaStream_t;
#endif

// Device information structure
typedef struct {
    int device_id;
    char device_name[256];
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    size_t free_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int is_available;
} CudaDeviceInfo;

// Maximum number of auxiliary streams for multi-class concurrent execution.
#define FLASH_CUDA_MAX_AUX_STREAMS 6

// CUDA context for execution
typedef struct CudaContext {
    CudaDeviceInfo device;
    cudaStream_t stream;
    void* pinned_buffer;
    size_t pinned_buffer_size;
    struct CudaSegmentWorkspace* segment_workspace;
    size_t segment_workspace_budget_bytes;
    int deterministic_mode;
    int glszm_deterministic_reduction;
    int glszm_ccl_tie_break_lowest_label;
    int glszm_ccl_allow_early_exit;
    int glszm_ccl_sync_interval;
    int glszm_ccl_flatten_interval;
    int has_bin_minimum;
    double bin_minimum;
    int error_code;
    char error_msg[256];
    // Multi-stream support for concurrent feature class execution.
    cudaStream_t aux_streams[FLASH_CUDA_MAX_AUX_STREAMS];
    int num_aux_streams;
} CudaContext;

typedef struct {
    int mode;
    int num_bins;
    int min_bin;
    float min_val;
    float max_val;
    float bin_width;
} CudaDiscretizationDescriptor;

typedef struct CudaSegmentWorkspace {
    float* d_image;
    uint8_t* d_mask;
    int* d_discretized;
    int ndim;
    int dims[3];
    int label;
    int n_voxels;
    size_t allocated_voxels;
    size_t allocated_bytes;
    size_t required_bytes;
    size_t workspace_budget_bytes;
    uintptr_t host_image_ptr;
    uintptr_t host_mask_ptr;
    double bin_width;
    int bin_count;
    int has_bin_minimum;
    double bin_minimum;
    CudaDiscretizationDescriptor discretization;
    int valid;
} CudaSegmentWorkspace;

// Memory pool for efficient memory management
typedef struct {
    // Host memory (CPU)
    float* h_image;
    uint8_t* h_mask;
    
    // Device memory (GPU)
    float* d_image;
    uint8_t* d_mask;
    int* d_histogram;
    float* d_glcm;
    int* d_glrlm;
    int* d_glszm;
    double* d_features;
    
    // Memory sizes
    size_t image_size;
    size_t mask_size;
    size_t histogram_size;
    
    // Allocation flags
    int image_allocated;
    int mask_allocated;
    int histogram_allocated;
} CudaMemoryPool;

// Kernel configuration for launch parameters
typedef struct {
    int grid_size_x;
    int grid_size_y;
    int grid_size_z;
    int block_size_x;
    int block_size_y;
    int block_size_z;
    size_t shared_mem_size;
    cudaStream_t stream;
} CudaKernelConfig;

// Device detection and initialization
int cuda_detect_device(CudaDeviceInfo* info);
CudaContext* cuda_init_context(int device_id);
void cuda_free_context(CudaContext* ctx);

// Memory management
void* cuda_malloc(size_t size, CudaContext* ctx);
void cuda_free(void* ptr);
int cuda_memcpy_h2d(void* dst, const void* src, size_t size, CudaContext* ctx);
int cuda_memcpy_d2h(void* dst, const void* src, size_t size, CudaContext* ctx);
void* cuda_malloc_pinned(size_t size);
void cuda_free_pinned(void* ptr);

// Error handling
const char* cuda_get_error_string(int error_code);
int cuda_check_last_error(CudaContext* ctx);

// Kernel launch helpers
int cuda_get_optimal_block_size(int n_elements);
int cuda_get_grid_size(int n_elements, int block_size);

// Deterministic execution controls.
void cuda_set_deterministic_mode(CudaContext* ctx, int enabled);
int cuda_get_deterministic_mode(const CudaContext* ctx);
void cuda_configure_glszm_determinism(
    CudaContext* ctx,
    int tie_break_lowest_label,
    int deterministic_reduction,
    int allow_early_exit,
    int ccl_sync_interval,
    int ccl_flatten_interval
);
void cuda_get_glszm_determinism(
    const CudaContext* ctx,
    int* tie_break_lowest_label,
    int* deterministic_reduction,
    int* allow_early_exit,
    int* ccl_sync_interval,
    int* ccl_flatten_interval
);

// Multi-stream helpers for concurrent voxel feature class execution.
cudaStream_t cuda_get_aux_stream(CudaContext* ctx, int index);
int cuda_create_aux_streams(CudaContext* ctx, int count);
void cuda_destroy_aux_streams(CudaContext* ctx);

// Segment-mode persistent workspace helpers
size_t cuda_segment_workspace_required_bytes(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask
);
int cuda_segment_workspace_configure_budget(CudaContext* ctx, size_t budget_bytes);
size_t cuda_segment_workspace_get_budget(const CudaContext* ctx);
int cuda_segment_workspace_prepare(
    CudaContext* ctx,
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    double bin_width,
    int bin_count,
    int has_bin_minimum,
    double bin_minimum
);
void cuda_segment_workspace_reset(CudaContext* ctx);
const float* cuda_segment_workspace_image(const CudaContext* ctx);
const uint8_t* cuda_segment_workspace_mask(const CudaContext* ctx);
const int* cuda_segment_workspace_discretized(const CudaContext* ctx);
int cuda_segment_workspace_num_bins(const CudaContext* ctx);
int cuda_segment_workspace_voxel_count(const CudaContext* ctx);
size_t cuda_segment_workspace_bytes(const CudaContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_CUDA_COMMON_H
