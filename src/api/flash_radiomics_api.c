#include "flash_radiomics_api.h"
#include "mps_detection.h"
#include "../core/utils.h"
#include "../features/firstorder_cpu.h"
#include "../features/glcm_cpu.h"
#include "../features/glrlm_cpu.h"
#include "../features/glszm_cpu.h"
#include "../features/gldm_cpu.h"
#include "../features/ngtdm_cpu.h"
#include "../features/shape_cpu.h"
#include <stdlib.h>
#include <string.h>

// Include MPS implementations
#ifdef __APPLE__
#include "../mps/firstorder_mps.h"
#include "../mps/glcm_mps.h"
#include "../mps/glrlm_mps.h"
#include "../mps/glszm_mps.h"
#endif

// Include CUDA implementations
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
#include "../cuda/cuda_common.h"
#include "../cuda/firstorder_cuda.h"
#include "../cuda/glcm_cuda.h"
#include "../cuda/glrlm_cuda.h"
#include "../cuda/glszm_cuda.h"
#include "../cuda/gldm_cuda.h"
#include "../cuda/ngtdm_cuda.h"
#include "../cuda/shape_cuda.h"
#endif

#ifdef FLASH_RADIOMICS_CUDA_ENABLED
#if defined(_MSC_VER)
#define FLASH_RADIOMICS_THREAD_LOCAL __declspec(thread)
#else
#define FLASH_RADIOMICS_THREAD_LOCAL _Thread_local
#endif

// Reuse one CUDA context per thread for segment-mode feature class calls.
// This avoids repeated stream/context creation for each feature class.
static FLASH_RADIOMICS_THREAD_LOCAL CudaContext* g_cached_cuda_ctx = NULL;
static FLASH_RADIOMICS_THREAD_LOCAL int g_cuda_deterministic_mode = 0;
static FLASH_RADIOMICS_THREAD_LOCAL int g_cuda_glszm_tie_break_lowest_label = 0;
static FLASH_RADIOMICS_THREAD_LOCAL int g_cuda_glszm_deterministic_reduction = 0;
static FLASH_RADIOMICS_THREAD_LOCAL int g_cuda_glszm_allow_early_exit = 1;
static FLASH_RADIOMICS_THREAD_LOCAL int g_cuda_glszm_ccl_sync_interval = 16;
static FLASH_RADIOMICS_THREAD_LOCAL int g_cuda_glszm_ccl_flatten_interval = 8;
static FLASH_RADIOMICS_THREAD_LOCAL int g_cuda_has_bin_minimum = 0;
static FLASH_RADIOMICS_THREAD_LOCAL double g_cuda_bin_minimum = 0.0;

static void flash_radiomics_apply_cuda_runtime_config(CudaContext* cuda_ctx) {
    if (!cuda_ctx) {
        return;
    }
    cuda_set_deterministic_mode(cuda_ctx, g_cuda_deterministic_mode);
    cuda_configure_glszm_determinism(
        cuda_ctx,
        g_cuda_glszm_tie_break_lowest_label,
        g_cuda_glszm_deterministic_reduction,
        g_cuda_glszm_allow_early_exit,
        g_cuda_glszm_ccl_sync_interval,
        g_cuda_glszm_ccl_flatten_interval
    );
    cuda_ctx->has_bin_minimum = g_cuda_has_bin_minimum;
    cuda_ctx->bin_minimum = g_cuda_bin_minimum;
}

static CudaContext* flash_radiomics_get_cached_cuda_context(void) {
    if (g_cached_cuda_ctx && g_cached_cuda_ctx->error_code == FLASH_CUDA_SUCCESS) {
        return g_cached_cuda_ctx;
    }

    if (g_cached_cuda_ctx) {
        cuda_free_context(g_cached_cuda_ctx);
        g_cached_cuda_ctx = NULL;
    }

    CudaContext* cuda_ctx = cuda_init_context(0);
    if (!cuda_ctx || cuda_ctx->error_code != FLASH_CUDA_SUCCESS) {
        if (cuda_ctx) {
            cuda_free_context(cuda_ctx);
        }
        return NULL;
    }

    flash_radiomics_apply_cuda_runtime_config(cuda_ctx);
    g_cached_cuda_ctx = cuda_ctx;
    return g_cached_cuda_ctx;
}

static void flash_radiomics_invalidate_cached_cuda_context(CudaContext* cuda_ctx) {
    if (!cuda_ctx) {
        return;
    }

    if (g_cached_cuda_ctx == cuda_ctx) {
        cuda_free_context(g_cached_cuda_ctx);
        g_cached_cuda_ctx = NULL;
        return;
    }

    cuda_free_context(cuda_ctx);
}
#endif

void flash_radiomics_cuda_reset_segment_workspace(void) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
    if (g_cached_cuda_ctx) {
        cuda_segment_workspace_reset(g_cached_cuda_ctx);
    }
#endif
}

size_t flash_radiomics_cuda_segment_workspace_required_bytes(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask
) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
    return cuda_segment_workspace_required_bytes(img, mask);
#else
    (void)img;
    (void)mask;
    return 0;
#endif
}

void flash_radiomics_cuda_set_deterministic_mode(int enabled) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
    g_cuda_deterministic_mode = (enabled != 0) ? 1 : 0;
    if (g_cached_cuda_ctx) {
        cuda_set_deterministic_mode(g_cached_cuda_ctx, g_cuda_deterministic_mode);
    }
#else
    (void)enabled;
#endif
}

int flash_radiomics_cuda_get_deterministic_mode(void) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
    return g_cuda_deterministic_mode;
#else
    return 0;
#endif
}

void flash_radiomics_cuda_configure_glszm_determinism(
    int tie_break_lowest_label,
    int deterministic_reduction,
    int allow_early_exit,
    int ccl_sync_interval,
    int ccl_flatten_interval
) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
    if (tie_break_lowest_label >= 0) {
        g_cuda_glszm_tie_break_lowest_label = (tie_break_lowest_label != 0) ? 1 : 0;
    }
    if (deterministic_reduction >= 0) {
        g_cuda_glszm_deterministic_reduction = (deterministic_reduction != 0) ? 1 : 0;
    }
    if (allow_early_exit >= 0) {
        g_cuda_glszm_allow_early_exit = (allow_early_exit != 0) ? 1 : 0;
    }
    if (ccl_sync_interval > 0) {
        g_cuda_glszm_ccl_sync_interval = ccl_sync_interval;
    }
    if (ccl_flatten_interval > 0) {
        g_cuda_glszm_ccl_flatten_interval = ccl_flatten_interval;
    }

    if (g_cuda_glszm_ccl_sync_interval < 1) {
        g_cuda_glszm_ccl_sync_interval = 1;
    }
    if (g_cuda_glszm_ccl_flatten_interval < 1) {
        g_cuda_glszm_ccl_flatten_interval = 1;
    }

    if (g_cached_cuda_ctx) {
        cuda_configure_glszm_determinism(
            g_cached_cuda_ctx,
            g_cuda_glszm_tie_break_lowest_label,
            g_cuda_glszm_deterministic_reduction,
            g_cuda_glszm_allow_early_exit,
            g_cuda_glszm_ccl_sync_interval,
            g_cuda_glszm_ccl_flatten_interval
        );
    }
#else
    (void)tie_break_lowest_label;
    (void)deterministic_reduction;
    (void)allow_early_exit;
    (void)ccl_sync_interval;
    (void)ccl_flatten_interval;
#endif
}

void flash_radiomics_cuda_set_bin_minimum(int enabled, double bin_minimum) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
    g_cuda_has_bin_minimum = (enabled != 0) ? 1 : 0;
    g_cuda_bin_minimum = bin_minimum;
    if (g_cached_cuda_ctx) {
        g_cached_cuda_ctx->has_bin_minimum = g_cuda_has_bin_minimum;
        g_cached_cuda_ctx->bin_minimum = g_cuda_bin_minimum;
    }
#else
    (void)enabled;
    (void)bin_minimum;
#endif
}

int flash_radiomics_detect_backends() {
    int backends = FLASH_RADIOMICS_BACKEND_CPU_AVAILABLE; // CPU always available
    
    // Check for MPS availability on Apple Silicon
    if (flash_radiomics_detect_mps_available()) {
        backends |= FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE;
    }
    
    // Check for CUDA availability
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
    CudaDeviceInfo cuda_info;
    if (cuda_detect_device(&cuda_info) == FLASH_CUDA_SUCCESS && cuda_info.is_available) {
        // Verify compute capability >= 7.0 (Volta or newer)
        if (cuda_info.compute_capability_major >= 7) {
            backends |= FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE;
        }
    }
#endif
    
    return backends;
}

static void flash_radiomics_configure_cpu_context(FlashRadiomicsContext* ctx) {
    ctx->backend = FLASH_RADIOMICS_CPU;
    ctx->num_threads = 1;
    ctx->device_handle = NULL;
    ctx->cuda_ctx = NULL;
    ctx->initialized = 1;
}

FlashRadiomicsContext* flash_radiomics_init_context(FlashRadiomicsBackend backend) {
    FlashRadiomicsContext* ctx = (FlashRadiomicsContext*)malloc(sizeof(FlashRadiomicsContext));
    if (!ctx) {
        return NULL;
    }
    
    // Initialize context fields
    ctx->backend = backend;
    ctx->num_threads = 1;
    ctx->device_handle = NULL;
    ctx->cuda_ctx = NULL;
    ctx->available_backends = flash_radiomics_detect_backends();
    ctx->initialized = 0;
    
    // Determine actual backend to use
    FlashRadiomicsBackend actual_backend = backend;
    if (backend == FLASH_RADIOMICS_AUTO) {
        // Auto-select: prefer CUDA > MPS > CPU
        if (ctx->available_backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else if (ctx->available_backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE) {
            actual_backend = FLASH_RADIOMICS_MPS;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
        ctx->backend = actual_backend;
    }
    
    // Initialize based on selected backend
    if (actual_backend == FLASH_RADIOMICS_CPU) {
        flash_radiomics_configure_cpu_context(ctx);
    } else if (actual_backend == FLASH_RADIOMICS_MPS) {
        // Initialize MPS backend
        if (!(ctx->available_backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE)) {
            // MPS requested but not available, fallback to CPU
            flash_radiomics_configure_cpu_context(ctx);
        } else {
            // Initialize Metal device
            void* device = flash_radiomics_create_mps_device();
            if (device) {
                ctx->device_handle = device;
                ctx->num_threads = 0; // Not used for MPS
                ctx->cuda_ctx = NULL;
                ctx->initialized = 1;
            } else {
                // Failed to create Metal device, fallback to CPU
                flash_radiomics_configure_cpu_context(ctx);
            }
        }
    } else if (actual_backend == FLASH_RADIOMICS_CUDA) {
        // Initialize CUDA backend
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
        if (!(ctx->available_backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE)) {
            // CUDA requested but not available, fallback to CPU
            flash_radiomics_configure_cpu_context(ctx);
        } else {
            // Initialize CUDA context
            CudaContext* cuda_ctx = cuda_init_context(0); // Use device 0
            if (cuda_ctx && cuda_ctx->error_code == FLASH_CUDA_SUCCESS) {
                flash_radiomics_apply_cuda_runtime_config(cuda_ctx);
                ctx->cuda_ctx = cuda_ctx;
                ctx->num_threads = 0; // Not used for CUDA
                ctx->device_handle = NULL;
                ctx->initialized = 1;
            } else {
                // Failed to create CUDA context, fallback to CPU
                if (cuda_ctx) {
                    cuda_free_context(cuda_ctx);
                }
                flash_radiomics_configure_cpu_context(ctx);
            }
        }
#else
        // CUDA not compiled in, fallback to CPU
        flash_radiomics_configure_cpu_context(ctx);
#endif
    }
    
    return ctx;
}

void flash_radiomics_free_context(FlashRadiomicsContext* ctx) {
    if (ctx) {
        // Clean up MPS device handle if present
        if (ctx->device_handle) {
            flash_radiomics_release_mps_device(ctx->device_handle);
            ctx->device_handle = NULL;
        }
        // Clean up CUDA context if present
        if (ctx->cuda_ctx) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
            cuda_free_context(ctx->cuda_ctx);
#endif
            ctx->cuda_ctx = NULL;
        }
        free(ctx);
    }
}

FlashRadiomicsResult* flash_radiomics_firstorder(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    double bin_width,
    int bin_count,
    double voxel_array_shift
) {
    // Automatic backend selection based on data size
    FlashRadiomicsBackend actual_backend = backend;
    
    if (backend == FLASH_RADIOMICS_AUTO) {
        // Calculate total voxels
        int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
        
        // Prefer CUDA > MPS > CPU for larger datasets
        int backends = flash_radiomics_detect_backends();
        if ((backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) && total_voxels > 100000) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else if ((backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE) && total_voxels > 1000000) {
            actual_backend = FLASH_RADIOMICS_MPS;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
    }
    
    // Execute on selected backend
    if (actual_backend == FLASH_RADIOMICS_CUDA) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
        // Check if CUDA is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE)) {
            // CUDA not available, fallback to CPU
            return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
        }
        
        // Reuse per-thread CUDA context.
        CudaContext* cuda_ctx = flash_radiomics_get_cached_cuda_context();
        if (!cuda_ctx) {
            return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
        }
        
        // Execute on CUDA
        FlashRadiomicsResult* result = flash_radiomics_firstorder_cuda(img, mask, cuda_ctx, bin_width, bin_count, voxel_array_shift);
        
        // Check for CUDA errors and fallback if needed
        if (!result || result->error_code != FLASH_RADIOMICS_SUCCESS) {
            if (result) {
                flash_radiomics_free_result(result);
            }
            flash_radiomics_invalidate_cached_cuda_context(cuda_ctx);
            // Fallback to CPU
            return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
        }

        return result;
#else
        // CUDA not compiled in, fallback to CPU
        return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
#endif
    }
    else if (actual_backend == FLASH_RADIOMICS_CPU) {
        return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
    }
    else if (actual_backend == FLASH_RADIOMICS_MPS) {
#if defined(__APPLE__) && defined(FLASH_RADIOMICS_MPS_ENABLED)
        // Check if MPS is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE)) {
            // MPS not available, fallback to CPU
            return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
        }
        
        // Create MPS device
        void* device = flash_radiomics_create_mps_device();
        if (!device) {
            // Failed to create device, fallback to CPU
            return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
        }
        
        // Execute on MPS
        FlashRadiomicsResult* result = flash_radiomics_firstorder_mps(img, mask, device, bin_width, bin_count, voxel_array_shift);
        
        // Release device
        flash_radiomics_release_mps_device(device);
        
        return result;
#else
        // MPS not available, fallback to CPU
        return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
#endif
    }
    
    // Default fallback to CPU
    return flash_radiomics_firstorder_cpu(img, mask, num_threads, bin_width, bin_count, voxel_array_shift);
}

FlashRadiomicsResult* flash_radiomics_glcm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
) {
    // Automatic backend selection based on data size
    FlashRadiomicsBackend actual_backend = backend;
    
    if (backend == FLASH_RADIOMICS_AUTO) {
        // Calculate total voxels
        int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
        
        // Prefer CUDA > MPS > CPU for larger datasets
        int backends = flash_radiomics_detect_backends();
        if ((backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) && total_voxels > 100000) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else if ((backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE) && total_voxels > 1000000) {
            actual_backend = FLASH_RADIOMICS_MPS;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
    }
    
    // Execute on selected backend
    if (actual_backend == FLASH_RADIOMICS_CUDA) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
        // Check if CUDA is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE)) {
            // CUDA not available, fallback to CPU
            return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
        }
        
        // Reuse per-thread CUDA context.
        CudaContext* cuda_ctx = flash_radiomics_get_cached_cuda_context();
        if (!cuda_ctx) {
            return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
        }
        
        // Execute on CUDA
        FlashRadiomicsResult* result = flash_radiomics_glcm_cuda(img, mask, cuda_ctx, distances, num_distances, bin_width, bin_count);
        
        // Check for CUDA errors and fallback if needed
        if (!result || result->error_code != FLASH_RADIOMICS_SUCCESS) {
            if (result) {
                flash_radiomics_free_result(result);
            }
            flash_radiomics_invalidate_cached_cuda_context(cuda_ctx);
            // Fallback to CPU
            return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
        }

        return result;
#else
        // CUDA not compiled in, fallback to CPU
        return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
#endif
    }
    else if (actual_backend == FLASH_RADIOMICS_CPU) {
        return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
    }
    else if (actual_backend == FLASH_RADIOMICS_MPS) {
#if defined(__APPLE__) && defined(FLASH_RADIOMICS_MPS_ENABLED)
        // Check if MPS is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE)) {
            // MPS not available, fallback to CPU
            return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
        }
        
        // Create MPS device
        void* device = flash_radiomics_create_mps_device();
        if (!device) {
            // Failed to create device, fallback to CPU
            return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
        }
        
        // Execute on MPS
        FlashRadiomicsResult* result = flash_radiomics_glcm_mps(img, mask, device, distances, num_distances, bin_width, bin_count);
        
        // Release device
        flash_radiomics_release_mps_device(device);
        
        return result;
#else
        // MPS not available, fallback to CPU
        return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
#endif
    }
    
    // Default fallback to CPU
    return flash_radiomics_glcm_cpu(img, mask, num_threads, distances, num_distances, bin_width, bin_count);
}

FlashRadiomicsResult* flash_radiomics_glrlm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    double bin_width,
    int bin_count
) {
    // Automatic backend selection based on data size
    FlashRadiomicsBackend actual_backend = backend;
    
    if (backend == FLASH_RADIOMICS_AUTO) {
        // Calculate total voxels
        int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
        
        // Prefer CUDA > MPS > CPU for larger datasets
        int backends = flash_radiomics_detect_backends();
        if ((backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) && total_voxels > 100000) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else if ((backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE) && total_voxels > 1000000) {
            actual_backend = FLASH_RADIOMICS_MPS;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
    }
    
    // Execute on selected backend
    if (actual_backend == FLASH_RADIOMICS_CUDA) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
        // Check if CUDA is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE)) {
            // CUDA not available, fallback to CPU
            return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Reuse per-thread CUDA context.
        CudaContext* cuda_ctx = flash_radiomics_get_cached_cuda_context();
        if (!cuda_ctx) {
            return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Execute on CUDA
        FlashRadiomicsResult* result = flash_radiomics_glrlm_cuda(img, mask, cuda_ctx, bin_width, bin_count);
        
        // Check for CUDA errors and fallback if needed
        if (!result || result->error_code != FLASH_RADIOMICS_SUCCESS) {
            if (result) {
                flash_radiomics_free_result(result);
            }
            flash_radiomics_invalidate_cached_cuda_context(cuda_ctx);
            // Fallback to CPU
            return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
        }

        return result;
#else
        // CUDA not compiled in, fallback to CPU
        return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
#endif
    }
    else if (actual_backend == FLASH_RADIOMICS_CPU) {
        return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
    }
    else if (actual_backend == FLASH_RADIOMICS_MPS) {
#if defined(__APPLE__) && defined(FLASH_RADIOMICS_MPS_ENABLED)
        // Check if MPS is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE)) {
            // MPS not available, fallback to CPU
            return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Create MPS device
        void* device = flash_radiomics_create_mps_device();
        if (!device) {
            // Failed to create device, fallback to CPU
            return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Execute on MPS
        FlashRadiomicsResult* result = flash_radiomics_glrlm_mps(img, mask, device, bin_width, bin_count);
        
        // Release device
        flash_radiomics_release_mps_device(device);
        
        return result;
#else
        // MPS not available, fallback to CPU
        return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
#endif
    }
    
    // Default fallback to CPU
    return flash_radiomics_glrlm_cpu(img, mask, num_threads, bin_width, bin_count);
}

FlashRadiomicsResult* flash_radiomics_glszm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    double bin_width,
    int bin_count
) {
    // Automatic backend selection based on data size
    FlashRadiomicsBackend actual_backend = backend;
    
    if (backend == FLASH_RADIOMICS_AUTO) {
        // Calculate total voxels
        int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
        
        // Prefer CUDA > MPS > CPU for larger datasets
        int backends = flash_radiomics_detect_backends();
        if ((backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) && total_voxels > 100000) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else if ((backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE) && total_voxels > 1000000) {
            actual_backend = FLASH_RADIOMICS_MPS;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
    }
    
    // Execute on selected backend
    if (actual_backend == FLASH_RADIOMICS_CUDA) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
        // Check if CUDA is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE)) {
            // CUDA not available, fallback to CPU
            return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Reuse per-thread CUDA context.
        CudaContext* cuda_ctx = flash_radiomics_get_cached_cuda_context();
        if (!cuda_ctx) {
            return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Execute on CUDA
        FlashRadiomicsResult* result = flash_radiomics_glszm_cuda(img, mask, cuda_ctx, bin_width, bin_count);
        
        // Check for CUDA errors and fallback if needed
        if (!result || result->error_code != FLASH_RADIOMICS_SUCCESS) {
            if (result) {
                flash_radiomics_free_result(result);
            }
            flash_radiomics_invalidate_cached_cuda_context(cuda_ctx);
            // Fallback to CPU
            return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
        }

        return result;
#else
        // CUDA not compiled in, fallback to CPU
        return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
#endif
    }
    else if (actual_backend == FLASH_RADIOMICS_CPU) {
        return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
    }
    else if (actual_backend == FLASH_RADIOMICS_MPS) {
#if defined(__APPLE__) && defined(FLASH_RADIOMICS_MPS_ENABLED)
        // Check if MPS is available
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE)) {
            // MPS not available, fallback to CPU
            return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Create MPS device
        void* device = flash_radiomics_create_mps_device();
        if (!device) {
            // Failed to create device, fallback to CPU
            return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
        }
        
        // Execute on MPS
        FlashRadiomicsResult* result = flash_radiomics_glszm_mps(img, mask, device, bin_width, bin_count);
        
        // Release device
        flash_radiomics_release_mps_device(device);
        
        return result;
#else
        // MPS not available, fallback to CPU
        return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
#endif
    }
    
    // Default fallback to CPU
    return flash_radiomics_glszm_cpu(img, mask, num_threads, bin_width, bin_count);
}

FlashRadiomicsResult* flash_radiomics_gldm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    double gldm_a,
    int force_2d
) {
    FlashRadiomicsBackend actual_backend = backend;

    if (backend == FLASH_RADIOMICS_AUTO) {
        int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
        int backends = flash_radiomics_detect_backends();
        if ((backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) && total_voxels > 100000) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
    }

    if (actual_backend == FLASH_RADIOMICS_CUDA) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE)) {
            return flash_radiomics_gldm_cpu(
                img, mask, num_threads, distances, num_distances, bin_width, bin_count, gldm_a
            );
        }

        CudaContext* cuda_ctx = flash_radiomics_get_cached_cuda_context();
        if (!cuda_ctx) {
            return flash_radiomics_gldm_cpu(
                img, mask, num_threads, distances, num_distances, bin_width, bin_count, gldm_a
            );
        }

        FlashRadiomicsResult* result = flash_radiomics_gldm_cuda(
            img,
            mask,
            cuda_ctx,
            distances,
            num_distances,
            bin_width,
            bin_count,
            gldm_a,
            force_2d
        );

        if (!result || result->error_code != FLASH_RADIOMICS_SUCCESS) {
            if (result) {
                flash_radiomics_free_result(result);
            }
            flash_radiomics_invalidate_cached_cuda_context(cuda_ctx);
            return flash_radiomics_gldm_cpu(
                img, mask, num_threads, distances, num_distances, bin_width, bin_count, gldm_a
            );
        }
        return result;
#else
        return flash_radiomics_gldm_cpu(
            img, mask, num_threads, distances, num_distances, bin_width, bin_count, gldm_a
        );
#endif
    }

    return flash_radiomics_gldm_cpu(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        bin_width,
        bin_count,
        gldm_a
    );
}

FlashRadiomicsResult* flash_radiomics_ngtdm(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    FlashRadiomicsBackend backend,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    int force_2d
) {
    FlashRadiomicsBackend actual_backend = backend;

    if (backend == FLASH_RADIOMICS_AUTO) {
        int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
        int backends = flash_radiomics_detect_backends();
        if ((backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) && total_voxels > 100000) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
    }

    if (actual_backend == FLASH_RADIOMICS_CUDA) {
#ifdef FLASH_RADIOMICS_CUDA_ENABLED
        int backends = flash_radiomics_detect_backends();
        if (!(backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE)) {
            return flash_radiomics_ngtdm_cpu(
                img, mask, num_threads, distances, num_distances, bin_width, bin_count
            );
        }

        CudaContext* cuda_ctx = flash_radiomics_get_cached_cuda_context();
        if (!cuda_ctx) {
            return flash_radiomics_ngtdm_cpu(
                img, mask, num_threads, distances, num_distances, bin_width, bin_count
            );
        }

        FlashRadiomicsResult* result = flash_radiomics_ngtdm_cuda(
            img,
            mask,
            cuda_ctx,
            distances,
            num_distances,
            bin_width,
            bin_count,
            force_2d
        );

        if (!result || result->error_code != FLASH_RADIOMICS_SUCCESS) {
            if (result) {
                flash_radiomics_free_result(result);
            }
            flash_radiomics_invalidate_cached_cuda_context(cuda_ctx);
            return flash_radiomics_ngtdm_cpu(
                img, mask, num_threads, distances, num_distances, bin_width, bin_count
            );
        }
        return result;
#else
        return flash_radiomics_ngtdm_cpu(
            img, mask, num_threads, distances, num_distances, bin_width, bin_count
        );
#endif
    }

    return flash_radiomics_ngtdm_cpu(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        bin_width,
        bin_count
    );
}

FlashRadiomicsResult* flash_radiomics_shape(
    FlashRadiomicsMask* mask,
    float* spacing,
    FlashRadiomicsBackend backend
) {
    // Automatic backend selection based on data size
    FlashRadiomicsBackend actual_backend = backend;
    
    if (backend == FLASH_RADIOMICS_AUTO) {
        // Calculate total voxels
        int total_voxels = mask->dims[0] * mask->dims[1] * mask->dims[2];
        
        // Prefer CUDA > CPU for larger datasets (shape doesn't have MPS implementation)
        int backends = flash_radiomics_detect_backends();
        if ((backends & FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE) && total_voxels > 100000) {
            actual_backend = FLASH_RADIOMICS_CUDA;
        } else {
            actual_backend = FLASH_RADIOMICS_CPU;
        }
    }
    
    // Execute on selected backend
    if (actual_backend == FLASH_RADIOMICS_CUDA) {
        // Current CUDA shape wrapper calls CPU implementation. Route directly
        // to CPU to avoid redundant CUDA setup overhead.
        return flash_radiomics_shape_cpu(mask, spacing);
    }
    
    // Default to CPU (shape doesn't have MPS implementation)
    return flash_radiomics_shape_cpu(mask, spacing);
}

FlashRadiomicsResult* flash_radiomics_shape2d(
    FlashRadiomicsMask* mask,
    float* spacing,
    FlashRadiomicsBackend backend
) {
    (void)backend;
    return flash_radiomics_shape2d_cpu(mask, spacing);
}

void flash_radiomics_free_result(FlashRadiomicsResult* result) {
    if (result) {
        if (result->features) {
            free(result->features);
        }
        free(result);
    }
}
