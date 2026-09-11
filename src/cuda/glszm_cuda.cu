#include "glszm_cuda.h"
#include "cuda_common.h"
#include "../core/types.h"
#include "../core/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef FLASH_RADIOMICS_CUDA_ENABLED

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define EPSILON 2.2e-16
#define CUDA_BLOCK_SIZE 256
#define GLSZM_CCL_SYNC_INTERVAL 16
#define GLSZM_CCL_FLATTEN_INTERVAL 8

__device__ __forceinline__ int ccl_find_root(const int* labels, int label) {
    int root = label;
    while (root >= 0) {
        int parent = labels[root];
        if (parent < 0 || parent == root) {
            break;
        }
        root = parent;
    }
    return root;
}

// Task 7.1: Initialize labels for connected component labeling
__global__ void ccl_init_kernel(
    const uint8_t* mask,
    int* labels,
    int n_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (mask[idx] != 0) {
            labels[idx] = idx;  // Each voxel starts with its own label
        } else {
            labels[idx] = -1;   // Outside ROI
        }
    }
}

// Task 7.1: Merge labels for connected components (2D - 4-connectivity)
__global__ void ccl_merge_kernel_2d(
    const int* discretized,
    const uint8_t* mask,
    int* labels,
    int* changed,
    int tie_break_lowest_label,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    if (mask[idx] == 0) return;
    
    int my_label = labels[idx];
    if (my_label < 0) return;
    int my_root = ccl_find_root(labels, my_label);
    int my_gray = discretized[idx];
    int min_label = my_root;
    
    // Check 4 neighbors (right, down, left, up)
    int dx[4] = {1, 0, -1, 0};
    int dy[4] = {0, 1, 0, -1};
    
    for (int d = 0; d < 4; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int nidx = ny * width + nx;
            if (tie_break_lowest_label && nidx > idx) {
                continue;
            }
            
            if (mask[nidx] != 0 && discretized[nidx] == my_gray) {
                int neighbor_label = labels[nidx];
                if (neighbor_label < 0) {
                    continue;
                }
                int neighbor_root = ccl_find_root(labels, neighbor_label);
                if (neighbor_root < 0) {
                    continue;
                }
                if (neighbor_root < min_label) {
                    min_label = neighbor_root;
                }
            }
        }
    }
    
    if (min_label < my_root) {
        int prev = atomicMin(&labels[my_root], min_label);
        if (min_label < prev) {
            atomicExch(changed, 1);
        }
    }
}

// Task 7.1: Merge labels for connected components (3D - 26-connectivity)
__global__ void ccl_merge_kernel_3d(
    const int* discretized,
    const uint8_t* mask,
    int* labels,
    int* changed,
    int tie_break_lowest_label,
    int width,
    int height,
    int depth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int idx = z * height * width + y * width + x;
    
    if (mask[idx] == 0) return;
    
    int my_label = labels[idx];
    if (my_label < 0) return;
    int my_root = ccl_find_root(labels, my_label);
    int my_gray = discretized[idx];
    int min_label = my_root;
    
    // Check 26 neighbors
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height && 
                    nz >= 0 && nz < depth) {
                    int nidx = nz * height * width + ny * width + nx;
                    if (tie_break_lowest_label && nidx > idx) {
                        continue;
                    }
                    
                    if (mask[nidx] != 0 && discretized[nidx] == my_gray) {
                        int neighbor_label = labels[nidx];
                        if (neighbor_label < 0) {
                            continue;
                        }
                        int neighbor_root = ccl_find_root(labels, neighbor_label);
                        if (neighbor_root < 0) {
                            continue;
                        }
                        if (neighbor_root < min_label) {
                            min_label = neighbor_root;
                        }
                    }
                }
            }
        }
    }
    
    if (min_label < my_root) {
        int prev = atomicMin(&labels[my_root], min_label);
        if (min_label < prev) {
            atomicExch(changed, 1);
        }
    }
}

// Task 7.1: Flatten label tree (path compression)
__global__ void ccl_flatten_kernel(
    int* labels,
    int n_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (labels[idx] >= 0) {
            // Follow the chain to find root
            int root = labels[idx];
            while (root != labels[root]) {
                root = labels[root];
            }
            labels[idx] = root;
        }
    }
}

// Task 7.2: Compute zone sizes
__global__ void compute_zone_sizes_kernel(
    const int* labels,
    const uint8_t* mask,
    int* zone_sizes,
    int n_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (mask[idx] != 0 && labels[idx] >= 0) {
            atomicAdd(&zone_sizes[labels[idx]], 1);
        }
    }
}

// Task 7.3: Build GLSZM matrix from zones
__global__ void build_glszm_kernel(
    const int* discretized,
    const int* labels,
    int* zone_sizes,
    const uint8_t* mask,
    int* glszm_matrix,
    int num_bins,
    int max_zone_size,
    int n_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (mask[idx] != 0 && labels[idx] >= 0) {
            int label = labels[idx];
            int gray_level = discretized[idx];
            int zone_size = zone_sizes[label];
            
            if (zone_size > 0 && zone_size <= max_zone_size && gray_level >= 0 && gray_level < num_bins) {
                // Only the root voxel of each connected component contributes one zone.
                if (label == idx) {
                    int matrix_idx = gray_level * max_zone_size + (zone_size - 1);
                    atomicAdd(&glszm_matrix[matrix_idx], 1);
                }
            }
        }
    }
}

// Simplified version: Mark zone representatives first, then build matrix
__global__ void mark_zone_representatives_kernel(
    const int* labels,
    const uint8_t* mask,
    int* zone_representatives,
    int n_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (mask[idx] != 0 && labels[idx] >= 0) {
            int label = labels[idx];
            atomicMin(&zone_representatives[label], idx);
        }
    }
}

__global__ void build_glszm_from_representatives_kernel(
    const int* discretized,
    const int* labels,
    const int* zone_sizes,
    const int* zone_representatives,
    const uint8_t* mask,
    int* glszm_matrix,
    int num_bins,
    int max_zone_size,
    int n_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_voxels) {
        if (mask[idx] != 0 && labels[idx] >= 0) {
            int label = labels[idx];
            
            // Only process if this is the representative voxel for this zone
            if (zone_representatives[label] == idx) {
                int gray_level = discretized[idx];
                int zone_size = zone_sizes[label];
                
                if (zone_size > 0 && zone_size <= max_zone_size && 
                    gray_level >= 0 && gray_level < num_bins) {
                    int matrix_idx = gray_level * max_zone_size + (zone_size - 1);
                    atomicAdd(&glszm_matrix[matrix_idx], 1);
                }
            }
        }
    }
}

enum {
    GLSZM_ACC_TOTAL = 0,
    GLSZM_ACC_SAE = 1,
    GLSZM_ACC_LAE = 2,
    GLSZM_ACC_LGZE = 3,
    GLSZM_ACC_HGZE = 4,
    GLSZM_ACC_SALGLE = 5,
    GLSZM_ACC_SAHGLE = 6,
    GLSZM_ACC_LALGLE = 7,
    GLSZM_ACC_LAHGLE = 8,
    GLSZM_ACC_GL_WEIGHTED = 9,
    GLSZM_ACC_GL2_WEIGHTED = 10,
    GLSZM_ACC_SZ_WEIGHTED = 11,
    GLSZM_ACC_SZ2_WEIGHTED = 12,
    GLSZM_ACC_PLOGP = 13,
    GLSZM_ACC_COUNT = 14
};

// Task 7.4: Compute GLSZM row/column marginals and feature numerators.
// Uses global buffers so max_zone_size can exceed 256 safely.
__global__ void compute_glszm_accumulators_kernel(
    const int* glszm_matrix,
    int num_bins,
    int max_zone_size,
    size_t matrix_size,
    double* row_sums,
    double* col_sums,
    double* accum
) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;

    for (size_t i = idx; i < matrix_size; i += stride) {
        int count = glszm_matrix[i];
        if (count <= 0) {
            continue;
        }

        int gray_level = (int)(i / (size_t)max_zone_size);
        int zone_size_idx = (int)(i % (size_t)max_zone_size);
        if (gray_level < 0 || gray_level >= num_bins || zone_size_idx < 0 || zone_size_idx >= max_zone_size) {
            continue;
        }

        double p_ij = (double)count;
        double gl = (double)(gray_level + 1);
        double sz = (double)(zone_size_idx + 1);
        double gl_sq = gl * gl;
        double sz_sq = sz * sz;

        atomicAdd(&row_sums[gray_level], p_ij);
        atomicAdd(&col_sums[zone_size_idx], p_ij);

        atomicAdd(&accum[GLSZM_ACC_TOTAL], p_ij);
        atomicAdd(&accum[GLSZM_ACC_SAE], p_ij / sz_sq);
        atomicAdd(&accum[GLSZM_ACC_LAE], p_ij * sz_sq);
        atomicAdd(&accum[GLSZM_ACC_LGZE], p_ij / gl_sq);
        atomicAdd(&accum[GLSZM_ACC_HGZE], p_ij * gl_sq);
        atomicAdd(&accum[GLSZM_ACC_SALGLE], p_ij / (gl_sq * sz_sq));
        atomicAdd(&accum[GLSZM_ACC_SAHGLE], p_ij * gl_sq / sz_sq);
        atomicAdd(&accum[GLSZM_ACC_LALGLE], p_ij * sz_sq / gl_sq);
        atomicAdd(&accum[GLSZM_ACC_LAHGLE], p_ij * gl_sq * sz_sq);
        atomicAdd(&accum[GLSZM_ACC_GL_WEIGHTED], gl * p_ij);
        atomicAdd(&accum[GLSZM_ACC_GL2_WEIGHTED], gl_sq * p_ij);
        atomicAdd(&accum[GLSZM_ACC_SZ_WEIGHTED], sz * p_ij);
        atomicAdd(&accum[GLSZM_ACC_SZ2_WEIGHTED], sz_sq * p_ij);
        atomicAdd(&accum[GLSZM_ACC_PLOGP], p_ij * log2(p_ij));
    }
}

__global__ void compute_glszm_accumulators_deterministic_kernel(
    const int* glszm_matrix,
    int num_bins,
    int max_zone_size,
    double* accum,
    double* gln_numerator,
    double* szn_numerator
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    double gln = 0.0;
    double szn = 0.0;
    for (int gl_idx = 0; gl_idx < num_bins; gl_idx++) {
        double row_sum = 0.0;
        for (int sz_idx = 0; sz_idx < max_zone_size; sz_idx++) {
            int count = glszm_matrix[(size_t)gl_idx * (size_t)max_zone_size + (size_t)sz_idx];
            if (count <= 0) {
                continue;
            }
            double p_ij = (double)count;
            double gl = (double)(gl_idx + 1);
            double sz = (double)(sz_idx + 1);
            double gl_sq = gl * gl;
            double sz_sq = sz * sz;

            row_sum += p_ij;
            accum[GLSZM_ACC_TOTAL] += p_ij;
            accum[GLSZM_ACC_SAE] += p_ij / sz_sq;
            accum[GLSZM_ACC_LAE] += p_ij * sz_sq;
            accum[GLSZM_ACC_LGZE] += p_ij / gl_sq;
            accum[GLSZM_ACC_HGZE] += p_ij * gl_sq;
            accum[GLSZM_ACC_SALGLE] += p_ij / (gl_sq * sz_sq);
            accum[GLSZM_ACC_SAHGLE] += p_ij * gl_sq / sz_sq;
            accum[GLSZM_ACC_LALGLE] += p_ij * sz_sq / gl_sq;
            accum[GLSZM_ACC_LAHGLE] += p_ij * gl_sq * sz_sq;
            accum[GLSZM_ACC_GL_WEIGHTED] += gl * p_ij;
            accum[GLSZM_ACC_GL2_WEIGHTED] += gl_sq * p_ij;
            accum[GLSZM_ACC_SZ_WEIGHTED] += sz * p_ij;
            accum[GLSZM_ACC_SZ2_WEIGHTED] += sz_sq * p_ij;
            accum[GLSZM_ACC_PLOGP] += p_ij * log2(p_ij);
        }
        gln += row_sum * row_sum;
    }

    for (int sz_idx = 0; sz_idx < max_zone_size; sz_idx++) {
        double col_sum = 0.0;
        for (int gl_idx = 0; gl_idx < num_bins; gl_idx++) {
            int count = glszm_matrix[(size_t)gl_idx * (size_t)max_zone_size + (size_t)sz_idx];
            if (count > 0) {
                col_sum += (double)count;
            }
        }
        szn += col_sum * col_sum;
    }

    gln_numerator[0] = gln;
    szn_numerator[0] = szn;
}

__global__ void reduce_max_int_kernel(const int* values, int n, int* out_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        int v = values[i];
        if (v > 0) {
            atomicMax(out_max, v);
        }
    }
}

__global__ void count_mask_voxels_kernel(const uint8_t* mask, int n_voxels, int* out_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n_voxels; i += stride) {
        if (mask[i] != 0) {
            atomicAdd(out_count, 1);
        }
    }
}

__global__ void reduce_square_sum_kernel(const double* values, int n, double* out_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        double v = values[i];
        atomicAdd(out_sum, v * v);
    }
}

__global__ void finalize_glszm_features_kernel(
    const double* accum,
    const double* gln_numerator,
    const double* szn_numerator,
    const int* num_voxels_in_roi,
    double* features
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (int i = 0; i < 16; i++) {
        features[i] = 0.0;
    }

    double total_sum = accum[GLSZM_ACC_TOTAL];
    if (total_sum <= EPSILON) {
        return;
    }

    double gln = gln_numerator[0];
    double szn = szn_numerator[0];

    features[0] = accum[GLSZM_ACC_SAE] / total_sum;
    features[1] = accum[GLSZM_ACC_LAE] / total_sum;
    features[2] = accum[GLSZM_ACC_LGZE] / total_sum;
    features[3] = accum[GLSZM_ACC_HGZE] / total_sum;
    features[4] = accum[GLSZM_ACC_SALGLE] / total_sum;
    features[5] = accum[GLSZM_ACC_SAHGLE] / total_sum;
    features[6] = accum[GLSZM_ACC_LALGLE] / total_sum;
    features[7] = accum[GLSZM_ACC_LAHGLE] / total_sum;
    features[8] = gln / total_sum;
    features[9] = gln / (total_sum * total_sum);
    features[10] = szn / total_sum;
    features[11] = szn / (total_sum * total_sum);
    features[12] = (num_voxels_in_roi[0] > 0) ? (total_sum / (double)num_voxels_in_roi[0]) : 0.0;

    double gl_mean = accum[GLSZM_ACC_GL_WEIGHTED] / total_sum;
    double gl_second = accum[GLSZM_ACC_GL2_WEIGHTED] / total_sum;
    double gl_var = gl_second - (gl_mean * gl_mean);
    if (gl_var < 0.0 && gl_var > -1e-12) {
        gl_var = 0.0;
    }
    features[13] = gl_var;

    double sz_mean = accum[GLSZM_ACC_SZ_WEIGHTED] / total_sum;
    double sz_second = accum[GLSZM_ACC_SZ2_WEIGHTED] / total_sum;
    double sz_var = sz_second - (sz_mean * sz_mean);
    if (sz_var < 0.0 && sz_var > -1e-12) {
        sz_var = 0.0;
    }
    features[14] = sz_var;

    features[15] = log2(total_sum) - (accum[GLSZM_ACC_PLOGP] / total_sum);
}

// Task 7.5: Main API function for GLSZM CUDA features
FlashRadiomicsResult* flash_radiomics_glszm_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    double bin_width,
    int bin_count
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStream_t stream = ctx ? ctx->stream : 0;
    cudaEventRecord(start, stream);
    
    // Allocate result structure
    FlashRadiomicsResult* result = (FlashRadiomicsResult*)malloc(sizeof(FlashRadiomicsResult));
    if (!result) return NULL;
    
    result->features = NULL;
    result->count = 0;
    result->compute_time_ms = 0.0;
    result->error_code = FLASH_RADIOMICS_SUCCESS;
    result->error_msg[0] = '\0';

    // Predeclare all locals used by cleanup path to avoid C++ goto-scope issues.
    const uint8_t* d_mask = NULL;
    const int* d_discretized = NULL;
    int* d_labels = NULL;
    int* d_zone_sizes = NULL;
    int* d_zone_representatives = NULL;
    int* d_glszm_matrix = NULL;
    double* d_row_sums = NULL;
    double* d_col_sums = NULL;
    double* d_accum = NULL;
    int* d_changed = NULL;
    int* d_max_zone_size = NULL;
    int* d_roi_voxel_count = NULL;
    double* d_gln_numerator = NULL;
    double* d_szn_numerator = NULL;
    double* d_features = NULL;
    cudaError_t err = cudaSuccess;
    int block_size = CUDA_BLOCK_SIZE;
    int grid_size = 0;
    int n_voxels = 0;
    int max_zone_size = 0;
    int ccl_passes = 0;
    int h_changed = 0;
    int reduction_grid_size = 0;
    int deterministic_mode = 0;
    int deterministic_reduction = 0;
    int tie_break_lowest_label = 0;
    int allow_early_exit = 1;
    int ccl_sync_interval = GLSZM_CCL_SYNC_INTERVAL;
    int ccl_flatten_interval = GLSZM_CCL_FLATTEN_INTERVAL;
    size_t glszm_size = 0;
    const int num_features = 16;
    double h_features[16] = {0.0};
    float milliseconds = 0.0f;
    
    // Validate inputs
    if (!img || !mask || !ctx) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                           "Image, mask, or context is NULL");
        return result;
    }
    
    if (img->ndim != mask->ndim) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH,
                           "Image and mask dimensionality mismatch");
        return result;
    }
    
    if (!flash_radiomics_dims_match(img->dims, mask->dims)) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH,
                           "Image and mask dimensions do not match");
        return result;
    }

    // Keep GLSZM work on the per-call stream so concurrent feature classes
    // do not fall back to device-wide synchronizations.
    
    size_t total_voxels = flash_radiomics_get_num_elements(img->ndim, img->dims);
    int width = img->dims[0];
    int height = img->dims[1];
    int depth = (img->ndim == 3) ? img->dims[2] : 1;
    n_voxels = (int)total_voxels;
    
    int workspace_status =
        cuda_segment_workspace_prepare(
            ctx, img, mask, bin_width, bin_count, ctx->has_bin_minimum, ctx->bin_minimum
        );
    if (workspace_status != FLASH_CUDA_SUCCESS) {
        flash_radiomics_set_error(
            &result->error_code,
            result->error_msg,
            FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
            ctx->error_msg
        );
        return result;
    }

    int num_bins = cuda_segment_workspace_num_bins(ctx);
    d_mask = cuda_segment_workspace_mask(ctx);
    d_discretized = cuda_segment_workspace_discretized(ctx);
    if (!d_mask || !d_discretized || num_bins <= 0) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
                           "CUDA segment workspace is not ready");
        return result;
    }

    deterministic_mode = cuda_get_deterministic_mode(ctx);
    cuda_get_glszm_determinism(
        ctx,
        &tie_break_lowest_label,
        &deterministic_reduction,
        &allow_early_exit,
        &ccl_sync_interval,
        &ccl_flatten_interval
    );
    if (deterministic_mode) {
        tie_break_lowest_label = 1;
        deterministic_reduction = 1;
        allow_early_exit = 0;
        ccl_sync_interval = 1;
        ccl_flatten_interval = 1;
    }
    if (ccl_sync_interval < 1) {
        ccl_sync_interval = 1;
    }
    if (ccl_flatten_interval < 1) {
        ccl_flatten_interval = 1;
    }
    
    err = cudaMalloc(&d_labels, total_voxels * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_zone_sizes, total_voxels * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_zone_representatives, total_voxels * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_changed, sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    // Initialize labels
    grid_size = (n_voxels + block_size - 1) / block_size;
    if (grid_size < 1) {
        grid_size = 1;
    }
    if (grid_size > 1024) {
        grid_size = 1024;
    }

    ccl_init_kernel<<<grid_size, block_size, 0, stream>>>(
        d_mask, d_labels, n_voxels
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    
    if (img->ndim == 2) {
        dim3 block_2d(16, 16);
        dim3 grid_2d((width + 15) / 16, (height + 15) / 16);
        ccl_passes = width + height;
        if (ccl_passes < 1) {
            ccl_passes = 1;
        }

        for (int base_pass = 0; base_pass < ccl_passes; base_pass += ccl_sync_interval) {
            int chunk_passes = ccl_passes - base_pass;
            if (chunk_passes > ccl_sync_interval) {
                chunk_passes = ccl_sync_interval;
            }

            h_changed = 0;
            err = cudaMemcpyAsync(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) goto cleanup;

            for (int pass = 0; pass < chunk_passes; pass++) {
                ccl_merge_kernel_2d<<<grid_2d, block_2d, 0, stream>>>(
                    d_discretized, d_mask, d_labels, d_changed,
                    tie_break_lowest_label, width, height
                );
                err = cudaGetLastError();
                if (err != cudaSuccess) goto cleanup;

                if (((pass + 1) % ccl_flatten_interval) == 0) {
                    ccl_flatten_kernel<<<grid_size, block_size, 0, stream>>>(d_labels, n_voxels);
                    err = cudaGetLastError();
                    if (err != cudaSuccess) goto cleanup;
                }
            }

            err = cudaMemcpyAsync(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) goto cleanup;
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) goto cleanup;
            if (allow_early_exit && h_changed == 0) {
                break;
            }
        }
    } else {
        dim3 block_3d(8, 8, 4);
        dim3 grid_3d((width + 7) / 8, (height + 7) / 8, (depth + 3) / 4);
        ccl_passes = width + height + depth;
        if (ccl_passes < 1) {
            ccl_passes = 1;
        }

        for (int base_pass = 0; base_pass < ccl_passes; base_pass += ccl_sync_interval) {
            int chunk_passes = ccl_passes - base_pass;
            if (chunk_passes > ccl_sync_interval) {
                chunk_passes = ccl_sync_interval;
            }

            h_changed = 0;
            err = cudaMemcpyAsync(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) goto cleanup;

            for (int pass = 0; pass < chunk_passes; pass++) {
                ccl_merge_kernel_3d<<<grid_3d, block_3d, 0, stream>>>(
                    d_discretized, d_mask, d_labels, d_changed,
                    tie_break_lowest_label, width, height, depth
                );
                err = cudaGetLastError();
                if (err != cudaSuccess) goto cleanup;

                if (((pass + 1) % ccl_flatten_interval) == 0) {
                    ccl_flatten_kernel<<<grid_size, block_size, 0, stream>>>(d_labels, n_voxels);
                    err = cudaGetLastError();
                    if (err != cudaSuccess) goto cleanup;
                }
            }

            err = cudaMemcpyAsync(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess) goto cleanup;
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) goto cleanup;
            if (allow_early_exit && h_changed == 0) {
                break;
            }
        }
    }

    // Flatten label tree
    ccl_flatten_kernel<<<grid_size, block_size, 0, stream>>>(d_labels, n_voxels);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    // Initialize zone sizes and representatives
    err = cudaMemsetAsync(d_zone_sizes, 0, total_voxels * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup;

    // Fill representatives with a large sentinel so first atomicMin wins.
    err = cudaMemsetAsync(d_zone_representatives, 0x7f, total_voxels * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup;

    // Compute zone sizes
    compute_zone_sizes_kernel<<<grid_size, block_size, 0, stream>>>(
        d_labels, d_mask, d_zone_sizes, n_voxels
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    // Mark zone representatives
    mark_zone_representatives_kernel<<<grid_size, block_size, 0, stream>>>(
        d_labels, d_mask, d_zone_representatives, n_voxels
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_max_zone_size, sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_max_zone_size, 0, sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup;

    reduce_max_int_kernel<<<grid_size, block_size, 0, stream>>>(d_zone_sizes, n_voxels, d_max_zone_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(&max_zone_size, d_max_zone_size, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) goto cleanup;
    
    if (max_zone_size == 0) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
                           "No zones found in ROI");
        goto cleanup;
    }
    
    // Allocate GLSZM matrix
    glszm_size = (size_t)num_bins * (size_t)max_zone_size;
    err = cudaMalloc(&d_glszm_matrix, glszm_size * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemsetAsync(d_glszm_matrix, 0, glszm_size * sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup;

    // Build GLSZM matrix
    build_glszm_from_representatives_kernel<<<grid_size, block_size, 0, stream>>>(
        d_discretized, d_labels, d_zone_sizes, d_zone_representatives,
        d_mask, d_glszm_matrix, num_bins, max_zone_size, n_voxels
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_roi_voxel_count, sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_roi_voxel_count, 0, sizeof(int), stream);
    if (err != cudaSuccess) goto cleanup;
    count_mask_voxels_kernel<<<grid_size, block_size, 0, stream>>>(
        d_mask, n_voxels, d_roi_voxel_count
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate global reduction buffers.
    err = cudaMalloc(&d_accum, GLSZM_ACC_COUNT * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_gln_numerator, sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_szn_numerator, sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_features, (size_t)num_features * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemsetAsync(d_accum, 0, GLSZM_ACC_COUNT * sizeof(double), stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_gln_numerator, 0, sizeof(double), stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_szn_numerator, 0, sizeof(double), stream);
    if (err != cudaSuccess) goto cleanup;
    if (deterministic_reduction) {
        compute_glszm_accumulators_deterministic_kernel<<<1, 1, 0, stream>>>(
            d_glszm_matrix,
            num_bins,
            max_zone_size,
            d_accum,
            d_gln_numerator,
            d_szn_numerator
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    } else {
        err = cudaMalloc(&d_row_sums, (size_t)num_bins * sizeof(double));
        if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(&d_col_sums, (size_t)max_zone_size * sizeof(double));
        if (err != cudaSuccess) goto cleanup;

        err = cudaMemsetAsync(d_row_sums, 0, (size_t)num_bins * sizeof(double), stream);
        if (err != cudaSuccess) goto cleanup;
        err = cudaMemsetAsync(d_col_sums, 0, (size_t)max_zone_size * sizeof(double), stream);
        if (err != cudaSuccess) goto cleanup;

        reduction_grid_size = (int)((glszm_size + (size_t)block_size - 1) / (size_t)block_size);
        if (reduction_grid_size < 1) {
            reduction_grid_size = 1;
        }

        compute_glszm_accumulators_kernel<<<reduction_grid_size, block_size, 0, stream>>>(
            d_glszm_matrix,
            num_bins,
            max_zone_size,
            glszm_size,
            d_row_sums,
            d_col_sums,
            d_accum
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;

        reduce_square_sum_kernel<<<grid_size, block_size, 0, stream>>>(
            d_row_sums, num_bins, d_gln_numerator
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;

        reduce_square_sum_kernel<<<grid_size, block_size, 0, stream>>>(
            d_col_sums, max_zone_size, d_szn_numerator
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    finalize_glszm_features_kernel<<<1, 1, 0, stream>>>(
        d_accum,
        d_gln_numerator,
        d_szn_numerator,
        d_roi_voxel_count,
        d_features
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpyAsync(
        h_features,
        d_features,
        (size_t)num_features * sizeof(double),
        cudaMemcpyDeviceToHost,
        stream
    );
    if (err != cudaSuccess) goto cleanup;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate result features
    result->features = (FlashRadiomicsFeature*)malloc(num_features * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate memory for features");
        goto cleanup;
    }
    
    result->count = num_features;

    static const char* feature_names[16] = {
        "glszm_SmallAreaEmphasis",
        "glszm_LargeAreaEmphasis",
        "glszm_LowGrayLevelZoneEmphasis",
        "glszm_HighGrayLevelZoneEmphasis",
        "glszm_SmallAreaLowGrayLevelEmphasis",
        "glszm_SmallAreaHighGrayLevelEmphasis",
        "glszm_LargeAreaLowGrayLevelEmphasis",
        "glszm_LargeAreaHighGrayLevelEmphasis",
        "glszm_GrayLevelNonUniformity",
        "glszm_GrayLevelNonUniformityNormalized",
        "glszm_SizeZoneNonUniformity",
        "glszm_SizeZoneNonUniformityNormalized",
        "glszm_ZonePercentage",
        "glszm_GrayLevelVariance",
        "glszm_SizeZoneVariance",
        "glszm_ZoneEntropy",
    };
    for (int i = 0; i < num_features; i++) {
        snprintf(result->features[i].name, sizeof(result->features[i].name), "%s", feature_names[i]);
        result->features[i].value = h_features[i];
    }
    
cleanup:
    if (err != cudaSuccess && result->error_code == FLASH_RADIOMICS_SUCCESS) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
                           cudaGetErrorString(err));
    }

    if (stream) {
        cudaError_t cleanup_sync_err = cudaStreamSynchronize(stream);
        if (err == cudaSuccess && cleanup_sync_err != cudaSuccess) {
            err = cleanup_sync_err;
        }
    }

    // Free device memory
    if (d_labels) cudaFree(d_labels);
    if (d_zone_sizes) cudaFree(d_zone_sizes);
    if (d_zone_representatives) cudaFree(d_zone_representatives);
    if (d_glszm_matrix) cudaFree(d_glszm_matrix);
    if (d_row_sums) cudaFree(d_row_sums);
    if (d_col_sums) cudaFree(d_col_sums);
    if (d_accum) cudaFree(d_accum);
    if (d_changed) cudaFree(d_changed);
    if (d_max_zone_size) cudaFree(d_max_zone_size);
    if (d_roi_voxel_count) cudaFree(d_roi_voxel_count);
    if (d_gln_numerator) cudaFree(d_gln_numerator);
    if (d_szn_numerator) cudaFree(d_szn_numerator);
    if (d_features) cudaFree(d_features);
    
    // Record computation time
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    result->compute_time_ms = milliseconds;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

#else // !FLASH_RADIOMICS_CUDA_ENABLED

// Stub implementation when CUDA is not available
FlashRadiomicsResult* flash_radiomics_glszm_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    double bin_width,
    int bin_count
) {
    FlashRadiomicsResult* result = (FlashRadiomicsResult*)malloc(sizeof(FlashRadiomicsResult));
    if (!result) return NULL;
    
    result->features = NULL;
    result->count = 0;
    result->compute_time_ms = 0.0;
    flash_radiomics_set_error(&result->error_code, result->error_msg,
                       FLASH_RADIOMICS_ERROR_NOT_IMPLEMENTED,
                       "CUDA support not compiled");
    return result;
}

#endif // FLASH_RADIOMICS_CUDA_ENABLED
