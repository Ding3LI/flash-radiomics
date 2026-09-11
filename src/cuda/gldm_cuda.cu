#include "gldm_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define GLDM_EPSILON 2.2e-16

namespace {

static const char* GLDM_FEATURE_NAMES[17] = {
    "gldm_SmallDependenceEmphasis",
    "gldm_LargeDependenceEmphasis",
    "gldm_GrayLevelNonUniformity",
    "gldm_DependenceNonUniformity",
    "gldm_DependenceNonUniformityNormalized",
    "gldm_GrayLevelVariance",
    "gldm_DependenceVariance",
    "gldm_DependenceEntropy",
    "gldm_LowGrayLevelEmphasis",
    "gldm_HighGrayLevelEmphasis",
    "gldm_SmallDependenceLowGrayLevelEmphasis",
    "gldm_SmallDependenceHighGrayLevelEmphasis",
    "gldm_LargeDependenceLowGrayLevelEmphasis",
    "gldm_LargeDependenceHighGrayLevelEmphasis",
    "gldm_GrayLevelNonUniformityNormalized",
    "gldm_DependenceCountPercentage",
    "gldm_DependenceCountEnergy"
};

// 8 directions for 2D
__constant__ int GLDM_DIRECTIONS_2D[8][2] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};

// 26 directions for 3D
__constant__ int GLDM_DIRECTIONS_3D[26][3] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
    {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1},
    {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1},
    {1, 1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, -1, 1},
    {1, -1, 1}, {-1, 1, -1}, {1, -1, -1}, {-1, 1, 1}
};

__global__ void build_gldm_matrix_kernel(
    const int* discretized,
    const uint8_t* mask,
    int* gldm_matrix,
    int width,
    int height,
    int depth,
    int ndim,
    int force_2d,
    const int* distances,
    int num_distances,
    int num_dirs,
    int max_dep,
    float gldm_a
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int slice_size = width * height;
    const int total_voxels = slice_size * depth;
    const int dep_len = max_dep + 1;

    for (int i = idx; i < total_voxels; i += stride) {
        if (mask[i] == 0) {
            continue;
        }
        const int center_gray = discretized[i];
        if (center_gray < 0) {
            continue;
        }

        const int z = i / slice_size;
        const int rem = i % slice_size;
        const int y = rem / width;
        const int x = rem % width;

        int dep = 0;
        for (int d = 0; d < num_distances; d++) {
            const int dist = distances[d];
            if (dist <= 0) {
                continue;
            }

            if (ndim == 2 || force_2d != 0) {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + GLDM_DIRECTIONS_2D[dir][0] * dist;
                    const int ny = y + GLDM_DIRECTIONS_2D[dir][1] * dist;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    const int nidx = z * slice_size + ny * width + nx;
                    if (mask[nidx] == 0) {
                        continue;
                    }
                    const int ngray = discretized[nidx];
                    if (ngray < 0) {
                        continue;
                    }
                    const int diff = abs(center_gray - ngray);
                    if (static_cast<float>(diff) <= gldm_a) {
                        dep++;
                    }
                }
            } else {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + GLDM_DIRECTIONS_3D[dir][0] * dist;
                    const int ny = y + GLDM_DIRECTIONS_3D[dir][1] * dist;
                    const int nz = z + GLDM_DIRECTIONS_3D[dir][2] * dist;
                    if (nx < 0 || nx >= width ||
                        ny < 0 || ny >= height ||
                        nz < 0 || nz >= depth) {
                        continue;
                    }
                    const int nidx = nz * slice_size + ny * width + nx;
                    if (mask[nidx] == 0) {
                        continue;
                    }
                    const int ngray = discretized[nidx];
                    if (ngray < 0) {
                        continue;
                    }
                    const int diff = abs(center_gray - ngray);
                    if (static_cast<float>(diff) <= gldm_a) {
                        dep++;
                    }
                }
            }
        }

        if (dep < 0) dep = 0;
        if (dep > max_dep) dep = max_dep;
        const int matrix_idx = center_gray * dep_len + dep;
        atomicAdd(&gldm_matrix[matrix_idx], 1);
    }
}

__global__ void finalize_gldm_features_kernel(
    const int* matrix_counts,
    int num_bins,
    int dep_len,
    double out_features[17]
) {
    if (blockIdx.x != 0 || !matrix_counts || !out_features || num_bins <= 0 || dep_len <= 0) {
        return;
    }

    __shared__ double shared_acc[17][256];
    __shared__ double shared_nz;
    __shared__ double shared_mean_g;
    __shared__ double shared_mean_d;

    const int tid = (int)threadIdx.x;
    const int matrix_size = num_bins * dep_len;
    double local_nz = 0.0;
    double local_mean_g_num = 0.0;
    double local_mean_d_num = 0.0;
    for (int linear_idx = tid; linear_idx < matrix_size; linear_idx += (int)blockDim.x) {
        const int i = linear_idx / dep_len;
        const int j = linear_idx - i * dep_len;
        const double v = static_cast<double>(matrix_counts[linear_idx]);
        local_nz += v;
        local_mean_g_num += static_cast<double>(i + 1) * v;
        local_mean_d_num += static_cast<double>(j + 1) * v;
    }

    shared_acc[0][tid] = local_nz;
    shared_acc[1][tid] = local_mean_g_num;
    shared_acc[2][tid] = local_mean_d_num;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_acc[0][tid] += shared_acc[0][tid + stride];
            shared_acc[1][tid] += shared_acc[1][tid + stride];
            shared_acc[2][tid] += shared_acc[2][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_nz = shared_acc[0][0];
        if (shared_nz > 0.0) {
            shared_mean_g = shared_acc[1][0] / shared_nz;
            shared_mean_d = shared_acc[2][0] / shared_nz;
        }
    }
    __syncthreads();

    if (shared_nz <= 0.0) {
        if (tid == 0) {
            for (int k = 0; k < 17; k++) {
                out_features[k] = NAN;
            }
        }
        return;
    }

    double local_acc[17];
    for (int k = 0; k < 17; k++) {
        local_acc[k] = 0.0;
    }

    const double Nz = shared_nz;
    const double mean_g = shared_mean_g;
    const double mean_d = shared_mean_d;
    for (int linear_idx = tid; linear_idx < matrix_size; linear_idx += (int)blockDim.x) {
        const int i = linear_idx / dep_len;
        const int j = linear_idx - i * dep_len;
        const double v = static_cast<double>(matrix_counts[linear_idx]);
        if (v <= 0.0) {
            continue;
        }
        const double i_val = static_cast<double>(i + 1);
        const double j_val = static_cast<double>(j + 1);
        const double i_sq = i_val * i_val;
        const double j_sq = j_val * j_val;
        const double p = v / Nz;

        local_acc[0] += p / j_sq;
        local_acc[1] += p * j_sq;
        local_acc[5] += p * (i_val - mean_g) * (i_val - mean_g);
        local_acc[6] += p * (j_val - mean_d) * (j_val - mean_d);
        local_acc[7] -= p * log2(p + GLDM_EPSILON);
        local_acc[8] += p / i_sq;
        local_acc[9] += p * i_sq;
        local_acc[10] += p / (i_sq * j_sq);
        local_acc[11] += p * i_sq / j_sq;
        local_acc[12] += p * j_sq / i_sq;
        local_acc[13] += p * i_sq * j_sq;
        local_acc[16] += p * p;
    }

    double local_gln = 0.0;
    for (int i = tid; i < num_bins; i += (int)blockDim.x) {
        double row_sum = 0.0;
        for (int j = 0; j < dep_len; j++) {
            row_sum += static_cast<double>(
                matrix_counts[static_cast<size_t>(i) * static_cast<size_t>(dep_len) + static_cast<size_t>(j)]
            );
        }
        local_gln += row_sum * row_sum;
    }
    local_acc[2] = local_gln;

    double local_dn = 0.0;
    for (int j = tid; j < dep_len; j += (int)blockDim.x) {
        double col_sum = 0.0;
        for (int i = 0; i < num_bins; i++) {
            col_sum += static_cast<double>(
                matrix_counts[static_cast<size_t>(i) * static_cast<size_t>(dep_len) + static_cast<size_t>(j)]
            );
        }
        local_dn += col_sum * col_sum;
    }
    local_acc[3] = local_dn;

    for (int k = 0; k < 17; k++) {
        shared_acc[k][tid] = local_acc[k];
    }
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int k = 0; k < 17; k++) {
                shared_acc[k][tid] += shared_acc[k][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_features[0] = shared_acc[0][0];
        out_features[1] = shared_acc[1][0];
        out_features[2] = shared_acc[2][0] / Nz;
        out_features[3] = shared_acc[3][0] / Nz;
        out_features[4] = shared_acc[3][0] / (Nz * Nz);
        out_features[5] = shared_acc[5][0];
        out_features[6] = shared_acc[6][0];
        out_features[7] = shared_acc[7][0];
        out_features[8] = shared_acc[8][0];
        out_features[9] = shared_acc[9][0];
        out_features[10] = shared_acc[10][0];
        out_features[11] = shared_acc[11][0];
        out_features[12] = shared_acc[12][0];
        out_features[13] = shared_acc[13][0];
        out_features[14] = shared_acc[2][0] / (Nz * Nz);
        out_features[15] = 1.0;
        out_features[16] = shared_acc[16][0];
    }
}

}  // namespace

FlashRadiomicsResult* flash_radiomics_gldm_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    double gldm_a,
    int force_2d
) {
    FlashRadiomicsResult* result =
        static_cast<FlashRadiomicsResult*>(std::calloc(1, sizeof(FlashRadiomicsResult)));
    if (!result) {
        return NULL;
    }

    if (!img || !mask || !ctx || !distances || num_distances <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        std::snprintf(result->error_msg, sizeof(result->error_msg), "Invalid input parameters");
        return result;
    }
    if (img->ndim != mask->ndim ||
        img->dims[0] != mask->dims[0] ||
        img->dims[1] != mask->dims[1] ||
        img->dims[2] != mask->dims[2]) {
        result->error_code = FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH;
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "Image and mask dimensions do not match");
        return result;
    }

    int workspace_status =
        cuda_segment_workspace_prepare(
            ctx, img, mask, bin_width, bin_count, ctx->has_bin_minimum, ctx->bin_minimum
        );
    if (workspace_status != FLASH_CUDA_SUCCESS) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "Failed to prepare CUDA segment workspace: %.*s",
                      200,
                      ctx->error_msg ? ctx->error_msg : "");
        return result;
    }

    const uint8_t* d_mask = cuda_segment_workspace_mask(ctx);
    const int* d_discretized = cuda_segment_workspace_discretized(ctx);
    const int num_bins = cuda_segment_workspace_num_bins(ctx);
    if (!d_mask || !d_discretized || num_bins <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "CUDA segment workspace is not ready");
        return result;
    }

    const int width = img->dims[0];
    const int height = img->dims[1];
    const int depth = img->dims[2];
    const int n_voxels = width * height * depth;
    const int num_dirs = (img->ndim == 2 || force_2d != 0) ? 8 : 26;
    const int max_dep = num_dirs * num_distances;
    const int dep_len = max_dep + 1;
    const size_t matrix_size = static_cast<size_t>(num_bins) * static_cast<size_t>(dep_len);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, ctx->stream);

    int* d_matrix = NULL;
    int* d_distances = NULL;
    double* d_features = NULL;

    cudaError_t err = cudaSuccess;
    double h_features[17] = {0.0};
    float elapsed_ms = 0.0f;

    int block_size = 256;
    int grid_size = (n_voxels + block_size - 1) / block_size;

    err = cudaMalloc(&d_matrix, matrix_size * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_distances, static_cast<size_t>(num_distances) * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_features, static_cast<size_t>(17) * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpyAsync(
        d_distances,
        distances,
        static_cast<size_t>(num_distances) * sizeof(int),
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemsetAsync(d_matrix, 0, matrix_size * sizeof(int), ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    build_gldm_matrix_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_discretized,
        d_mask,
        d_matrix,
        width,
        height,
        depth,
        img->ndim,
        force_2d,
        d_distances,
        num_distances,
        num_dirs,
        max_dep,
        static_cast<float>(gldm_a)
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    finalize_gldm_features_kernel<<<1, 256, 0, ctx->stream>>>(
        d_matrix,
        num_bins,
        dep_len,
        d_features
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpyAsync(
        h_features,
        d_features,
        static_cast<size_t>(17) * sizeof(double),
        cudaMemcpyDeviceToHost,
        ctx->stream
    );
    if (err != cudaSuccess) goto cleanup;

    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    {
        result->count = 17;
        result->features = static_cast<FlashRadiomicsFeature*>(
            std::calloc(static_cast<size_t>(result->count), sizeof(FlashRadiomicsFeature))
        );
        if (!result->features) {
            result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            std::snprintf(result->error_msg, sizeof(result->error_msg),
                          "Failed to allocate GLDM feature array");
            goto cleanup;
        }

        for (int i = 0; i < result->count; i++) {
            std::snprintf(result->features[i].name, sizeof(result->features[i].name),
                          "%s", GLDM_FEATURE_NAMES[i]);
            result->features[i].value = h_features[i];
        }
        result->error_code = FLASH_RADIOMICS_SUCCESS;
    }

    cudaEventRecord(stop, ctx->stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    result->compute_time_ms = elapsed_ms;

cleanup:
    if (err != cudaSuccess && result->error_code == FLASH_RADIOMICS_SUCCESS) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "CUDA runtime error: %s", cudaGetErrorString(err));
    }

    if (d_matrix) cudaFree(d_matrix);
    if (d_distances) cudaFree(d_distances);
    if (d_features) cudaFree(d_features);

    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    return result;
}
