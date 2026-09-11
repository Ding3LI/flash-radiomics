#include "ngtdm_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

static const char* NGTDM_FEATURE_NAMES[5] = {
    "ngtdm_Coarseness",
    "ngtdm_Contrast",
    "ngtdm_Busyness",
    "ngtdm_Complexity",
    "ngtdm_Strength"
};

// 8 directions for 2D
__constant__ int NGTDM_DIRECTIONS_2D[8][2] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};

// 26 directions for 3D
__constant__ int NGTDM_DIRECTIONS_3D[26][3] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
    {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1},
    {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1},
    {1, 1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, -1, 1},
    {1, -1, 1}, {-1, 1, -1}, {1, -1, -1}, {-1, 1, 1}
};

__global__ void build_ngtdm_vectors_kernel(
    const int* discretized,
    const uint8_t* mask,
    double* n_i,
    double* s_i,
    int width,
    int height,
    int depth,
    int ndim,
    int force_2d,
    const int* distances,
    int num_distances,
    int num_dirs
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int slice_size = width * height;
    const int total_voxels = slice_size * depth;

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

        double sum_neighbors = 0.0;
        int count_neighbors = 0;
        for (int d = 0; d < num_distances; d++) {
            const int dist = distances[d];
            if (dist <= 0) {
                continue;
            }

            if (ndim == 2 || force_2d != 0) {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + NGTDM_DIRECTIONS_2D[dir][0] * dist;
                    const int ny = y + NGTDM_DIRECTIONS_2D[dir][1] * dist;
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
                    sum_neighbors += static_cast<double>(ngray);
                    count_neighbors++;
                }
            } else {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + NGTDM_DIRECTIONS_3D[dir][0] * dist;
                    const int ny = y + NGTDM_DIRECTIONS_3D[dir][1] * dist;
                    const int nz = z + NGTDM_DIRECTIONS_3D[dir][2] * dist;
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
                    sum_neighbors += static_cast<double>(ngray);
                    count_neighbors++;
                }
            }
        }

        if (count_neighbors <= 0) {
            continue;
        }

        const double avg = sum_neighbors / static_cast<double>(count_neighbors);
        const double diff = fabs(static_cast<double>(center_gray) - avg);
        atomicAdd(&n_i[center_gray], 1.0);
        atomicAdd(&s_i[center_gray], diff);
    }
}

__global__ void finalize_ngtdm_features_kernel(
    const double* n_i,
    const double* s_i,
    int num_bins,
    double out_features[5]
) {
    if (blockIdx.x != 0 || !n_i || !s_i || !out_features || num_bins <= 0) {
        return;
    }

    __shared__ double shared_acc[8][256];
    __shared__ double shared_nvp;
    __shared__ double shared_ngp;
    __shared__ double shared_sum_p_s;
    __shared__ double shared_sum_s_i;

    const int tid = (int)threadIdx.x;
    double local_nvp = 0.0;
    double local_ngp = 0.0;
    for (int i = tid; i < num_bins; i += (int)blockDim.x) {
        const double count = n_i[i];
        local_nvp += count;
        if (count > 0.0) {
            local_ngp += 1.0;
        }
    }

    shared_acc[0][tid] = local_nvp;
    shared_acc[1][tid] = local_ngp;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_acc[0][tid] += shared_acc[0][tid + stride];
            shared_acc[1][tid] += shared_acc[1][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_nvp = shared_acc[0][0];
        shared_ngp = shared_acc[1][0];
        for (int i = 0; i < 5; i++) {
            out_features[i] = NAN;
        }
    }
    __syncthreads();

    if (shared_nvp <= 0.0) {
        return;
    }

    double local_sum_p_s = 0.0;
    double local_sum_s_i = 0.0;
    for (int i = tid; i < num_bins; i += (int)blockDim.x) {
        if (n_i[i] <= 0.0) {
            continue;
        }
        const double p = n_i[i] / shared_nvp;
        local_sum_p_s += p * s_i[i];
        local_sum_s_i += s_i[i];
    }

    shared_acc[0][tid] = local_sum_p_s;
    shared_acc[1][tid] = local_sum_s_i;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_acc[0][tid] += shared_acc[0][tid + stride];
            shared_acc[1][tid] += shared_acc[1][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_sum_p_s = shared_acc[0][0];
        shared_sum_s_i = shared_acc[1][0];
    }
    __syncthreads();

    double local_contrast = 0.0;
    double local_busy_denom = 0.0;
    double local_complexity = 0.0;
    double local_strength = 0.0;
    const size_t pair_count = (size_t)num_bins * (size_t)num_bins;
    for (size_t pair_idx = (size_t)tid; pair_idx < pair_count; pair_idx += (size_t)blockDim.x) {
        const int i = (int)(pair_idx / (size_t)num_bins);
        const int j = (int)(pair_idx - (size_t)i * (size_t)num_bins);
        if (j <= i || n_i[i] <= 0.0 || n_i[j] <= 0.0) {
            continue;
        }
        const double p_ii = n_i[i] / shared_nvp;
        const double p_jj = n_i[j] / shared_nvp;
        const double i_val = static_cast<double>(i + 1);
        const double j_val = static_cast<double>(j + 1);
        const double gray_diff = i_val - j_val;
        const double gray_diff_sq = gray_diff * gray_diff;
        local_contrast += 2.0 * p_ii * p_jj * gray_diff_sq;
        local_busy_denom += 2.0 * fabs(i_val * p_ii - j_val * p_jj);
        const double denom = p_ii + p_jj;
        if (denom > 0.0) {
            local_complexity += 2.0 * (fabs(gray_diff) / denom) *
                (p_ii * s_i[i] + p_jj * s_i[j]);
        }
        local_strength += 2.0 * (p_ii + p_jj) * gray_diff_sq;
    }

    shared_acc[0][tid] = local_contrast;
    shared_acc[1][tid] = local_busy_denom;
    shared_acc[2][tid] = local_complexity;
    shared_acc[3][tid] = local_strength;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_acc[0][tid] += shared_acc[0][tid + stride];
            shared_acc[1][tid] += shared_acc[1][tid + stride];
            shared_acc[2][tid] += shared_acc[2][tid + stride];
            shared_acc[3][tid] += shared_acc[3][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_features[0] = (shared_sum_p_s != 0.0) ? (1.0 / shared_sum_p_s) : 1e6;
        out_features[1] = 0.0;
        if (shared_ngp > 1.0) {
            out_features[1] =
                (shared_acc[0][0] / (shared_ngp * (shared_ngp - 1.0))) *
                (shared_sum_s_i / shared_nvp);
        }
        out_features[2] = (shared_acc[1][0] != 0.0) ? (shared_sum_p_s / shared_acc[1][0]) : 0.0;
        out_features[3] = shared_acc[2][0] / shared_nvp;
        out_features[4] = (shared_sum_s_i != 0.0) ? (shared_acc[3][0] / shared_sum_s_i) : 0.0;
    }
}

}  // namespace

FlashRadiomicsResult* flash_radiomics_ngtdm_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
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

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, ctx->stream);

    int* d_distances = NULL;
    double* d_n_i = NULL;
    double* d_s_i = NULL;
    double* d_features = NULL;

    cudaError_t err = cudaSuccess;
    double h_features[5] = {0.0};
    float elapsed_ms = 0.0f;

    int block_size = 256;
    int grid_size = (n_voxels + block_size - 1) / block_size;

    err = cudaMalloc(&d_distances, static_cast<size_t>(num_distances) * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_n_i, static_cast<size_t>(num_bins) * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_s_i, static_cast<size_t>(num_bins) * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_features, static_cast<size_t>(5) * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(
        d_distances,
        distances,
        static_cast<size_t>(num_distances) * sizeof(int),
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemsetAsync(d_n_i, 0, static_cast<size_t>(num_bins) * sizeof(double), ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_s_i, 0, static_cast<size_t>(num_bins) * sizeof(double), ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    build_ngtdm_vectors_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_discretized,
        d_mask,
        d_n_i,
        d_s_i,
        width,
        height,
        depth,
        img->ndim,
        force_2d,
        d_distances,
        num_distances,
        num_dirs
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    finalize_ngtdm_features_kernel<<<1, 256, 0, ctx->stream>>>(
        d_n_i,
        d_s_i,
        num_bins,
        d_features
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpyAsync(
        h_features,
        d_features,
        static_cast<size_t>(5) * sizeof(double),
        cudaMemcpyDeviceToHost,
        ctx->stream
    );
    if (err != cudaSuccess) goto cleanup;

    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    {
        result->count = 5;
        result->features = static_cast<FlashRadiomicsFeature*>(
            std::calloc(static_cast<size_t>(result->count), sizeof(FlashRadiomicsFeature))
        );
        if (!result->features) {
            result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            std::snprintf(result->error_msg, sizeof(result->error_msg),
                          "Failed to allocate NGTDM feature array");
            goto cleanup;
        }

        for (int i = 0; i < result->count; i++) {
            std::snprintf(result->features[i].name, sizeof(result->features[i].name),
                          "%s", NGTDM_FEATURE_NAMES[i]);
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

    if (d_distances) cudaFree(d_distances);
    if (d_n_i) cudaFree(d_n_i);
    if (d_s_i) cudaFree(d_s_i);
    if (d_features) cudaFree(d_features);

    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    return result;
}
