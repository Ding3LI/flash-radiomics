#include "firstorder_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define EXTRACT_BLOCK_SIZE 256

namespace {

enum {
    FIRSTORDER_FEATURE_COUNT = 18
};

static const char* kFirstorderFeatureNames[FIRSTORDER_FEATURE_COUNT] = {
    "firstorder_Mean",
    "firstorder_Variance",
    "firstorder_StandardDeviation",
    "firstorder_Skewness",
    "firstorder_Kurtosis",
    "firstorder_Minimum",
    "firstorder_Maximum",
    "firstorder_Range",
    "firstorder_Median",
    "firstorder_10Percentile",
    "firstorder_90Percentile",
    "firstorder_InterquartileRange",
    "firstorder_Energy",
    "firstorder_TotalEnergy",
    "firstorder_Entropy",
    "firstorder_RootMeanSquared",
    "firstorder_Uniformity",
    "firstorder_MeanAbsoluteDeviation"
};

__global__ void extract_masked_values_kernel(
    const float* image,
    const uint8_t* mask,
    float* output,
    int* output_count,
    int n_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n_voxels; i += stride) {
        if (mask[i] > 0) {
            int out_idx = atomicAdd(output_count, 1);
            output[out_idx] = image[i];
        }
    }
}

__device__ inline void accumulate_entropy_uniformity(
    int count,
    int total_count,
    double* entropy,
    double* uniformity
) {
    if (count <= 0 || total_count <= 0 || !entropy || !uniformity) {
        return;
    }
    const double p = static_cast<double>(count) / static_cast<double>(total_count);
    *entropy -= p * (log(p + DBL_EPSILON) / log(2.0));
    *uniformity += p * p;
}

__device__ inline double linear_percentile_from_sorted_float(
    const float* values,
    int count,
    double percentile
) {
    if (!values || count <= 0) {
        return NAN;
    }
    if (count == 1) {
        return static_cast<double>(values[0]);
    }

    double bounded_percentile = percentile;
    if (bounded_percentile < 0.0) {
        bounded_percentile = 0.0;
    } else if (bounded_percentile > 100.0) {
        bounded_percentile = 100.0;
    }

    const double rank = (static_cast<double>(count) - 1.0) * (bounded_percentile / 100.0);
    const int lower_index = static_cast<int>(floor(rank));
    const int upper_index = static_cast<int>(ceil(rank));
    if (lower_index == upper_index) {
        return static_cast<double>(values[lower_index]);
    }

    const double interpolation = rank - static_cast<double>(lower_index);
    const double lower = static_cast<double>(values[lower_index]);
    const double upper = static_cast<double>(values[upper_index]);
    return lower + (upper - lower) * interpolation;
}

__device__ inline int firstorder_gray_bin_from_value(
    double x,
    double min_val,
    double max_val,
    double bin_width,
    int bin_count,
    int has_bin_minimum,
    double bin_minimum
) {
    if (bin_count > 0) {
        if (max_val <= min_val) {
            return 1;
        }
        int gray = static_cast<int>(floor(((x - min_val) / (max_val - min_val)) * bin_count)) + 1;
        if (x >= max_val) gray = bin_count;
        if (gray < 1) gray = 1;
        if (gray > bin_count) gray = bin_count;
        return gray;
    }
    if (bin_width > 0.0) {
        if (has_bin_minimum != 0) {
            int gray = static_cast<int>(floor((x - bin_minimum) / bin_width)) + 1;
            if (gray < 1) gray = 1;
            return gray;
        }
        const int min_bin = static_cast<int>(floor(min_val / bin_width));
        int gray = static_cast<int>(floor(x / bin_width)) - min_bin + 1;
        if (gray < 1) gray = 1;
        return gray;
    }
    return 0;
}

__global__ void finalize_firstorder_features_kernel(
    const float* sorted_values,
    int n_values,
    double bin_width,
    int bin_count,
    int has_bin_minimum,
    double bin_minimum,
    double voxel_array_shift,
    double voxel_volume,
    double* out_features
) {
    if (blockIdx.x != 0 || !sorted_values || !out_features || n_values <= 0) {
        return;
    }

    __shared__ double shared_sum[256];
    __shared__ double shared_sum_sq[256];
    __shared__ double shared_sum_shifted_sq[256];
    __shared__ double shared_m3[256];
    __shared__ double shared_m4[256];
    __shared__ double shared_abs_dev[256];
    __shared__ double shared_entropy[256];
    __shared__ double shared_uniformity[256];
    __shared__ double shared_mean;
    __shared__ double shared_variance;
    __shared__ double shared_std_dev;
    __shared__ double shared_min_val;
    __shared__ double shared_max_val;

    const int tid = (int)threadIdx.x;
    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    double local_sum_shifted_sq = 0.0;
    for (int i = tid; i < n_values; i += (int)blockDim.x) {
        const double x = static_cast<double>(sorted_values[i]);
        const double shifted = x + voxel_array_shift;
        local_sum += x;
        local_sum_sq += x * x;
        local_sum_shifted_sq += shifted * shifted;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    shared_sum_shifted_sq[tid] = local_sum_shifted_sq;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
            shared_sum_shifted_sq[tid] += shared_sum_shifted_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_min_val = static_cast<double>(sorted_values[0]);
        shared_max_val = static_cast<double>(sorted_values[n_values - 1]);
        const double inv_n = 1.0 / static_cast<double>(n_values);
        shared_mean = shared_sum[0] * inv_n;
        double variance = (shared_sum_sq[0] * inv_n) - (shared_mean * shared_mean);
        if (variance < 0.0 && variance > -1e-12) {
            variance = 0.0;
        }
        shared_variance = variance;
        shared_std_dev = sqrt(fmax(0.0, variance));
    }
    __syncthreads();

    double local_m3 = 0.0;
    double local_m4 = 0.0;
    double local_abs_dev = 0.0;
    const double mean = shared_mean;
    for (int i = tid; i < n_values; i += (int)blockDim.x) {
        const double d = static_cast<double>(sorted_values[i]) - mean;
        const double d2 = d * d;
        local_m3 += d2 * d;
        local_m4 += d2 * d2;
        local_abs_dev += fabs(d);
    }

    shared_m3[tid] = local_m3;
    shared_m4[tid] = local_m4;
    shared_abs_dev[tid] = local_abs_dev;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_m3[tid] += shared_m3[tid + stride];
            shared_m4[tid] += shared_m4[tid + stride];
            shared_abs_dev[tid] += shared_abs_dev[tid + stride];
        }
        __syncthreads();
    }

    double local_entropy = 0.0;
    double local_uniformity = 0.0;
    const double min_val = shared_min_val;
    const double max_val = shared_max_val;
    if (!(bin_count > 0 && max_val <= min_val)) {
        for (int i = tid; i < n_values; i += (int)blockDim.x) {
            const double x = static_cast<double>(sorted_values[i]);
            int is_run_start = (i == 0) ? 1 : 0;
            if (i > 0) {
                const double prev = static_cast<double>(sorted_values[i - 1]);
                if (bin_count > 0 || bin_width > 0.0) {
                    is_run_start =
                        firstorder_gray_bin_from_value(
                            x, min_val, max_val, bin_width, bin_count,
                            has_bin_minimum, bin_minimum
                        ) !=
                        firstorder_gray_bin_from_value(
                            prev, min_val, max_val, bin_width, bin_count,
                            has_bin_minimum, bin_minimum
                        );
                } else {
                    is_run_start = (x != prev);
                }
            }
            if (!is_run_start) {
                continue;
            }

            int run_count = 1;
            if (bin_count > 0 || bin_width > 0.0) {
                const int gray = firstorder_gray_bin_from_value(
                    x, min_val, max_val, bin_width, bin_count,
                    has_bin_minimum, bin_minimum
                );
                for (int j = i + 1; j < n_values; j++) {
                    const double next = static_cast<double>(sorted_values[j]);
                    const int next_gray = firstorder_gray_bin_from_value(
                        next, min_val, max_val, bin_width, bin_count,
                        has_bin_minimum, bin_minimum
                    );
                    if (next_gray != gray) {
                        break;
                    }
                    run_count++;
                }
            } else {
                for (int j = i + 1; j < n_values; j++) {
                    if (sorted_values[j] != sorted_values[i]) {
                        break;
                    }
                    run_count++;
                }
            }
            accumulate_entropy_uniformity(
                run_count,
                n_values,
                &local_entropy,
                &local_uniformity
            );
        }
    } else if (tid == 0) {
        local_uniformity = 1.0;
    }

    shared_entropy[tid] = local_entropy;
    shared_uniformity[tid] = local_uniformity;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_entropy[tid] += shared_entropy[tid + stride];
            shared_uniformity[tid] += shared_uniformity[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double inv_n = 1.0 / static_cast<double>(n_values);
        const double variance = shared_variance;
        double skewness = 0.0;
        double kurtosis = 0.0;
        if (variance > 1e-16) {
            const double m3 = shared_m3[0] * inv_n;
            const double m4 = shared_m4[0] * inv_n;
            skewness = m3 / pow(variance, 1.5);
            // IBSI defines kurtosis as excess kurtosis (Pearson kurtosis minus 3).
            kurtosis = m4 / (variance * variance) - 3.0;
        }

        const double median = linear_percentile_from_sorted_float(sorted_values, n_values, 50.0);
        const double p10 = linear_percentile_from_sorted_float(sorted_values, n_values, 10.0);
        const double p25 = linear_percentile_from_sorted_float(sorted_values, n_values, 25.0);
        const double p75 = linear_percentile_from_sorted_float(sorted_values, n_values, 75.0);
        const double p90 = linear_percentile_from_sorted_float(sorted_values, n_values, 90.0);
        const double iqr = p75 - p25;

        const double energy = shared_sum_shifted_sq[0];
        const double rms = sqrt(shared_sum_shifted_sq[0] * inv_n);
        const double total_energy = energy * voxel_volume;
        const double mad = shared_abs_dev[0] * inv_n;

        out_features[0] = shared_mean;
        out_features[1] = variance;
        out_features[2] = shared_std_dev;
        out_features[3] = skewness;
        out_features[4] = kurtosis;
        out_features[5] = min_val;
        out_features[6] = max_val;
        out_features[7] = max_val - min_val;
        out_features[8] = median;
        out_features[9] = p10;
        out_features[10] = p90;
        out_features[11] = iqr;
        out_features[12] = energy;
        out_features[13] = total_energy;
        out_features[14] = shared_entropy[0];
        out_features[15] = rms;
        out_features[16] = shared_uniformity[0];
        out_features[17] = mad;
    }
}

static void populate_firstorder_result(
    const double* features,
    FlashRadiomicsResult* result
) {
    if (!features || !result) {
        return;
    }
    result->features = static_cast<FlashRadiomicsFeature*>(
        std::calloc(FIRSTORDER_FEATURE_COUNT, sizeof(FlashRadiomicsFeature)));
    if (!result->features) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "Failed to allocate firstorder feature buffer");
        return;
    }
    result->count = FIRSTORDER_FEATURE_COUNT;
    for (int i = 0; i < FIRSTORDER_FEATURE_COUNT; ++i) {
        std::snprintf(
            result->features[i].name,
            sizeof(result->features[i].name),
            "%s",
            kFirstorderFeatureNames[i]
        );
        result->features[i].value = features[i];
    }
    result->error_code = FLASH_RADIOMICS_SUCCESS;
}

} // namespace

FlashRadiomicsResult* flash_radiomics_firstorder_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    double bin_width,
    int bin_count,
    double voxel_array_shift
) {
    FlashRadiomicsResult* result =
        static_cast<FlashRadiomicsResult*>(std::calloc(1, sizeof(FlashRadiomicsResult)));
    if (!result) return NULL;

    if (!img || !mask || !ctx) {
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

    const int n_voxels = img->dims[0] * img->dims[1] * img->dims[2];
    if (n_voxels <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        std::snprintf(result->error_msg, sizeof(result->error_msg), "Invalid voxel count");
        return result;
    }

    int workspace_status =
        cuda_segment_workspace_prepare(
            ctx,
            img,
            mask,
            bin_width,
            bin_count,
            ctx->has_bin_minimum,
            ctx->bin_minimum
        );
    if (workspace_status != FLASH_CUDA_SUCCESS) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        std::snprintf(
            result->error_msg,
            sizeof(result->error_msg),
            "Failed to prepare CUDA segment workspace: %.*s",
            200,
            ctx->error_msg ? ctx->error_msg : ""
        );
        return result;
    }

    const float* d_image = cuda_segment_workspace_image(ctx);
    const uint8_t* d_mask = cuda_segment_workspace_mask(ctx);
    if (!d_image || !d_mask) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "CUDA segment workspace is not ready");
        return result;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, ctx->stream);

    float* d_masked_values = NULL;
    int* d_masked_count = NULL;
    double* d_feature_values = NULL;

    cudaError_t err = cudaSuccess;
    int h_masked_count = 0;
    double h_feature_values[FIRSTORDER_FEATURE_COUNT] = {0.0};
    float milliseconds = 0.0f;

    const int block_size = EXTRACT_BLOCK_SIZE;
    int grid_size = (n_voxels + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024;

    err = cudaMalloc(&d_masked_values, static_cast<size_t>(n_voxels) * sizeof(float));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_masked_count, sizeof(int));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemsetAsync(d_masked_count, 0, sizeof(int), ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    extract_masked_values_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
        d_image, d_mask, d_masked_values, d_masked_count, n_voxels);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpyAsync(&h_masked_count, d_masked_count, sizeof(int),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    if (h_masked_count <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        std::snprintf(result->error_msg, sizeof(result->error_msg), "No valid voxels in mask");
        goto cleanup;
    }

    thrust::sort(
        thrust::device_pointer_cast(d_masked_values),
        thrust::device_pointer_cast(d_masked_values) + h_masked_count
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_feature_values, FIRSTORDER_FEATURE_COUNT * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_feature_values, 0, FIRSTORDER_FEATURE_COUNT * sizeof(double), ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    {
        double voxel_volume = static_cast<double>(img->spacing[0]) *
                              static_cast<double>(img->spacing[1]) *
                              ((img->ndim == 3) ? static_cast<double>(img->spacing[2]) : 1.0);
        if (!(voxel_volume > 0.0)) voxel_volume = 1.0;

        finalize_firstorder_features_kernel<<<1, 256, 0, ctx->stream>>>(
            d_masked_values,
            h_masked_count,
            bin_width,
            bin_count,
            ctx->has_bin_minimum,
            ctx->bin_minimum,
            voxel_array_shift,
            voxel_volume,
            d_feature_values
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    err = cudaMemcpyAsync(
        h_feature_values,
        d_feature_values,
        FIRSTORDER_FEATURE_COUNT * sizeof(double),
        cudaMemcpyDeviceToHost,
        ctx->stream
    );
    if (err != cudaSuccess) goto cleanup;
    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) goto cleanup;

    populate_firstorder_result(h_feature_values, result);
    if (result->error_code != FLASH_RADIOMICS_SUCCESS) goto cleanup;

    cudaEventRecord(stop, ctx->stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    result->compute_time_ms = milliseconds;

cleanup:
    if (err != cudaSuccess && result->error_code == FLASH_RADIOMICS_SUCCESS) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "CUDA runtime error: %s", cudaGetErrorString(err));
    }

    if (d_masked_values) cudaFree(d_masked_values);
    if (d_masked_count) cudaFree(d_masked_count);
    if (d_feature_values) cudaFree(d_feature_values);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
}
