#include "glcm_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// Number of bins for discretization (default for radiomics)
#define DEFAULT_NUM_BINS 256
#define EPSILON 2.2e-16

static const char* GLCM_FEATURE_NAMES[23] = {
    "glcm_Autocorrelation",
    "glcm_JointAverage",
    "glcm_ClusterProminence",
    "glcm_ClusterShade",
    "glcm_ClusterTendency",
    "glcm_Contrast",
    "glcm_Correlation",
    "glcm_DifferenceAverage",
    "glcm_DifferenceEntropy",
    "glcm_DifferenceVariance",
    "glcm_JointEnergy",
    "glcm_JointEntropy",
    "glcm_Id",
    "glcm_Idm",
    "glcm_Idmn",
    "glcm_Idn",
    "glcm_InverseVariance",
    "glcm_MaximumProbability",
    "glcm_SumAverage",
    "glcm_SumEntropy",
    "glcm_SumSquares",
    "glcm_Imc1",
    "glcm_Imc2"
};

// Direction vectors for 2D (4 directions: 0°, 45°, 90°, 135°)
__constant__ int DIRECTIONS_2D[4][2] = {
    {1, 0},   // 0° (horizontal)
    {1, 1},   // 45° (diagonal)
    {0, 1},   // 90° (vertical)
    {-1, 1}   // 135° (anti-diagonal)
};

// Direction vectors for 3D (13 directions)
__constant__ int DIRECTIONS_3D[13][3] = {
    {1, 0, 0},   // 0
    {0, 1, 0},   // 1
    {0, 0, 1},   // 2
    {1, 1, 0},   // 3
    {1, -1, 0},  // 4
    {1, 0, 1},   // 5
    {1, 0, -1},  // 6
    {0, 1, 1},   // 7
    {0, 1, -1},  // 8
    {1, 1, 1},   // 9
    {1, 1, -1},  // 10
    {1, -1, 1},  // 11
    {1, -1, -1}  // 12
};

// ============================================================================
// Kernel: Discretize image values into bins
// ============================================================================
__global__ void discretize_image_kernel(
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
        if (mask[i] != 0) {
            float val = image[i];
            int bin_idx = 0;
            if (mode == 1) {
                if (bin_width > 0.0f) {
                    bin_idx = (int)floorf(val / bin_width) - min_bin;
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
            
            // Clamp to valid range
            if (bin_idx >= num_bins) bin_idx = num_bins - 1;
            if (bin_idx < 0) bin_idx = 0;
            
            discretized[i] = bin_idx;
        } else {
            discretized[i] = -1;  // Mark as outside ROI
        }
    }
}

// ============================================================================
// Kernel: Compute GLCM matrix for 2D images
// ============================================================================
// This kernel computes the Gray Level Co-occurrence Matrix (GLCM) for 2D images.
// Each thread processes one voxel and checks its neighbor in the specified direction.
// Uses atomic operations for thread-safe matrix updates with shared memory staging
// to reduce global memory contention.

__global__ void compute_glcm_2d_kernel(
    const int* discretized,
    const uint8_t* mask,
    int* glcm_matrix,
    int width,
    int height,
    int num_bins,
    int distance,
    int dir_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Get direction offset
    int dx = DIRECTIONS_2D[dir_idx][0] * distance;
    int dy = DIRECTIONS_2D[dir_idx][1] * distance;
    
    // Process voxels
    for (int i = idx; i < width * height; i += stride) {
        if (mask[i] == 0) continue;
        
        int x = i % width;
        int y = i / width;
        
        int nx = x + dx;
        int ny = y + dy;
        
        // Check bounds
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int nidx = ny * width + nx;
            
            if (mask[nidx] != 0) {
                int bin_i = discretized[i];
                int bin_j = discretized[nidx];
                
                // Symmetric GLCM: count both (i,j) and (j,i)
                atomicAdd(&glcm_matrix[bin_i * num_bins + bin_j], 1);
                atomicAdd(&glcm_matrix[bin_j * num_bins + bin_i], 1);
            }
        }
    }
}

// ============================================================================
// Kernel: Compute GLCM matrix for 3D images
// ============================================================================
// Similar to 2D kernel but handles 3D spatial relationships.
// Processes 3D volumes with 13 directional neighbors.

__global__ void compute_glcm_3d_kernel(
    const int* discretized,
    const uint8_t* mask,
    int* glcm_matrix,
    int width,
    int height,
    int depth,
    int num_bins,
    int distance,
    int dir_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Get direction offset
    int dx = DIRECTIONS_3D[dir_idx][0] * distance;
    int dy = DIRECTIONS_3D[dir_idx][1] * distance;
    int dz = DIRECTIONS_3D[dir_idx][2] * distance;
    
    int slice_size = width * height;
    int total_voxels = slice_size * depth;
    
    // Process voxels
    for (int i = idx; i < total_voxels; i += stride) {
        if (mask[i] == 0) continue;
        
        int z = i / slice_size;
        int rem = i % slice_size;
        int y = rem / width;
        int x = rem % width;
        
        int nx = x + dx;
        int ny = y + dy;
        int nz = z + dz;
        
        // Check bounds
        if (nx >= 0 && nx < width && 
            ny >= 0 && ny < height && 
            nz >= 0 && nz < depth) {
            int nidx = nz * slice_size + ny * width + nx;
            
            if (mask[nidx] != 0) {
                int bin_i = discretized[i];
                int bin_j = discretized[nidx];
                
                // Symmetric GLCM: count both (i,j) and (j,i)
                atomicAdd(&glcm_matrix[bin_i * num_bins + bin_j], 1);
                atomicAdd(&glcm_matrix[bin_j * num_bins + bin_i], 1);
            }
        }
    }
}


// ============================================================================
// Kernel: Normalize GLCM matrix to probabilities
// ============================================================================
// Converts raw co-occurrence counts to probabilities by dividing by total pairs.
// One thread per matrix element for parallel normalization.

__global__ void normalize_glcm_kernel(
    const int* glcm_counts,
    double* glcm_normalized,
    int num_bins,
    const unsigned long long* total_pairs_ptr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = num_bins * num_bins;
    const unsigned long long total_pairs = *total_pairs_ptr;
    
    if (idx < matrix_size) {
        if (total_pairs > 0) {
            glcm_normalized[idx] = (double)glcm_counts[idx] / (double)total_pairs;
        } else {
            glcm_normalized[idx] = 0.0;
        }
    }
}

// ============================================================================
// Kernel: Compute marginal probabilities and coefficients
// ============================================================================
// Computes marginal probabilities px[i] and py[j], means, standard deviations,
// and other coefficients needed for GLCM feature calculation.

__global__ void compute_glcm_marginals_kernel(
    const double* P,
    double* px,
    double* py,
    double* pxAddy,
    double* pxSuby,
    int num_bins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute marginal probabilities px and py
    if (idx < num_bins) {
        double sum_row = 0.0;
        double sum_col = 0.0;
        
        for (int j = 0; j < num_bins; j++) {
            sum_row += P[idx * num_bins + j];  // px[i]
            sum_col += P[j * num_bins + idx];  // py[j]
        }
        
        px[idx] = sum_row;
        py[idx] = sum_col;
    }
    
    // Compute p(i+j) and p(|i-j|)
    // Use a 2D grid to parallelize over all (i,j) pairs
    int i = blockIdx.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < num_bins && j < num_bins) {
        double p = P[i * num_bins + j];
        
        int sum_idx = i + j;
        int diff_idx = abs(i - j);
        
        atomicAdd(&pxAddy[sum_idx], p);
        atomicAdd(&pxSuby[diff_idx], p);
    }
}

// ============================================================================
// Kernel: Compute GLCM texture features
// ============================================================================
// Computes all GLCM texture features in parallel from the normalized GLCM matrix.
// Features include: Contrast, Energy, Entropy, Correlation, Homogeneity, etc.

__global__ void compute_glcm_features_kernel(
    const double* P,
    const double* px,
    const double* py,
    const double* pxAddy,
    const double* pxSuby,
    double* features,
    int num_bins,
    const unsigned long long* total_pairs_ptr
) {
    if (*total_pairs_ptr == 0ULL) {
        return;
    }

    // Shared memory for partial sums
    __shared__ double shared_sums[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_sums[tid] = 0.0;
    
    // Compute means and standard deviations
    double ux = 0.0, uy = 0.0;
    for (int i = 0; i < num_bins; i++) {
        double gray = (double)(i + 1);
        ux += gray * px[i];
        uy += gray * py[i];
    }
    
    double var_x = 0.0, var_y = 0.0;
    for (int i = 0; i < num_bins; i++) {
        double centered = (double)(i + 1) - ux;
        var_x += px[i] * centered * centered;
        centered = (double)(i + 1) - uy;
        var_y += py[i] * centered * centered;
    }
    double sigx = sqrt(var_x);
    double sigy = sqrt(var_y);
    
    // Feature 0: Autocorrelation
    if (blockIdx.y == 0) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            int row = i / num_bins;
            int col = i % num_bins;
            local_sum += P[i] * (double)(row + 1) * (double)(col + 1);
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        // Reduce
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[0], shared_sums[0]);
        }
    }
    
    // Feature 1: Joint Average (ux)
    if (blockIdx.y == 1 && blockIdx.x == 0 && tid == 0) {
        atomicAdd(&features[1], ux);
    }
    
    // Feature 2: Cluster Prominence
    if (blockIdx.y == 2) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            int row = i / num_bins;
            int col = i % num_bins;
            double diff = ((double)(row + 1) + (double)(col + 1) - ux - uy);
            local_sum += P[i] * diff * diff * diff * diff;
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[2], shared_sums[0]);
        }
    }
    
    // Feature 3: Cluster Shade
    if (blockIdx.y == 3) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            int row = i / num_bins;
            int col = i % num_bins;
            double diff = ((double)(row + 1) + (double)(col + 1) - ux - uy);
            local_sum += P[i] * diff * diff * diff;
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[3], shared_sums[0]);
        }
    }
    
    // Feature 4: Cluster Tendency
    if (blockIdx.y == 4) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            int row = i / num_bins;
            int col = i % num_bins;
            double diff = ((double)(row + 1) + (double)(col + 1) - ux - uy);
            local_sum += P[i] * diff * diff;
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[4], shared_sums[0]);
        }
    }
    
    // Feature 5: Contrast
    if (blockIdx.y == 5) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            int row = i / num_bins;
            int col = i % num_bins;
            int diff = abs(row - col);
            local_sum += P[i] * diff * diff;
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[5], shared_sums[0]);
        }
    }
    
    // Feature 6: Correlation
    if (blockIdx.y == 6) {
        if (sigx > 0 && sigy > 0) {
            double local_sum = 0.0;
            for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
                int row = i / num_bins;
                int col = i % num_bins;
                double row_gray = (double)(row + 1);
                double col_gray = (double)(col + 1);
                local_sum += P[i] * (row_gray - ux) * (col_gray - uy);
            }
            shared_sums[tid] = local_sum;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared_sums[tid] += shared_sums[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                atomicAdd(&features[6], shared_sums[0] / (sigx * sigy));
            }
        } else if (blockIdx.x == 0 && tid == 0) {
            atomicAdd(&features[6], 1.0);  // Flat region
        }
    }
    
    // Feature 7: Difference Average
    if (blockIdx.y == 7) {
        double local_sum = 0.0;
        for (int k = idx; k < num_bins; k += blockDim.x * gridDim.x) {
            local_sum += k * pxSuby[k];
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[7], shared_sums[0]);
        }
    }
    
    // Feature 8: Difference Entropy
    if (blockIdx.y == 8) {
        double local_sum = 0.0;
        for (int k = idx; k < num_bins; k += blockDim.x * gridDim.x) {
            if (pxSuby[k] > 0) {
                local_sum -= pxSuby[k] * log2(pxSuby[k] + EPSILON);
            }
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[8], shared_sums[0]);
        }
    }
    
    // Feature 9: Difference Variance
    if (blockIdx.y == 9) {
        // First compute difference average
        double diff_avg = 0.0;
        for (int k = 0; k < num_bins; k++) {
            diff_avg += k * pxSuby[k];
        }
        
        double local_sum = 0.0;
        for (int k = idx; k < num_bins; k += blockDim.x * gridDim.x) {
            local_sum += pxSuby[k] * (k - diff_avg) * (k - diff_avg);
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[9], shared_sums[0]);
        }
    }
    
    // Feature 10: Joint Energy
    if (blockIdx.y == 10) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            local_sum += P[i] * P[i];
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[10], shared_sums[0]);
        }
    }
    
    // Feature 11: Joint Entropy (HXY)
    if (blockIdx.y == 11) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            if (P[i] > 0) {
                local_sum -= P[i] * log2(P[i] + EPSILON);
            }
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[11], shared_sums[0]);
        }
    }
    
    // Feature 12: Homogeneity (ID - Inverse Difference)
    if (blockIdx.y == 12) {
        double local_sum = 0.0;
        for (int k = idx; k < num_bins; k += blockDim.x * gridDim.x) {
            local_sum += pxSuby[k] / (1.0 + k);
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[12], shared_sums[0]);
        }
    }
    
    // Feature 13: IDM (Inverse Difference Moment)
    if (blockIdx.y == 13) {
        double local_sum = 0.0;
        for (int k = idx; k < num_bins; k += blockDim.x * gridDim.x) {
            local_sum += pxSuby[k] / (1.0 + k * k);
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[13], shared_sums[0]);
        }
    }
    
    // Feature 14: IDMN (Inverse Difference Moment Normalized)
    if (blockIdx.y == 14) {
        double local_sum = 0.0;
        for (int k = idx; k < num_bins; k += blockDim.x * gridDim.x) {
            local_sum += pxSuby[k] / (1.0 + (k * k) / (double)(num_bins * num_bins));
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[14], shared_sums[0]);
        }
    }
    
    // Feature 15: IDN (Inverse Difference Normalized)
    if (blockIdx.y == 15) {
        double local_sum = 0.0;
        for (int k = idx; k < num_bins; k += blockDim.x * gridDim.x) {
            local_sum += pxSuby[k] / (1.0 + k / (double)num_bins);
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[15], shared_sums[0]);
        }
    }
    
    // Feature 16: Inverse Variance (skip k=0)
    if (blockIdx.y == 16) {
        double local_sum = 0.0;
        for (int k = idx + 1; k < num_bins; k += blockDim.x * gridDim.x) {
            local_sum += pxSuby[k] / (k * k);
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[16], shared_sums[0]);
        }
    }
    
    // Feature 17: Maximum Probability
    if (blockIdx.y == 17) {
        double local_max = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            if (P[i] > local_max) {
                local_max = P[i];
            }
        }
        shared_sums[tid] = local_max;
        __syncthreads();
        
        // Max reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (shared_sums[tid + s] > shared_sums[tid]) {
                    shared_sums[tid] = shared_sums[tid + s];
                }
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            // Atomic max
            unsigned long long* address_as_ull = (unsigned long long*)&features[17];
            unsigned long long old = *address_as_ull, assumed;
            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(fmax(shared_sums[0], __longlong_as_double(assumed))));
            } while (assumed != old);
        }
    }
    
    // Feature 18: Sum Average
    if (blockIdx.y == 18) {
        double local_sum = 0.0;
        for (int k = idx; k < 2 * num_bins - 1; k += blockDim.x * gridDim.x) {
            local_sum += (k + 2) * pxAddy[k];
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[18], shared_sums[0]);
        }
    }
    
    // Feature 19: Sum Entropy
    if (blockIdx.y == 19) {
        double local_sum = 0.0;
        for (int k = idx; k < 2 * num_bins - 1; k += blockDim.x * gridDim.x) {
            if (pxAddy[k] > 0) {
                local_sum -= pxAddy[k] * log2(pxAddy[k] + EPSILON);
            }
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[19], shared_sums[0]);
        }
    }
    
    // Feature 20: Sum Squares (Joint Variance)
    if (blockIdx.y == 20) {
        double local_sum = 0.0;
        for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
            int row = i / num_bins;
            double row_gray = (double)(row + 1);
            local_sum += P[i] * (row_gray - ux) * (row_gray - ux);
        }
        shared_sums[tid] = local_sum;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_sums[tid] += shared_sums[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            atomicAdd(&features[20], shared_sums[0]);
        }
    }
}

enum {
    GLCM_CORE_ACC_AUTOCORRELATION = 0,
    GLCM_CORE_ACC_CLUSTER_PROMINENCE = 1,
    GLCM_CORE_ACC_CLUSTER_SHADE = 2,
    GLCM_CORE_ACC_CLUSTER_TENDENCY = 3,
    GLCM_CORE_ACC_CORRELATION_NUMERATOR = 4,
    GLCM_CORE_ACC_JOINT_ENERGY = 5,
    GLCM_CORE_ACC_JOINT_ENTROPY = 6,
    GLCM_CORE_ACC_SUM_SQUARES = 7,
    GLCM_CORE_ACC_COUNT = 8
};

__global__ void compute_glcm_core_features_fused_kernel(
    const double* P,
    const double* px,
    const double* py,
    double* features,
    int num_bins,
    const unsigned long long* total_pairs_ptr
) {
    if (*total_pairs_ptr == 0ULL) {
        return;
    }

    __shared__ double shared_acc[GLCM_CORE_ACC_COUNT][256];
    __shared__ double shared_ux;
    __shared__ double shared_uy;
    __shared__ double shared_sigx;
    __shared__ double shared_sigy;

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        double ux = 0.0;
        double uy = 0.0;
        for (int i = 0; i < num_bins; i++) {
            const double gray = (double)(i + 1);
            ux += gray * px[i];
            uy += gray * py[i];
        }

        double var_x = 0.0;
        double var_y = 0.0;
        for (int i = 0; i < num_bins; i++) {
            const double gray = (double)(i + 1);
            const double centered_x = gray - ux;
            const double centered_y = gray - uy;
            var_x += px[i] * centered_x * centered_x;
            var_y += py[i] * centered_y * centered_y;
        }

        shared_ux = ux;
        shared_uy = uy;
        shared_sigx = sqrt(var_x);
        shared_sigy = sqrt(var_y);
    }

    for (int acc_idx = 0; acc_idx < GLCM_CORE_ACC_COUNT; acc_idx++) {
        shared_acc[acc_idx][tid] = 0.0;
    }
    __syncthreads();

    double local_acc[GLCM_CORE_ACC_COUNT];
    for (int acc_idx = 0; acc_idx < GLCM_CORE_ACC_COUNT; acc_idx++) {
        local_acc[acc_idx] = 0.0;
    }

    const double ux = shared_ux;
    const double uy = shared_uy;
    const int matrix_size = num_bins * num_bins;
    for (int linear_idx = idx; linear_idx < matrix_size; linear_idx += blockDim.x * gridDim.x) {
        const double p = P[linear_idx];
        if (p <= 0.0) {
            continue;
        }

        const int row = linear_idx / num_bins;
        const int col = linear_idx % num_bins;
        const double row_gray = (double)(row + 1);
        const double col_gray = (double)(col + 1);
        const double centered_row = row_gray - ux;
        const double centered_col = col_gray - uy;
        const double cluster = row_gray + col_gray - ux - uy;
        const double cluster_sq = cluster * cluster;

        local_acc[GLCM_CORE_ACC_AUTOCORRELATION] += p * row_gray * col_gray;
        local_acc[GLCM_CORE_ACC_CLUSTER_PROMINENCE] += p * cluster_sq * cluster_sq;
        local_acc[GLCM_CORE_ACC_CLUSTER_SHADE] += p * cluster_sq * cluster;
        local_acc[GLCM_CORE_ACC_CLUSTER_TENDENCY] += p * cluster_sq;
        local_acc[GLCM_CORE_ACC_CORRELATION_NUMERATOR] += p * centered_row * centered_col;
        local_acc[GLCM_CORE_ACC_JOINT_ENERGY] += p * p;
        local_acc[GLCM_CORE_ACC_JOINT_ENTROPY] -= p * log2(p + EPSILON);
        local_acc[GLCM_CORE_ACC_SUM_SQUARES] += p * centered_row * centered_row;
    }

    for (int acc_idx = 0; acc_idx < GLCM_CORE_ACC_COUNT; acc_idx++) {
        shared_acc[acc_idx][tid] = local_acc[acc_idx];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int acc_idx = 0; acc_idx < GLCM_CORE_ACC_COUNT; acc_idx++) {
                shared_acc[acc_idx][tid] += shared_acc[acc_idx][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&features[0], shared_acc[GLCM_CORE_ACC_AUTOCORRELATION][0]);
        atomicAdd(&features[2], shared_acc[GLCM_CORE_ACC_CLUSTER_PROMINENCE][0]);
        atomicAdd(&features[3], shared_acc[GLCM_CORE_ACC_CLUSTER_SHADE][0]);
        atomicAdd(&features[4], shared_acc[GLCM_CORE_ACC_CLUSTER_TENDENCY][0]);
        atomicAdd(&features[10], shared_acc[GLCM_CORE_ACC_JOINT_ENERGY][0]);
        atomicAdd(&features[11], shared_acc[GLCM_CORE_ACC_JOINT_ENTROPY][0]);
        atomicAdd(&features[20], shared_acc[GLCM_CORE_ACC_SUM_SQUARES][0]);

        if (shared_sigx > 0.0 && shared_sigy > 0.0) {
            atomicAdd(
                &features[6],
                shared_acc[GLCM_CORE_ACC_CORRELATION_NUMERATOR][0] / (shared_sigx * shared_sigy)
            );
        } else if (blockIdx.x == 0) {
            atomicAdd(&features[6], 1.0);
        }

        if (blockIdx.x == 0) {
            atomicAdd(&features[1], shared_ux);
        }
    }
}

enum {
    GLCM_DIST_ACC_CONTRAST = 0,
    GLCM_DIST_ACC_DIFFERENCE_AVERAGE = 1,
    GLCM_DIST_ACC_DIFFERENCE_ENTROPY = 2,
    GLCM_DIST_ACC_ID = 3,
    GLCM_DIST_ACC_IDM = 4,
    GLCM_DIST_ACC_IDMN = 5,
    GLCM_DIST_ACC_IDN = 6,
    GLCM_DIST_ACC_INVERSE_VARIANCE = 7,
    GLCM_DIST_ACC_SUM_AVERAGE = 8,
    GLCM_DIST_ACC_SUM_ENTROPY = 9,
    GLCM_DIST_ACC_COUNT = 10
};

__global__ void compute_glcm_distribution_features_fused_kernel(
    const double* pxAddy,
    const double* pxSuby,
    double* features,
    int num_bins,
    const unsigned long long* total_pairs_ptr
) {
    if (*total_pairs_ptr == 0ULL) {
        return;
    }

    __shared__ double shared_acc[GLCM_DIST_ACC_COUNT][256];
    const int tid = threadIdx.x;

    double local_acc[GLCM_DIST_ACC_COUNT];
    for (int acc_idx = 0; acc_idx < GLCM_DIST_ACC_COUNT; acc_idx++) {
        local_acc[acc_idx] = 0.0;
        shared_acc[acc_idx][tid] = 0.0;
    }
    __syncthreads();

    const double ng = (double)num_bins;
    const double ng_sq = ng * ng;
    for (int k = tid; k < num_bins; k += blockDim.x) {
        const double p = pxSuby[k];
        const double diff = (double)k;
        const double diff_sq = diff * diff;

        local_acc[GLCM_DIST_ACC_CONTRAST] += diff_sq * p;
        local_acc[GLCM_DIST_ACC_DIFFERENCE_AVERAGE] += diff * p;
        local_acc[GLCM_DIST_ACC_ID] += p / (1.0 + diff);
        local_acc[GLCM_DIST_ACC_IDM] += p / (1.0 + diff_sq);
        local_acc[GLCM_DIST_ACC_IDMN] += p / (1.0 + diff_sq / ng_sq);
        local_acc[GLCM_DIST_ACC_IDN] += p / (1.0 + diff / ng);
        if (k > 0) {
            local_acc[GLCM_DIST_ACC_INVERSE_VARIANCE] += p / diff_sq;
        }
        if (p > 0.0) {
            local_acc[GLCM_DIST_ACC_DIFFERENCE_ENTROPY] -= p * log2(p + EPSILON);
        }
    }

    for (int k = tid; k < 2 * num_bins - 1; k += blockDim.x) {
        const double p = pxAddy[k];
        local_acc[GLCM_DIST_ACC_SUM_AVERAGE] += (double)(k + 2) * p;
        if (p > 0.0) {
            local_acc[GLCM_DIST_ACC_SUM_ENTROPY] -= p * log2(p + EPSILON);
        }
    }

    for (int acc_idx = 0; acc_idx < GLCM_DIST_ACC_COUNT; acc_idx++) {
        shared_acc[acc_idx][tid] = local_acc[acc_idx];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int acc_idx = 0; acc_idx < GLCM_DIST_ACC_COUNT; acc_idx++) {
                shared_acc[acc_idx][tid] += shared_acc[acc_idx][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double contrast = shared_acc[GLCM_DIST_ACC_CONTRAST][0];
        const double diff_avg = shared_acc[GLCM_DIST_ACC_DIFFERENCE_AVERAGE][0];

        atomicAdd(&features[5], contrast);
        atomicAdd(&features[7], diff_avg);
        atomicAdd(&features[8], shared_acc[GLCM_DIST_ACC_DIFFERENCE_ENTROPY][0]);
        atomicAdd(&features[9], contrast - diff_avg * diff_avg);
        atomicAdd(&features[12], shared_acc[GLCM_DIST_ACC_ID][0]);
        atomicAdd(&features[13], shared_acc[GLCM_DIST_ACC_IDM][0]);
        atomicAdd(&features[14], shared_acc[GLCM_DIST_ACC_IDMN][0]);
        atomicAdd(&features[15], shared_acc[GLCM_DIST_ACC_IDN][0]);
        atomicAdd(&features[16], shared_acc[GLCM_DIST_ACC_INVERSE_VARIANCE][0]);
        atomicAdd(&features[18], shared_acc[GLCM_DIST_ACC_SUM_AVERAGE][0]);
        atomicAdd(&features[19], shared_acc[GLCM_DIST_ACC_SUM_ENTROPY][0]);
    }
}

// ============================================================================
// Kernel: Compute IMC entropy terms for one GLCM matrix.
// ============================================================================
__global__ void compute_imc_terms_kernel(
    const double* P,
    const double* px,
    double* imc_terms,
    int num_bins,
    const unsigned long long* total_pairs_ptr
) {
    if (*total_pairs_ptr == 0ULL) {
        return;
    }

    __shared__ double shared_hxy1[256];
    __shared__ double shared_hxy2[256];
    __shared__ double shared_hx[256];
    __shared__ double shared_hxy[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared_hxy1[tid] = 0.0;
    shared_hxy2[tid] = 0.0;
    shared_hx[tid] = 0.0;
    shared_hxy[tid] = 0.0;
    
    // Compute HXY and HXY1
    double local_hxy = 0.0;
    double local_hxy1 = 0.0;
    for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
        int row = i / num_bins;
        int col = i % num_bins;
        if (P[i] > 0) {
            local_hxy -= P[i] * log2(P[i] + EPSILON);
            local_hxy1 -= P[i] * log2(px[row] * px[col] + EPSILON);
        }
    }
    shared_hxy[tid] = local_hxy;
    shared_hxy1[tid] = local_hxy1;
    
    // Compute HXY2
    double local_hxy2 = 0.0;
    for (int i = idx; i < num_bins * num_bins; i += blockDim.x * gridDim.x) {
        int row = i / num_bins;
        int col = i % num_bins;
        double pxy = px[row] * px[col];
        if (pxy > 0) {
            local_hxy2 -= pxy * log2(pxy + EPSILON);
        }
    }
    shared_hxy2[tid] = local_hxy2;
    
    // Compute HX
    double local_hx = 0.0;
    for (int i = idx; i < num_bins; i += blockDim.x * gridDim.x) {
        if (px[i] > 0) {
            local_hx -= px[i] * log2(px[i] + EPSILON);
        }
    }
    shared_hx[tid] = local_hx;
    
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_hxy1[tid] += shared_hxy1[tid + s];
            shared_hxy2[tid] += shared_hxy2[tid + s];
            shared_hx[tid] += shared_hx[tid + s];
            shared_hxy[tid] += shared_hxy[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&imc_terms[0], shared_hxy1[0]);
        atomicAdd(&imc_terms[1], shared_hx[0]);
        atomicAdd(&imc_terms[2], shared_hxy2[0]);
        atomicAdd(&imc_terms[3], shared_hxy[0]);
    }
}

__global__ void accumulate_imc_features_kernel(
    const double* imc_terms,
    double* features,
    const unsigned long long* total_pairs_ptr
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *total_pairs_ptr == 0ULL) {
        return;
    }

    const double hxy1 = imc_terms[0];
    const double hx = imc_terms[1];
    const double hxy2 = imc_terms[2];
    const double hxy = imc_terms[3];

    const double imc1 = (hx == 0.0) ? 0.0 : (hxy - hxy1) / hx;
    const double imc2 =
        (hxy2 <= hxy) ? 0.0 : sqrt(fmax(0.0, 1.0 - exp(-2.0 * (hxy2 - hxy))));

    atomicAdd(&features[21], imc1);
    atomicAdd(&features[22], imc2);
}

__global__ void reduce_glcm_count_stats_kernel(
    const int* glcm_counts,
    int matrix_size,
    unsigned long long* total_pairs_out,
    unsigned int* max_count_out
) {
    __shared__ unsigned long long shared_sum[256];
    __shared__ unsigned int shared_max[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned long long local_sum = 0ULL;
    unsigned int local_max = 0U;
    for (int i = idx; i < matrix_size; i += stride) {
        unsigned int c = (unsigned int)((glcm_counts[i] > 0) ? glcm_counts[i] : 0);
        local_sum += (unsigned long long)c;
        if (c > local_max) {
            local_max = c;
        }
    }

    shared_sum[tid] = local_sum;
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total_pairs_out, shared_sum[0]);
        atomicMax(max_count_out, shared_max[0]);
    }
}

__global__ void accumulate_glcm_matrix_stats_kernel(
    const unsigned long long* total_pairs,
    const unsigned int* max_count,
    int* total_matrices,
    double* max_probability_sum
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const unsigned long long pairs = *total_pairs;
    if (pairs == 0ULL) {
        return;
    }
    atomicAdd(total_matrices, 1);
    atomicAdd(max_probability_sum, (double)(*max_count) / (double)pairs);
}


// ============================================================================
// Helper function: Compute min/max for discretization
// ============================================================================
__global__ void compute_min_max_kernel(
    const float* image,
    const uint8_t* mask,
    float* min_val,
    float* max_val,
    int n_voxels
) {
    __shared__ float shared_min[256];
    __shared__ float shared_max[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    for (int i = idx; i < n_voxels; i += stride) {
        if (mask[i] != 0) {
            float val = image[i];
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }
    }
    
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_min[tid + s] < shared_min[tid]) {
                shared_min[tid] = shared_min[tid + s];
            }
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Atomic min/max
        unsigned int* min_as_uint = (unsigned int*)min_val;
        unsigned int* max_as_uint = (unsigned int*)max_val;
        
        atomicMin(min_as_uint, __float_as_uint(shared_min[0]));
        atomicMax(max_as_uint, __float_as_uint(shared_max[0]));
    }
}

// ============================================================================
// Main API function: flash_radiomics_glcm_cuda
// ============================================================================

FlashRadiomicsResult* flash_radiomics_glcm_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
) {
    // Allocate result structure
    FlashRadiomicsResult* result = (FlashRadiomicsResult*)calloc(1, sizeof(FlashRadiomicsResult));
    if (!result) {
        return NULL;
    }
    
    // Validate inputs
    if (!img || !mask || !ctx || !distances || num_distances <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        snprintf(result->error_msg, sizeof(result->error_msg), 
                 "Invalid input parameters");
        return result;
    }
    
    // Check dimensions match
    if (img->ndim != mask->ndim ||
        img->dims[0] != mask->dims[0] ||
        img->dims[1] != mask->dims[1] ||
        img->dims[2] != mask->dims[2]) {
        result->error_code = FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH;
        snprintf(result->error_msg, sizeof(result->error_msg), 
                 "Image and mask dimensions do not match");
        return result;
    }
    
    // Calculate dimensions
    int n_voxels = img->dims[0] * img->dims[1] * img->dims[2];
    int num_directions = (img->ndim == 2) ? 4 : 13;

    int workspace_status =
        cuda_segment_workspace_prepare(
            ctx, img, mask, bin_width, bin_count, ctx->has_bin_minimum, ctx->bin_minimum
        );
    if (workspace_status != FLASH_CUDA_SUCCESS) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(
            result->error_msg,
            sizeof(result->error_msg),
            "Failed to prepare CUDA segment workspace: %.*s",
            200,
            ctx->error_msg ? ctx->error_msg : ""
        );
        return result;
    }

    const uint8_t* d_mask = cuda_segment_workspace_mask(ctx);
    const int* d_discretized = cuda_segment_workspace_discretized(ctx);
    int num_bins = cuda_segment_workspace_num_bins(ctx);
    if (!d_mask || !d_discretized || num_bins <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "CUDA segment workspace is not ready");
        return result;
    }

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, ctx->stream);
    
    // ========================================================================
    // Allocate GPU memory
    // ========================================================================
    
    int* d_glcm_counts = NULL;
    double* d_glcm_normalized = NULL;
    double* d_px = NULL;
    double* d_py = NULL;
    double* d_pxAddy = NULL;
    double* d_pxSuby = NULL;
    double* d_features = NULL;
    double* d_imc = NULL;
    unsigned long long* d_matrix_total_pairs = NULL;
    unsigned int* d_matrix_max_count = NULL;
    int* d_total_matrices = NULL;
    double* d_max_probability_sum = NULL;
    
    cudaError_t err = cudaSuccess;
    int matrix_size = num_bins * num_bins;
    int block_size = 256;
    int grid_size = (n_voxels + block_size - 1) / block_size;
    int norm_block_size = 256;
    int norm_grid_size = 0;
    int reduction_grid_size = 0;
    int feature_grid_size = 0;
    double* h_features = NULL;
    int h_total_matrices = 0;
    double h_max_probability_sum = 0.0;
    float milliseconds = 0.0f;

    // Allocate GLCM matrices
    err = cudaMalloc(&d_glcm_counts, matrix_size * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_glcm_normalized, matrix_size * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate marginal probabilities
    err = cudaMalloc(&d_px, num_bins * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_py, num_bins * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_pxAddy, (2 * num_bins - 1) * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_pxSuby, num_bins * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate features array (23 features)
    err = cudaMalloc(&d_features, 23 * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    
    // Per-matrix scratch: HXY1, HX, HXY2, HXY.
    err = cudaMalloc(&d_imc, 4 * sizeof(double));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMalloc(&d_matrix_total_pairs, sizeof(unsigned long long));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_matrix_max_count, sizeof(unsigned int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_total_matrices, sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_max_probability_sum, sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    
    // ========================================================================
    // Compute GLCM matrices for all distances and directions
    // ========================================================================
    
    // Initialize accumulated features to zero
    err = cudaMemsetAsync(d_features, 0, 23 * sizeof(double), ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_total_matrices, 0, sizeof(int), ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_max_probability_sum, 0, sizeof(double), ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    
    for (int d_idx = 0; d_idx < num_distances; d_idx++) {
        int distance = distances[d_idx];
        
        for (int dir_idx = 0; dir_idx < num_directions; dir_idx++) {
            // Initialize GLCM matrix to zero
            err = cudaMemsetAsync(d_glcm_counts, 0, matrix_size * sizeof(int), ctx->stream);
            if (err != cudaSuccess) goto cleanup;
            err = cudaMemsetAsync(d_px, 0, num_bins * sizeof(double), ctx->stream);
            if (err != cudaSuccess) goto cleanup;
            err = cudaMemsetAsync(d_py, 0, num_bins * sizeof(double), ctx->stream);
            if (err != cudaSuccess) goto cleanup;
            err = cudaMemsetAsync(d_pxAddy, 0, (2 * num_bins - 1) * sizeof(double), ctx->stream);
            if (err != cudaSuccess) goto cleanup;
            err = cudaMemsetAsync(d_pxSuby, 0, num_bins * sizeof(double), ctx->stream);
            if (err != cudaSuccess) goto cleanup;
            
            // Launch GLCM construction kernel
            if (img->ndim == 2) {
                compute_glcm_2d_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
                    d_discretized, d_mask, d_glcm_counts,
                    img->dims[0], img->dims[1], num_bins, distance, dir_idx
                );
            } else {
                compute_glcm_3d_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
                    d_discretized, d_mask, d_glcm_counts,
                    img->dims[0], img->dims[1], img->dims[2], num_bins, distance, dir_idx
                );
            }
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg), 
                         "GLCM construction kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }

            err = cudaMemsetAsync(d_matrix_total_pairs, 0, sizeof(unsigned long long), ctx->stream);
            if (err != cudaSuccess) goto cleanup;
            err = cudaMemsetAsync(d_matrix_max_count, 0, sizeof(unsigned int), ctx->stream);
            if (err != cudaSuccess) goto cleanup;

            reduction_grid_size = (matrix_size + block_size - 1) / block_size;
            if (reduction_grid_size > 1024) {
                reduction_grid_size = 1024;
            }
            reduce_glcm_count_stats_kernel<<<reduction_grid_size, block_size, 0, ctx->stream>>>(
                d_glcm_counts,
                matrix_size,
                d_matrix_total_pairs,
                d_matrix_max_count
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg),
                         "GLCM stats reduction kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }

            accumulate_glcm_matrix_stats_kernel<<<1, 1, 0, ctx->stream>>>(
                d_matrix_total_pairs,
                d_matrix_max_count,
                d_total_matrices,
                d_max_probability_sum
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg),
                         "GLCM stats accumulation kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }
            
            // Normalize GLCM matrix
            norm_grid_size = (matrix_size + norm_block_size - 1) / norm_block_size;
            
            normalize_glcm_kernel<<<norm_grid_size, norm_block_size, 0, ctx->stream>>>(
                d_glcm_counts, d_glcm_normalized, num_bins, d_matrix_total_pairs
            );
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg), 
                         "Normalization kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }
            
            // Compute marginal probabilities
            compute_glcm_marginals_kernel<<<dim3(1, num_bins), dim3(256), 0, ctx->stream>>>(
                d_glcm_normalized, d_px, d_py, d_pxAddy, d_pxSuby, num_bins
            );
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg), 
                         "Marginals kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }
            
            // Compute feature groups from shared prerequisites in fused passes.
            feature_grid_size = (matrix_size + block_size - 1) / block_size;
            if (feature_grid_size < 1) {
                feature_grid_size = 1;
            }
            if (feature_grid_size > 64) {
                feature_grid_size = 64;
            }

            compute_glcm_core_features_fused_kernel<<<feature_grid_size, block_size, 0, ctx->stream>>>(
                d_glcm_normalized, d_px, d_py, d_features, num_bins, d_matrix_total_pairs
            );
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg), 
                         "Core features kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }

            // Keep this single-block: DifferenceVariance needs the full pxSuby
            // distribution to subtract the matrix-level DifferenceAverage.
            compute_glcm_distribution_features_fused_kernel<<<1, 256, 0, ctx->stream>>>(
                d_pxAddy, d_pxSuby, d_features, num_bins, d_matrix_total_pairs
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg),
                         "Distribution features kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }
            
            // Compute IMC feature values for this matrix and accumulate them.
            err = cudaMemsetAsync(d_imc, 0, 4 * sizeof(double), ctx->stream);
            if (err != cudaSuccess) goto cleanup;

            compute_imc_terms_kernel<<<4, 256, 0, ctx->stream>>>(
                d_glcm_normalized, d_px, d_imc, num_bins, d_matrix_total_pairs
            );
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg), 
                         "IMC kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }

            accumulate_imc_features_kernel<<<1, 1, 0, ctx->stream>>>(
                d_imc,
                d_features,
                d_matrix_total_pairs
            );

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                snprintf(result->error_msg, sizeof(result->error_msg),
                         "IMC accumulation kernel failed: %s", cudaGetErrorString(err));
                goto cleanup;
            }
            
        }
    }
    
    // ========================================================================
    // Transfer results back to CPU and average
    // ========================================================================
    
    h_features = (double*)malloc(23 * sizeof(double));
    if (!h_features) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg), "Failed to allocate host feature buffers");
        goto cleanup;
    }
    
    err = cudaMemcpyAsync(h_features, d_features, 23 * sizeof(double),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(&h_total_matrices, d_total_matrices, sizeof(int),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(&h_max_probability_sum, d_max_probability_sum, sizeof(double),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    
    cudaStreamSynchronize(ctx->stream);
    
    // Average over all matrices
    if (h_total_matrices > 0) {
        for (int i = 0; i < 23; i++) {
            h_features[i] /= h_total_matrices;
        }
        h_features[17] = h_max_probability_sum / (double)h_total_matrices;
    }
    
    // ========================================================================
    // Populate result structure
    // ========================================================================
    
    result->count = 23;
    result->features = (FlashRadiomicsFeature*)malloc(result->count * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg), "Failed to allocate result feature array");
        goto cleanup;
    }
    
    for (int i = 0; i < 23; i++) {
        snprintf(result->features[i].name, sizeof(result->features[i].name), "%s", GLCM_FEATURE_NAMES[i]);
        result->features[i].value = h_features[i];
    }
    
    free(h_features);
    h_features = NULL;
    
    result->error_code = FLASH_RADIOMICS_SUCCESS;
    
    // ========================================================================
    // Record timing and cleanup
    // ========================================================================
    
    cudaEventRecord(stop, ctx->stream);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    result->compute_time_ms = milliseconds;
    
cleanup:
    if (err != cudaSuccess && result->error_code == FLASH_RADIOMICS_SUCCESS) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "CUDA runtime error: %s", cudaGetErrorString(err));
    }

    // Free GPU memory
    if (d_glcm_counts) cudaFree(d_glcm_counts);
    if (d_glcm_normalized) cudaFree(d_glcm_normalized);
    if (d_px) cudaFree(d_px);
    if (d_py) cudaFree(d_py);
    if (d_pxAddy) cudaFree(d_pxAddy);
    if (d_pxSuby) cudaFree(d_pxSuby);
    if (d_features) cudaFree(d_features);
    if (d_imc) cudaFree(d_imc);
    if (d_matrix_total_pairs) cudaFree(d_matrix_total_pairs);
    if (d_matrix_max_count) cudaFree(d_matrix_max_count);
    if (d_total_matrices) cudaFree(d_total_matrices);
    if (d_max_probability_sum) cudaFree(d_max_probability_sum);
    if (h_features) free(h_features);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}
