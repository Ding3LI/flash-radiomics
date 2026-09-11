#include "glrlm_cuda.h"
#include "reduction_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// Number of bins for discretization (default for radiomics)
#define DEFAULT_NUM_BINS 256
#define EPSILON 2.2e-16

static const char* GLRLM_FEATURE_NAMES[16] = {
    "glrlm_ShortRunEmphasis",
    "glrlm_LongRunEmphasis",
    "glrlm_GrayLevelNonUniformity",
    "glrlm_GrayLevelNonUniformityNormalized",
    "glrlm_RunLengthNonUniformity",
    "glrlm_RunLengthNonUniformityNormalized",
    "glrlm_RunPercentage",
    "glrlm_LowGrayLevelRunEmphasis",
    "glrlm_HighGrayLevelRunEmphasis",
    "glrlm_ShortRunLowGrayLevelEmphasis",
    "glrlm_ShortRunHighGrayLevelEmphasis",
    "glrlm_LongRunLowGrayLevelEmphasis",
    "glrlm_LongRunHighGrayLevelEmphasis",
    "glrlm_GrayLevelVariance",
    "glrlm_RunVariance",
    "glrlm_RunEntropy"
};

// Direction vectors for 2D (4 directions: 0°, 45°, 90°, 135°)
__constant__ int DIRECTIONS_2D_GLRLM[4][2] = {
    {1, 0},   // 0° (horizontal)
    {1, 1},   // 45° (diagonal)
    {0, 1},   // 90° (vertical)
    {-1, 1}   // 135° (anti-diagonal)
};

// Direction vectors for 3D (13 directions)
__constant__ int DIRECTIONS_3D_GLRLM[13][3] = {
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
__global__ void discretize_image_glrlm_kernel(
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
// Kernel: Compute GLRLM matrix for 2D images
// ============================================================================
// This kernel computes the Gray Level Run Length Matrix (GLRLM) for 2D images.
// One thread per scan line in the specified direction.
// Each thread sequentially scans along its line to find runs of equal gray levels.
// Uses atomic operations for thread-safe matrix updates.

__global__ void compute_glrlm_2d_kernel(
    const int* discretized,
    const uint8_t* mask,
    int* glrlm_matrix,
    int width,
    int height,
    int num_bins,
    int max_run_length,
    int dir_idx
) {
    // Get direction offset
    int dx = DIRECTIONS_2D_GLRLM[dir_idx][0];
    int dy = DIRECTIONS_2D_GLRLM[dir_idx][1];
    
    // Each thread processes one scan line
    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Determine starting points based on direction
    int start_x, start_y;
    int num_lines;
    
    // Calculate number of scan lines and starting points
    if (dx == 1 && dy == 0) {
        // Horizontal: one line per row
        num_lines = height;
        if (line_idx >= num_lines) return;
        start_x = 0;
        start_y = line_idx;
    } else if (dx == 0 && dy == 1) {
        // Vertical: one line per column
        num_lines = width;
        if (line_idx >= num_lines) return;
        start_x = line_idx;
        start_y = 0;
    } else if (dx == 1 && dy == 1) {
        // Diagonal (45°): lines start from left edge and bottom edge
        num_lines = width + height - 1;
        if (line_idx >= num_lines) return;
        if (line_idx < height) {
            start_x = 0;
            start_y = height - 1 - line_idx;
        } else {
            start_x = line_idx - height + 1;
            start_y = 0;
        }
    } else {  // dx == -1 && dy == 1
        // Anti-diagonal (135°): lines start from right edge and bottom edge
        num_lines = width + height - 1;
        if (line_idx >= num_lines) return;
        if (line_idx < height) {
            start_x = width - 1;
            start_y = height - 1 - line_idx;
        } else {
            start_x = width - 1 - (line_idx - height + 1);
            start_y = 0;
        }
    }
    
    // Scan along the line to find runs
    int x = start_x;
    int y = start_y;
    int current_gray = -1;
    int run_length = 0;
    
    while (x >= 0 && x < width && y >= 0 && y < height) {
        int idx = y * width + x;
        
        if (mask[idx] != 0) {
            int gray_level = discretized[idx];
            
            if (gray_level == current_gray) {
                // Continue current run
                run_length++;
            } else {
                // End previous run and start new one
                if (current_gray >= 0 && run_length > 0) {
                    int clamped_run = (run_length > max_run_length) ? max_run_length : run_length;
                    int matrix_idx = current_gray * max_run_length + (clamped_run - 1);
                    atomicAdd(&glrlm_matrix[matrix_idx], 1);
                }
                current_gray = gray_level;
                run_length = 1;
            }
        } else {
            // Outside mask - end current run
            if (current_gray >= 0 && run_length > 0) {
                int clamped_run = (run_length > max_run_length) ? max_run_length : run_length;
                int matrix_idx = current_gray * max_run_length + (clamped_run - 1);
                atomicAdd(&glrlm_matrix[matrix_idx], 1);
            }
            current_gray = -1;
            run_length = 0;
        }
        
        // Move to next position
        x += dx;
        y += dy;
    }
    
    // End final run if exists
    if (current_gray >= 0 && run_length > 0) {
        int clamped_run = (run_length > max_run_length) ? max_run_length : run_length;
        int matrix_idx = current_gray * max_run_length + (clamped_run - 1);
        atomicAdd(&glrlm_matrix[matrix_idx], 1);
    }
}

// ============================================================================
// Kernel: Compute GLRLM matrix for 3D images
// ============================================================================
// Similar to 2D kernel but handles 3D spatial relationships.
// Processes 3D volumes with 13 directional scan lines.

__global__ void compute_glrlm_3d_kernel(
    const int* discretized,
    const uint8_t* mask,
    int* glrlm_matrix,
    int width,
    int height,
    int depth,
    int num_bins,
    int max_run_length,
    int dir_idx
) {
    // Get direction offset
    int dx = DIRECTIONS_3D_GLRLM[dir_idx][0];
    int dy = DIRECTIONS_3D_GLRLM[dir_idx][1];
    int dz = DIRECTIONS_3D_GLRLM[dir_idx][2];
    
    int slice_size = width * height;
    
    // Each thread processes one scan line
    // For 3D, we need to enumerate all possible starting points
    // This is more complex than 2D, so we use a simpler approach:
    // Each thread processes a starting voxel and scans in the direction
    
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int voxel_stride = blockDim.x * gridDim.x;
    int total_voxels = width * height * depth;

    for (int linear_idx = voxel_idx; linear_idx < total_voxels; linear_idx += voxel_stride) {
        // Convert linear index to 3D coordinates
        int z = linear_idx / slice_size;
        int rem = linear_idx % slice_size;
        int y = rem / width;
        int x = rem % width;

        // A geometric line is processed only from its boundary start voxel.
        int prev_x = x - dx;
        int prev_y = y - dy;
        int prev_z = z - dz;
        bool is_start = (
            prev_x < 0 || prev_x >= width ||
            prev_y < 0 || prev_y >= height ||
            prev_z < 0 || prev_z >= depth
        );
        if (!is_start) {
            continue;
        }

        // Scan along the full line and split runs at mask gaps.
        int cx = x;
        int cy = y;
        int cz = z;
        int current_gray = -1;
        int run_length = 0;

        while (cx >= 0 && cx < width &&
               cy >= 0 && cy < height &&
               cz >= 0 && cz < depth) {
            int idx = cz * slice_size + cy * width + cx;

            if (mask[idx] != 0) {
                int gray_level = discretized[idx];

                if (gray_level == current_gray) {
                    run_length++;
                } else {
                    if (current_gray >= 0 && run_length > 0) {
                        int clamped_run = (run_length > max_run_length) ? max_run_length : run_length;
                        int matrix_idx = current_gray * max_run_length + (clamped_run - 1);
                        atomicAdd(&glrlm_matrix[matrix_idx], 1);
                    }
                    current_gray = gray_level;
                    run_length = 1;
                }
            } else {
                if (current_gray >= 0 && run_length > 0) {
                    int clamped_run = (run_length > max_run_length) ? max_run_length : run_length;
                    int matrix_idx = current_gray * max_run_length + (clamped_run - 1);
                    atomicAdd(&glrlm_matrix[matrix_idx], 1);
                }
                current_gray = -1;
                run_length = 0;
            }

            cx += dx;
            cy += dy;
            cz += dz;
        }

        if (current_gray >= 0 && run_length > 0) {
            int clamped_run = (run_length > max_run_length) ? max_run_length : run_length;
            int matrix_idx = current_gray * max_run_length + (clamped_run - 1);
            atomicAdd(&glrlm_matrix[matrix_idx], 1);
        }
    }
}

// ============================================================================
// Kernel: Compute GLRLM texture features
// ============================================================================
// Computes all GLRLM texture features in parallel from the GLRLM matrix.
// Features include: Short/Long Run Emphasis, Gray Level Non-Uniformity,
// Run Length Non-Uniformity, Run Percentage, Low/High Gray Level Run Emphasis,
// and combined emphasis features.

__global__ void compute_glrlm_features_kernel(
    const int* glrlm,
    double* features,
    int num_bins,
    int max_run_length,
    const unsigned long long* num_runs_ptr
) {
    // Shared memory for partial sums
    __shared__ double shared_sums[256];
    
    int tid = threadIdx.x;
    int feature_idx = blockIdx.y;
    
    // Initialize shared memory
    shared_sums[tid] = 0.0;
    
    // Compute total number of runs (Ns)
    double Ns = (double)(*num_runs_ptr);
    if (Ns == 0.0) {
        return;
    }
    
    // Feature 0: Short Run Emphasis (SRE)
    if (feature_idx == 0) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int run = i % max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double j = (double)(run + 1);  // Run length (1-indexed)
                local_sum += (double)count / (j * j);
            }
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
            features[0] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 1: Long Run Emphasis (LRE)
    if (feature_idx == 1) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int run = i % max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double j = (double)(run + 1);
                local_sum += (double)count * j * j;
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
            features[1] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 2: Gray Level Non-Uniformity (GLN)
    if (feature_idx == 2) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins; i += blockDim.x) {
            double row_sum = 0.0;
            for (int j = 0; j < max_run_length; j++) {
                row_sum += (double)glrlm[i * max_run_length + j];
            }
            local_sum += row_sum * row_sum;
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
            features[2] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 3: Gray Level Non-Uniformity Normalized (GLNN)
    if (feature_idx == 3) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins; i += blockDim.x) {
            double row_sum = 0.0;
            for (int j = 0; j < max_run_length; j++) {
                row_sum += (double)glrlm[i * max_run_length + j];
            }
            local_sum += row_sum * row_sum;
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
            features[3] += shared_sums[0] / (Ns * Ns);
        }
    }
    
    // Feature 4: Run Length Non-Uniformity (RLN)
    if (feature_idx == 4) {
        double local_sum = 0.0;
        for (int j = tid; j < max_run_length; j += blockDim.x) {
            double col_sum = 0.0;
            for (int i = 0; i < num_bins; i++) {
                col_sum += (double)glrlm[i * max_run_length + j];
            }
            local_sum += col_sum * col_sum;
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
            features[4] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 5: Run Length Non-Uniformity Normalized (RLNN)
    if (feature_idx == 5) {
        double local_sum = 0.0;
        for (int j = tid; j < max_run_length; j += blockDim.x) {
            double col_sum = 0.0;
            for (int i = 0; i < num_bins; i++) {
                col_sum += (double)glrlm[i * max_run_length + j];
            }
            local_sum += col_sum * col_sum;
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
            features[5] += shared_sums[0] / (Ns * Ns);
        }
    }
    
    // Feature 6: Run Percentage (RP)
    // This needs to be computed on host with total voxels
    
    // Feature 7: Low Gray Level Run Emphasis (LGLRE)
    if (feature_idx == 7) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int gray = i / max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double g = (double)(gray + 1);  // Gray level (1-indexed)
                local_sum += (double)count / (g * g);
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
            features[7] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 8: High Gray Level Run Emphasis (HGLRE)
    if (feature_idx == 8) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int gray = i / max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double g = (double)(gray + 1);
                local_sum += (double)count * g * g;
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
            features[8] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 9: Short Run Low Gray Level Emphasis (SRLGLE)
    if (feature_idx == 9) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int gray = i / max_run_length;
            int run = i % max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double g = (double)(gray + 1);
                double j = (double)(run + 1);
                local_sum += (double)count / (g * g * j * j);
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
            features[9] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 10: Short Run High Gray Level Emphasis (SRHGLE)
    if (feature_idx == 10) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int gray = i / max_run_length;
            int run = i % max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double g = (double)(gray + 1);
                double j = (double)(run + 1);
                local_sum += (double)count * g * g / (j * j);
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
            features[10] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 11: Long Run Low Gray Level Emphasis (LRLGLE)
    if (feature_idx == 11) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int gray = i / max_run_length;
            int run = i % max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double g = (double)(gray + 1);
                double j = (double)(run + 1);
                local_sum += (double)count * j * j / (g * g);
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
            features[11] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 12: Long Run High Gray Level Emphasis (LRHGLE)
    if (feature_idx == 12) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int gray = i / max_run_length;
            int run = i % max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double g = (double)(gray + 1);
                double j = (double)(run + 1);
                local_sum += (double)count * g * g * j * j;
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
            features[12] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 13: Gray Level Variance (GLV)
    if (feature_idx == 13) {
        // First compute mean gray level
        double mean_gray = 0.0;
        for (int i = 0; i < num_bins * max_run_length; i++) {
            int gray = i / max_run_length;
            int count = glrlm[i];
            mean_gray += (double)count * (double)(gray + 1);
        }
        mean_gray /= Ns;
        
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int gray = i / max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double g = (double)(gray + 1);
                double diff = g - mean_gray;
                local_sum += (double)count * diff * diff;
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
            features[13] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 14: Run Length Variance (RLV)
    if (feature_idx == 14) {
        // First compute mean run length
        double mean_run = 0.0;
        for (int i = 0; i < num_bins * max_run_length; i++) {
            int run = i % max_run_length;
            int count = glrlm[i];
            mean_run += (double)count * (double)(run + 1);
        }
        mean_run /= Ns;
        
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int run = i % max_run_length;
            int count = glrlm[i];
            if (count > 0) {
                double j = (double)(run + 1);
                double diff = j - mean_run;
                local_sum += (double)count * diff * diff;
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
            features[14] += shared_sums[0] / Ns;
        }
    }
    
    // Feature 15: Run Entropy (RE)
    if (feature_idx == 15) {
        double local_sum = 0.0;
        for (int i = tid; i < num_bins * max_run_length; i += blockDim.x) {
            int count = glrlm[i];
            if (count > 0) {
                double p = (double)count / Ns;
                local_sum -= p * log2(p + EPSILON);
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
            features[15] += shared_sums[0];
        }
    }
}

enum {
    GLRLM_FUSED_ACC_SRE = 0,
    GLRLM_FUSED_ACC_LRE = 1,
    GLRLM_FUSED_ACC_LGLRE = 2,
    GLRLM_FUSED_ACC_HGLRE = 3,
    GLRLM_FUSED_ACC_SRLGLE = 4,
    GLRLM_FUSED_ACC_SRHGLE = 5,
    GLRLM_FUSED_ACC_LRLGLE = 6,
    GLRLM_FUSED_ACC_LRHGLE = 7,
    GLRLM_FUSED_ACC_GRAY_MEAN_NUM = 8,
    GLRLM_FUSED_ACC_RUN_MEAN_NUM = 9,
    GLRLM_FUSED_ACC_RUN_ENTROPY = 10,
    GLRLM_FUSED_ACC_GLN = 11,
    GLRLM_FUSED_ACC_RLN = 12,
    GLRLM_FUSED_ACC_COUNT = 13
};

__global__ void compute_glrlm_features_fused_kernel(
    const int* glrlm,
    double* features,
    int num_bins,
    int max_run_length,
    const unsigned long long* num_runs_ptr
) {
    __shared__ double shared_acc[GLRLM_FUSED_ACC_COUNT][256];
    __shared__ double shared_gray_mean;
    __shared__ double shared_run_mean;

    const int tid = threadIdx.x;
    const double Ns = (double)(*num_runs_ptr);
    if (Ns == 0.0) {
        return;
    }

    double local_acc[GLRLM_FUSED_ACC_COUNT];
    for (int acc_idx = 0; acc_idx < GLRLM_FUSED_ACC_COUNT; acc_idx++) {
        local_acc[acc_idx] = 0.0;
        shared_acc[acc_idx][tid] = 0.0;
    }

    const int matrix_size = num_bins * max_run_length;
    for (int idx = tid; idx < matrix_size; idx += blockDim.x) {
        const int count = glrlm[idx];
        if (count <= 0) {
            continue;
        }

        const int gray = idx / max_run_length;
        const int run = idx % max_run_length;
        const double c = (double)count;
        const double g = (double)(gray + 1);
        const double j = (double)(run + 1);
        const double g_sq = g * g;
        const double j_sq = j * j;
        const double p = c / Ns;

        local_acc[GLRLM_FUSED_ACC_SRE] += c / j_sq;
        local_acc[GLRLM_FUSED_ACC_LRE] += c * j_sq;
        local_acc[GLRLM_FUSED_ACC_LGLRE] += c / g_sq;
        local_acc[GLRLM_FUSED_ACC_HGLRE] += c * g_sq;
        local_acc[GLRLM_FUSED_ACC_SRLGLE] += c / (g_sq * j_sq);
        local_acc[GLRLM_FUSED_ACC_SRHGLE] += c * g_sq / j_sq;
        local_acc[GLRLM_FUSED_ACC_LRLGLE] += c * j_sq / g_sq;
        local_acc[GLRLM_FUSED_ACC_LRHGLE] += c * g_sq * j_sq;
        local_acc[GLRLM_FUSED_ACC_GRAY_MEAN_NUM] += c * g;
        local_acc[GLRLM_FUSED_ACC_RUN_MEAN_NUM] += c * j;
        local_acc[GLRLM_FUSED_ACC_RUN_ENTROPY] -= p * log2(p + EPSILON);
    }

    for (int gray = tid; gray < num_bins; gray += blockDim.x) {
        double row_sum = 0.0;
        const int row_offset = gray * max_run_length;
        for (int run = 0; run < max_run_length; run++) {
            row_sum += (double)glrlm[row_offset + run];
        }
        local_acc[GLRLM_FUSED_ACC_GLN] += row_sum * row_sum;
    }

    for (int run = tid; run < max_run_length; run += blockDim.x) {
        double col_sum = 0.0;
        for (int gray = 0; gray < num_bins; gray++) {
            col_sum += (double)glrlm[gray * max_run_length + run];
        }
        local_acc[GLRLM_FUSED_ACC_RLN] += col_sum * col_sum;
    }

    for (int acc_idx = 0; acc_idx < GLRLM_FUSED_ACC_COUNT; acc_idx++) {
        shared_acc[acc_idx][tid] = local_acc[acc_idx];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int acc_idx = 0; acc_idx < GLRLM_FUSED_ACC_COUNT; acc_idx++) {
                shared_acc[acc_idx][tid] += shared_acc[acc_idx][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        shared_gray_mean = shared_acc[GLRLM_FUSED_ACC_GRAY_MEAN_NUM][0] / Ns;
        shared_run_mean = shared_acc[GLRLM_FUSED_ACC_RUN_MEAN_NUM][0] / Ns;
    }
    __syncthreads();

    double local_gray_var = 0.0;
    double local_run_var = 0.0;
    for (int idx = tid; idx < matrix_size; idx += blockDim.x) {
        const int count = glrlm[idx];
        if (count <= 0) {
            continue;
        }

        const int gray = idx / max_run_length;
        const int run = idx % max_run_length;
        const double c = (double)count;
        const double gray_diff = (double)(gray + 1) - shared_gray_mean;
        const double run_diff = (double)(run + 1) - shared_run_mean;
        local_gray_var += c * gray_diff * gray_diff;
        local_run_var += c * run_diff * run_diff;
    }

    shared_acc[GLRLM_FUSED_ACC_GRAY_MEAN_NUM][tid] = local_gray_var;
    shared_acc[GLRLM_FUSED_ACC_RUN_MEAN_NUM][tid] = local_run_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_acc[GLRLM_FUSED_ACC_GRAY_MEAN_NUM][tid] +=
                shared_acc[GLRLM_FUSED_ACC_GRAY_MEAN_NUM][tid + stride];
            shared_acc[GLRLM_FUSED_ACC_RUN_MEAN_NUM][tid] +=
                shared_acc[GLRLM_FUSED_ACC_RUN_MEAN_NUM][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double inv_ns = 1.0 / Ns;
        const double gray_second = shared_acc[GLRLM_FUSED_ACC_HGLRE][0] * inv_ns;
        const double run_second = shared_acc[GLRLM_FUSED_ACC_LRE][0] * inv_ns;

        features[0] += shared_acc[GLRLM_FUSED_ACC_SRE][0] * inv_ns;
        features[1] += run_second;
        features[2] += shared_acc[GLRLM_FUSED_ACC_GLN][0] * inv_ns;
        features[3] += shared_acc[GLRLM_FUSED_ACC_GLN][0] * inv_ns * inv_ns;
        features[4] += shared_acc[GLRLM_FUSED_ACC_RLN][0] * inv_ns;
        features[5] += shared_acc[GLRLM_FUSED_ACC_RLN][0] * inv_ns * inv_ns;
        features[7] += shared_acc[GLRLM_FUSED_ACC_LGLRE][0] * inv_ns;
        features[8] += gray_second;
        features[9] += shared_acc[GLRLM_FUSED_ACC_SRLGLE][0] * inv_ns;
        features[10] += shared_acc[GLRLM_FUSED_ACC_SRHGLE][0] * inv_ns;
        features[11] += shared_acc[GLRLM_FUSED_ACC_LRLGLE][0] * inv_ns;
        features[12] += shared_acc[GLRLM_FUSED_ACC_LRHGLE][0] * inv_ns;
        features[13] += shared_acc[GLRLM_FUSED_ACC_GRAY_MEAN_NUM][0] * inv_ns;
        features[14] += shared_acc[GLRLM_FUSED_ACC_RUN_MEAN_NUM][0] * inv_ns;
        features[15] += shared_acc[GLRLM_FUSED_ACC_RUN_ENTROPY][0];
    }
}

__global__ void reduce_glrlm_run_sum_kernel(
    const int* glrlm,
    int matrix_size,
    unsigned long long* run_sum_out
) {
    __shared__ unsigned long long shared_sum[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned long long local_sum = 0ULL;
    for (int i = idx; i < matrix_size; i += stride) {
        int v = glrlm[i];
        if (v > 0) {
            local_sum += (unsigned long long)v;
        }
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(run_sum_out, shared_sum[0]);
    }
}

__global__ void accumulate_glrlm_direction_stats_kernel(
    const unsigned long long* dir_runs,
    unsigned long long* total_runs,
    int* valid_directions
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const unsigned long long runs = *dir_runs;
    if (runs == 0ULL) {
        return;
    }
    atomicAdd(total_runs, runs);
    atomicAdd(valid_directions, 1);
}

// ============================================================================
// Helper function: Compute min/max for discretization
// ============================================================================
__global__ void compute_min_max_glrlm_kernel(
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
// Main API function: flash_radiomics_glrlm_cuda
// ============================================================================

FlashRadiomicsResult* flash_radiomics_glrlm_cuda(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    CudaContext* ctx,
    double bin_width,
    int bin_count
) {
    // Allocate result structure
    FlashRadiomicsResult* result = (FlashRadiomicsResult*)calloc(1, sizeof(FlashRadiomicsResult));
    if (!result) {
        return NULL;
    }
    
    // Validate inputs
    if (!img || !mask || !ctx) {
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
    int max_run_length = img->dims[0];
    if (img->dims[1] > max_run_length) max_run_length = img->dims[1];
    if (img->dims[2] > max_run_length) max_run_length = img->dims[2];
    if (max_run_length < 1) max_run_length = 1;

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
    
    int* d_glrlm_matrix = NULL;
    double* d_features = NULL;
    unsigned long long* d_dir_runs = NULL;
    unsigned long long* d_total_runs = NULL;
    int* d_valid_directions = NULL;
    
    cudaError_t err = cudaSuccess;
    int matrix_size = num_bins * max_run_length;
    int block_size = 256;
    int total_voxels_in_mask = 0;
    uint8_t* h_mask = mask->data;
    double* h_features = NULL;
    unsigned long long h_total_runs = 0ULL;
    int h_valid_directions = 0;
    int run_reduction_grid_size = 0;
    float milliseconds = 0.0f;

    // Allocate GLRLM matrix
    err = cudaMalloc(&d_glrlm_matrix, matrix_size * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate features array (16 features)
    err = cudaMalloc(&d_features, 16 * sizeof(double));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_dir_runs, sizeof(unsigned long long));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_total_runs, sizeof(unsigned long long));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_valid_directions, sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    // ========================================================================
    // Compute GLRLM matrices for all directions
    // ========================================================================
    
    // Initialize accumulated features to zero
    err = cudaMemsetAsync(d_features, 0, 16 * sizeof(double), ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_total_runs, 0, sizeof(unsigned long long), ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemsetAsync(d_valid_directions, 0, sizeof(int), ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    
    // Count voxels in mask for run percentage calculation
    for (int i = 0; i < n_voxels; i++) {
        if (h_mask[i] != 0) {
            total_voxels_in_mask++;
        }
    }
    
    for (int dir_idx = 0; dir_idx < num_directions; dir_idx++) {
        // Initialize GLRLM matrix to zero
        err = cudaMemsetAsync(d_glrlm_matrix, 0, matrix_size * sizeof(int), ctx->stream);
        if (err != cudaSuccess) goto cleanup;
        
        // Launch GLRLM construction kernel
        if (img->ndim == 2) {
            // For 2D, calculate number of scan lines based on direction
            int num_lines;
            if (dir_idx == 0) {
                num_lines = img->dims[1];  // Horizontal
            } else if (dir_idx == 2) {
                num_lines = img->dims[0];  // Vertical
            } else {
                num_lines = img->dims[0] + img->dims[1] - 1;  // Diagonals
            }
            
            int glrlm_block_size = 256;
            int glrlm_grid_size = (num_lines + glrlm_block_size - 1) / glrlm_block_size;
            
            compute_glrlm_2d_kernel<<<glrlm_grid_size, glrlm_block_size, 0, ctx->stream>>>(
                d_discretized, d_mask, d_glrlm_matrix,
                img->dims[0], img->dims[1], num_bins, max_run_length, dir_idx
            );
        } else {
            // For 3D, each voxel is a potential starting point
            int glrlm_block_size = 256;
            int glrlm_grid_size = (n_voxels + glrlm_block_size - 1) / glrlm_block_size;
            
            compute_glrlm_3d_kernel<<<glrlm_grid_size, glrlm_block_size, 0, ctx->stream>>>(
                d_discretized, d_mask, d_glrlm_matrix,
                img->dims[0], img->dims[1], img->dims[2], num_bins, max_run_length, dir_idx
            );
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            snprintf(result->error_msg, sizeof(result->error_msg), 
                     "GLRLM construction kernel failed: %s", cudaGetErrorString(err));
            goto cleanup;
        }

        err = cudaMemsetAsync(d_dir_runs, 0, sizeof(unsigned long long), ctx->stream);
        if (err != cudaSuccess) goto cleanup;

        run_reduction_grid_size = (matrix_size + block_size - 1) / block_size;
        if (run_reduction_grid_size > 1024) {
            run_reduction_grid_size = 1024;
        }
        reduce_glrlm_run_sum_kernel<<<run_reduction_grid_size, block_size, 0, ctx->stream>>>(
            d_glrlm_matrix,
            matrix_size,
            d_dir_runs
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            snprintf(result->error_msg, sizeof(result->error_msg),
                     "GLRLM run reduction kernel failed: %s", cudaGetErrorString(err));
            goto cleanup;
        }

        accumulate_glrlm_direction_stats_kernel<<<1, 1, 0, ctx->stream>>>(
            d_dir_runs,
            d_total_runs,
            d_valid_directions
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            snprintf(result->error_msg, sizeof(result->error_msg),
                     "GLRLM direction stats kernel failed: %s", cudaGetErrorString(err));
            goto cleanup;
        }
        
        // Compute all direction features in one fused pass over the matrix.
        compute_glrlm_features_fused_kernel<<<1, 256, 0, ctx->stream>>>(
            d_glrlm_matrix, d_features, num_bins, max_run_length, d_dir_runs
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            snprintf(result->error_msg, sizeof(result->error_msg), 
                     "Features kernel failed: %s", cudaGetErrorString(err));
            goto cleanup;
        }
    }
    
    // ========================================================================
    // Transfer results back to CPU and average
    // ========================================================================
    
    h_features = (double*)malloc(16 * sizeof(double));
    if (!h_features) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg), "Failed to allocate host feature buffer");
        goto cleanup;
    }
    
    err = cudaMemcpyAsync(h_features, d_features, 16 * sizeof(double),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(&h_total_runs, d_total_runs, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(&h_valid_directions, d_valid_directions, sizeof(int),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) goto cleanup;
    
    cudaStreamSynchronize(ctx->stream);
    
    // Average over all directions
    if (h_valid_directions > 0) {
        for (int i = 0; i < 16; i++) {
            if (i != 6) {  // Skip run percentage for now
                h_features[i] /= h_valid_directions;
            }
        }
        
        // Compute Run Percentage (feature 6)
        if (total_voxels_in_mask > 0) {
            h_features[6] = (double)h_total_runs / (double)(total_voxels_in_mask * h_valid_directions);
        } else {
            h_features[6] = 0.0;
        }
    }
    
    // ========================================================================
    // Populate result structure
    // ========================================================================
    
    result->count = 16;
    result->features = (FlashRadiomicsFeature*)malloc(result->count * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg), "Failed to allocate result feature array");
        goto cleanup;
    }
    
    for (int i = 0; i < 16; i++) {
        snprintf(result->features[i].name, sizeof(result->features[i].name), "%s", GLRLM_FEATURE_NAMES[i]);
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
    if (d_glrlm_matrix) cudaFree(d_glrlm_matrix);
    if (d_features) cudaFree(d_features);
    if (d_dir_runs) cudaFree(d_dir_runs);
    if (d_total_runs) cudaFree(d_total_runs);
    if (d_valid_directions) cudaFree(d_valid_directions);
    if (h_features) free(h_features);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}
