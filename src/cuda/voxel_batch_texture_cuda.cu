#include "cuda_common.h"
#include "glcm_cuda.h"
#include "glrlm_cuda.h"
#include "glszm_cuda.h"
#include "gldm_cuda.h"
#include "ngtdm_cuda.h"

extern "C" {
#include "../features/glcm_cpu.h"
#include "../features/glrlm_cpu.h"
#include "../features/glszm_cpu.h"
#include "../features/gldm_cpu.h"
#include "../features/ngtdm_cpu.h"
}

#include <cuda_runtime.h>

#include <chrono>
#include <cctype>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

static const double VOXEL_BATCH_EPS = 2.2e-16;

struct VoxelCudaContextCache {
    CudaContext* ctx;

    VoxelCudaContextCache() : ctx(NULL) {}

    ~VoxelCudaContextCache() {
        if (ctx) {
            cuda_free_context(ctx);
            ctx = NULL;
        }
    }
};

static thread_local VoxelCudaContextCache g_voxel_cuda_context_cache;

static CudaContext* get_cached_voxel_cuda_context(void) {
    if (g_voxel_cuda_context_cache.ctx &&
        g_voxel_cuda_context_cache.ctx->error_code == FLASH_CUDA_SUCCESS) {
        return g_voxel_cuda_context_cache.ctx;
    }

    if (g_voxel_cuda_context_cache.ctx) {
        cuda_free_context(g_voxel_cuda_context_cache.ctx);
        g_voxel_cuda_context_cache.ctx = NULL;
    }

    CudaContext* ctx = cuda_init_context(0);
    if (!ctx) {
        return NULL;
    }
    if (ctx->error_code != FLASH_CUDA_SUCCESS) {
        cuda_free_context(ctx);
        return NULL;
    }

    g_voxel_cuda_context_cache.ctx = ctx;
    return g_voxel_cuda_context_cache.ctx;
}

struct WindowSpec {
    int ndim;
    int dims[3];
    size_t window_voxels;
};

static int init_window_spec(int ndim, const int window_dims[3], WindowSpec* spec) {
    if (!window_dims || !spec) {
        return 0;
    }
    if (ndim != 2 && ndim != 3) {
        return 0;
    }
    spec->ndim = ndim;
    spec->dims[0] = window_dims[0];
    spec->dims[1] = window_dims[1];
    spec->dims[2] = (ndim == 3) ? window_dims[2] : 1;
    if (spec->dims[0] <= 0 || spec->dims[1] <= 0 || spec->dims[2] <= 0) {
        return 0;
    }
    spec->window_voxels =
        (size_t)spec->dims[0] * (size_t)spec->dims[1] * (size_t)spec->dims[2];
    return 1;
}

static int count_enabled_features(const uint8_t* include_features, int feature_count) {
    if (!include_features) {
        return feature_count;
    }
    int count = 0;
    for (int i = 0; i < feature_count; i++) {
        if (include_features[i] != 0) {
            count++;
        }
    }
    return count;
}

static int safe_mul_size_t(size_t a, size_t b, size_t* out) {
    if (!out) {
        return 0;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return 1;
    }
    if (a > (std::numeric_limits<size_t>::max() / b)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int safe_add_size_t(size_t a, size_t b, size_t* out) {
    if (!out || a > (std::numeric_limits<size_t>::max() - b)) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int compute_grid_size_1d(size_t total, int block_size, int max_grid) {
    if (block_size <= 0) {
        return 1;
    }
    size_t grid = (total + (size_t)block_size - 1u) / (size_t)block_size;
    if (grid < 1u) {
        grid = 1u;
    }
    if (grid > (size_t)max_grid) {
        grid = (size_t)max_grid;
    }
    return (int)grid;
}

static int flash_radiomics_env_truthy(const char* name) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return 0;
    }

    char first = (char)std::tolower((unsigned char)value[0]);
    return first == '1' || first == 't' || first == 'y' || first == 'o';
}

static double elapsed_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& stop
) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

static int parse_positive_env_int(const char* name, int* value_out) {
    if (!name || !value_out) {
        return 0;
    }
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return 0;
    }
    char* end_ptr = NULL;
    long parsed = std::strtol(raw, &end_ptr, 10);
    if (end_ptr == raw || parsed <= 0 || parsed > INT_MAX) {
        return 0;
    }
    while (end_ptr && *end_ptr != '\0') {
        if (!std::isspace((unsigned char)(*end_ptr))) {
            return 0;
        }
        end_ptr++;
    }
    *value_out = (int)parsed;
    return 1;
}

static int normalize_block_size(int requested, int fallback) {
    int block_size = requested;
    if (block_size <= 0) {
        block_size = fallback;
    }
    if (block_size < 32) {
        block_size = 32;
    }
    if (block_size > 512) {
        block_size = 512;
    }
    int snapped = 32;
    while ((snapped << 1) <= block_size && snapped < 512) {
        snapped <<= 1;
    }
    if (snapped < 32) {
        snapped = 32;
    }
    return snapped;
}

static int choose_voxel_build_block_size(size_t window_voxels, int ng) {
    int block_size = 256;
    if (window_voxels >= 343u || ng >= 128) {
        block_size = 128;
    }

    int env_block_size = 0;
    if (parse_positive_env_int("FLASH_RADIOMICS_VOXEL_CUDA_BLOCK_SIZE", &env_block_size)) {
        block_size = env_block_size;
    }
    return normalize_block_size(block_size, 256);
}

static int choose_voxel_finalize_block_size(size_t current_batch) {
    int block_size = 32;
    if (current_batch >= 4096u) {
        block_size = 128;
    } else if (current_batch >= 1024u) {
        block_size = 64;
    }

    int env_block_size = 0;
    if (parse_positive_env_int("FLASH_RADIOMICS_VOXEL_CUDA_FINALIZE_BLOCK_SIZE", &env_block_size)) {
        block_size = env_block_size;
    }
    return normalize_block_size(block_size, 32);
}

static size_t choose_chunk_target_bytes(
    size_t default_target_bytes,
    size_t window_voxels,
    int ng
) {
    size_t target_bytes = default_target_bytes;
    if (window_voxels >= 729u || ng >= 192) {
        target_bytes = default_target_bytes / 3u;
    } else if (window_voxels >= 343u || ng >= 128) {
        target_bytes = default_target_bytes / 2u;
    } else if (window_voxels <= 27u && ng <= 64) {
        if (default_target_bytes <= (std::numeric_limits<size_t>::max() / 2u)) {
            target_bytes = default_target_bytes * 2u;
        }
    }

    int env_chunk_target_mb = 0;
    if (parse_positive_env_int("FLASH_RADIOMICS_VOXEL_CUDA_CHUNK_TARGET_MB", &env_chunk_target_mb)) {
        size_t env_target_bytes = 0;
        if (safe_mul_size_t((size_t)env_chunk_target_mb, 1024u * 1024u, &env_target_bytes)) {
            target_bytes = env_target_bytes;
        }
    }

    const size_t min_target_bytes = 8ull * 1024ull * 1024ull;
    const size_t max_target_bytes = 1024ull * 1024ull * 1024ull;
    if (target_bytes < min_target_bytes) {
        target_bytes = min_target_bytes;
    }
    if (target_bytes > max_target_bytes) {
        target_bytes = max_target_bytes;
    }
    return target_bytes;
}

static size_t cap_chunk_batch_by_target(
    size_t requested_chunk_batch,
    size_t bytes_per_patch,
    size_t target_bytes
) {
    size_t chunk_batch = requested_chunk_batch;
    if (bytes_per_patch > 0) {
        const size_t max_by_target = target_bytes / bytes_per_patch;
        if (max_by_target > 0 && chunk_batch > max_by_target) {
            chunk_batch = max_by_target;
        }
    }
    if (chunk_batch == 0) {
        chunk_batch = 1;
    }
    return chunk_batch;
}


static void log_voxel_cuda_stage_summary(
    const char* class_name,
    int batch_size,
    size_t chunk_batch,
    size_t window_voxels,
    size_t host_transfer_bytes,
    size_t device_input_bytes,
    size_t device_scratch_bytes,
    int host_buffer_is_pinned,
    double h2d_ms,
    double device_build_ms,
    double device_finalize_ms,
    double d2h_ms,
    double host_finalize_ms
) {
    std::fprintf(
        stderr,
        "[flash_radiomics][voxel-cuda][%s] batch=%d chunk_batch=%zu window_voxels=%zu "
        "host_transfer_bytes=%zu device_input_bytes=%zu device_scratch_bytes=%zu "
        "host_buffer=%s h2d_ms=%.3f device_build_ms=%.3f device_finalize_ms=%.3f "
        "d2h_ms=%.3f host_finalize_ms=%.3f\n",
        class_name,
        batch_size,
        chunk_batch,
        window_voxels,
        host_transfer_bytes,
        device_input_bytes,
        device_scratch_bytes,
        host_buffer_is_pinned ? "pinned" : "pageable",
        h2d_ms,
        device_build_ms,
        device_finalize_ms,
        d2h_ms,
        host_finalize_ms
    );
}

__device__ inline void write_full_index_feature_row_device(
    const double* full_row,
    const uint8_t* include_features,
    int full_count,
    double* out_row,
    int out_feature_stride
) {
    const double nan_v = NAN;
    for (int i = 0; i < full_count; i++) {
        if (!include_features || include_features[i] != 0) {
            out_row[i] = full_row[i];
        } else {
            out_row[i] = nan_v;
        }
    }
    for (int i = full_count; i < out_feature_stride; i++) {
        out_row[i] = nan_v;
    }
}

__device__ inline void write_compact_feature_row_device(
    const double* full_row,
    const uint8_t* include_features,
    int full_count,
    double* out_row,
    int out_feature_stride
) {
    const double nan_v = NAN;
    int out_idx = 0;
    for (int i = 0; i < full_count; i++) {
        if (include_features && include_features[i] == 0) {
            continue;
        }
        if (out_idx < out_feature_stride) {
            out_row[out_idx] = full_row[i];
        }
        out_idx++;
    }
    for (int i = out_idx; i < out_feature_stride; i++) {
        out_row[i] = nan_v;
    }
}

enum GlcmFeatureIndex {
    GLCM_FEATURE_AUTOCORRELATION = 0,
    GLCM_FEATURE_JOINT_AVERAGE = 1,
    GLCM_FEATURE_CLUSTER_PROMINENCE = 2,
    GLCM_FEATURE_CLUSTER_SHADE = 3,
    GLCM_FEATURE_CLUSTER_TENDENCY = 4,
    GLCM_FEATURE_CONTRAST = 5,
    GLCM_FEATURE_CORRELATION = 6,
    GLCM_FEATURE_DIFFERENCE_AVERAGE = 7,
    GLCM_FEATURE_DIFFERENCE_ENTROPY = 8,
    GLCM_FEATURE_DIFFERENCE_VARIANCE = 9,
    GLCM_FEATURE_JOINT_ENERGY = 10,
    GLCM_FEATURE_JOINT_ENTROPY = 11,
    GLCM_FEATURE_IMC1 = 12,
    GLCM_FEATURE_IMC2 = 13,
    GLCM_FEATURE_IDM = 14,
    GLCM_FEATURE_MCC = 15,
    GLCM_FEATURE_IDMN = 16,
    GLCM_FEATURE_ID = 17,
    GLCM_FEATURE_IDN = 18,
    GLCM_FEATURE_INVERSE_VARIANCE = 19,
    GLCM_FEATURE_MAXIMUM_PROBABILITY = 20,
    GLCM_FEATURE_SUM_AVERAGE = 21,
    GLCM_FEATURE_SUM_ENTROPY = 22,
    GLCM_FEATURE_SUM_SQUARES = 23,
};

__constant__ int GLCM_BATCH_DIRECTIONS_2D[4][2] = {
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1}
};

__constant__ int GLCM_BATCH_DIRECTIONS_3D[13][3] = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 1, 0},
    {1, -1, 0},
    {1, 0, 1},
    {1, 0, -1},
    {0, 1, 1},
    {0, 1, -1},
    {1, 1, 1},
    {1, 1, -1},
    {1, -1, 1},
    {1, -1, -1}
};

// Direction vectors for GLRLM (2D/3D).
__constant__ int GLRLM_BATCH_DIRECTIONS_2D[4][2] = {
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1}
};

__constant__ int GLRLM_BATCH_DIRECTIONS_3D[13][3] = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 1, 0},
    {1, -1, 0},
    {1, 0, 1},
    {1, 0, -1},
    {0, 1, 1},
    {0, 1, -1},
    {1, 1, 1},
    {1, 1, -1},
    {1, -1, 1},
    {1, -1, -1}
};

// Direction vectors for GLDM/NGTDM (2D/3D).
__constant__ int TEXTURE_BATCH_DIRECTIONS_2D[8][2] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};

__constant__ int TEXTURE_BATCH_DIRECTIONS_3D[26][3] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
    {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1},
    {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1},
    {1, 1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, -1, 1},
    {1, -1, 1}, {-1, 1, -1}, {1, -1, -1}, {-1, 1, 1}
};

__global__ void fill_int_kernel(int* data, int value, size_t count) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    for (size_t i = idx; i < count; i += stride) {
        data[i] = value;
    }
}

__global__ void build_glcm_counts_batched_kernel(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int* glcm_counts,
    int batch_size,
    size_t window_voxels,
    int width,
    int height,
    int depth,
    int ndim,
    const int* distances,
    int num_distances,
    int num_directions,
    int num_bins
) {
    const size_t total_voxels = (size_t)batch_size * window_voxels;
    const size_t matrix_size = (size_t)num_bins * (size_t)num_bins;
    const size_t thread_id =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const int slice_stride = width * height;

    for (size_t global_idx = thread_id; global_idx < total_voxels; global_idx += stride) {
        if (mask_windows[global_idx] == 0) {
            continue;
        }
        const int bin_i_raw = discretized_windows[global_idx];
        if (bin_i_raw <= 0 || bin_i_raw > num_bins) {
            continue;
        }
        const int bin_i = bin_i_raw - 1;

        const int patch_idx = (int)(global_idx / window_voxels);
        const int local_idx = (int)(global_idx - (size_t)patch_idx * window_voxels);

        const int x = local_idx % width;
        const int yz = local_idx / width;
        const int y = yz % height;
        const int z = yz / height;

        for (int distance_idx = 0; distance_idx < num_distances; distance_idx++) {
            const int distance = distances[distance_idx];
            if (distance <= 0) {
                continue;
            }

            for (int dir_idx = 0; dir_idx < num_directions; dir_idx++) {
                int dx;
                int dy;
                int dz = 0;
                if (ndim == 2) {
                    dx = GLCM_BATCH_DIRECTIONS_2D[dir_idx][0] * distance;
                    dy = GLCM_BATCH_DIRECTIONS_2D[dir_idx][1] * distance;
                } else {
                    dx = GLCM_BATCH_DIRECTIONS_3D[dir_idx][0] * distance;
                    dy = GLCM_BATCH_DIRECTIONS_3D[dir_idx][1] * distance;
                    dz = GLCM_BATCH_DIRECTIONS_3D[dir_idx][2] * distance;
                }

                const int nx = x + dx;
                const int ny = y + dy;
                const int nz = z + dz;
                if (nx < 0 || nx >= width ||
                    ny < 0 || ny >= height ||
                    nz < 0 || nz >= depth) {
                    continue;
                }

                const int neighbor_local_idx = nz * slice_stride + ny * width + nx;
                const size_t neighbor_global_idx =
                    (size_t)patch_idx * window_voxels + (size_t)neighbor_local_idx;
                if (mask_windows[neighbor_global_idx] == 0) {
                    continue;
                }

                const int bin_j_raw = discretized_windows[neighbor_global_idx];
                if (bin_j_raw <= 0 || bin_j_raw > num_bins) {
                    continue;
                }
                const int bin_j = bin_j_raw - 1;

                const size_t matrix_idx =
                    (((size_t)patch_idx * (size_t)num_distances + (size_t)distance_idx) *
                         (size_t)num_directions) +
                    (size_t)dir_idx;
                const size_t base = matrix_idx * matrix_size;
                atomicAdd(&glcm_counts[base + (size_t)bin_i * (size_t)num_bins + (size_t)bin_j], 1);
                atomicAdd(&glcm_counts[base + (size_t)bin_j * (size_t)num_bins + (size_t)bin_i], 1);
            }
        }
    }
}

__global__ void build_glrlm_counts_batched_kernel(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int* glrlm_counts,
    int* dir_has_multi_element,
    int batch_size,
    size_t window_voxels,
    int width,
    int height,
    int depth,
    int ndim,
    int num_directions,
    int num_bins,
    int max_run_length
) {
    const size_t tasks_per_patch = (size_t)num_directions * window_voxels;
    const size_t total_tasks = (size_t)batch_size * tasks_per_patch;
    const size_t thread_id =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const int slice_stride = width * height;
    const size_t matrix_size = (size_t)num_bins * (size_t)max_run_length;

    for (size_t task_id = thread_id; task_id < total_tasks; task_id += stride) {
        const int patch_idx = (int)(task_id / tasks_per_patch);
        const size_t patch_task = task_id - (size_t)patch_idx * tasks_per_patch;
        const int dir_idx = (int)(patch_task / window_voxels);
        const int local_idx = (int)(patch_task - (size_t)dir_idx * window_voxels);

        const size_t patch_offset = (size_t)patch_idx * window_voxels;

        const int x = local_idx % width;
        const int yz = local_idx / width;
        const int y = yz % height;
        const int z = yz / height;

        int dx = 0;
        int dy = 0;
        int dz = 0;
        if (ndim == 2) {
            dx = GLRLM_BATCH_DIRECTIONS_2D[dir_idx][0];
            dy = GLRLM_BATCH_DIRECTIONS_2D[dir_idx][1];
        } else {
            dx = GLRLM_BATCH_DIRECTIONS_3D[dir_idx][0];
            dy = GLRLM_BATCH_DIRECTIONS_3D[dir_idx][1];
            dz = GLRLM_BATCH_DIRECTIONS_3D[dir_idx][2];
        }

        // Start points are boundary voxels with no geometric predecessor.
        const int px = x - dx;
        const int py = y - dy;
        const int pz = z - dz;
        if (px >= 0 && px < width &&
            py >= 0 && py < height &&
            pz >= 0 && pz < depth) {
            continue;
        }

        int cx = x;
        int cy = y;
        int cz = z;
        int current_gray = -1;
        int run_length = 0;
        int line_valid_elements = 0;

        while (cx >= 0 && cx < width &&
               cy >= 0 && cy < height &&
               cz >= 0 && cz < depth) {
            const int scan_local = cz * slice_stride + cy * width + cx;
            const size_t scan_global = patch_offset + (size_t)scan_local;

            int raw_level = 0;
            if (mask_windows[scan_global] != 0) {
                raw_level = discretized_windows[scan_global];
            }

            if (raw_level > 0 && raw_level <= num_bins) {
                line_valid_elements++;
                if (raw_level == current_gray) {
                    run_length++;
                } else {
                    if (current_gray > 0 && run_length > 0) {
                        int clamped = run_length;
                        if (clamped > max_run_length) {
                            clamped = max_run_length;
                        }
                        const size_t matrix_base =
                            ((size_t)patch_idx * (size_t)num_directions + (size_t)dir_idx) *
                            matrix_size;
                        const size_t entry =
                            matrix_base +
                            (size_t)(current_gray - 1) * (size_t)max_run_length +
                            (size_t)(clamped - 1);
                        atomicAdd(&glrlm_counts[entry], 1);
                    }
                    current_gray = raw_level;
                    run_length = 1;
                }
            } else {
                if (current_gray > 0 && run_length > 0) {
                    int clamped = run_length;
                    if (clamped > max_run_length) {
                        clamped = max_run_length;
                    }
                    const size_t matrix_base =
                        ((size_t)patch_idx * (size_t)num_directions + (size_t)dir_idx) *
                        matrix_size;
                    const size_t entry =
                        matrix_base +
                        (size_t)(current_gray - 1) * (size_t)max_run_length +
                        (size_t)(clamped - 1);
                    atomicAdd(&glrlm_counts[entry], 1);
                }
                current_gray = -1;
                run_length = 0;
            }

            cx += dx;
            cy += dy;
            cz += dz;
        }

        if (current_gray > 0 && run_length > 0) {
            int clamped = run_length;
            if (clamped > max_run_length) {
                clamped = max_run_length;
            }
            const size_t matrix_base =
                ((size_t)patch_idx * (size_t)num_directions + (size_t)dir_idx) * matrix_size;
            const size_t entry =
                matrix_base +
                (size_t)(current_gray - 1) * (size_t)max_run_length +
                (size_t)(clamped - 1);
            atomicAdd(&glrlm_counts[entry], 1);
        }
        if (line_valid_elements > 1 && dir_has_multi_element) {
            const size_t flag_idx =
                (size_t)patch_idx * (size_t)num_directions + (size_t)dir_idx;
            atomicExch(&dir_has_multi_element[flag_idx], 1);
        }
    }
}


#define GLCM_MCC_MAX_ACTIVE_BINS 64

__device__ void jacobi_eigenvalues_symmetric_device(double* matrix, int n, double* eigenvalues) {
    if (!matrix || !eigenvalues || n <= 0) {
        return;
    }

    const int max_iter = n * n * 64;
    const double tol = 1e-12;
    for (int iter = 0; iter < max_iter; iter++) {
        int p = 0;
        int q = 1;
        double max_offdiag = 0.0;
        for (int i = 0; i < n; i++) {
            const double* row = matrix + i * n;
            for (int j = i + 1; j < n; j++) {
                const double magnitude = fabs(row[j]);
                if (magnitude > max_offdiag) {
                    max_offdiag = magnitude;
                    p = i;
                    q = j;
                }
            }
        }

        if (max_offdiag < tol) {
            break;
        }

        const int pn = p * n;
        const int qn = q * n;
        const double app = matrix[pn + p];
        const double aqq = matrix[qn + q];
        const double apq = matrix[pn + q];
        if (fabs(apq) <= tol) {
            continue;
        }

        const double tau = (aqq - app) / (2.0 * apq);
        const double t =
            ((tau >= 0.0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
        const double c = 1.0 / sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < n; k++) {
            if (k == p || k == q) {
                continue;
            }

            const int kn = k * n;
            const double akp = matrix[kn + p];
            const double akq = matrix[kn + q];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;

            matrix[kn + p] = new_kp;
            matrix[pn + k] = new_kp;
            matrix[kn + q] = new_kq;
            matrix[qn + k] = new_kq;
        }

        matrix[pn + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[qn + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[pn + q] = 0.0;
        matrix[qn + p] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = matrix[i * n + i];
    }
}

__device__ int compute_glcm_matrix_features_from_counts_device(
    const int* counts,
    int ng,
    double* px,
    double* py,
    double* px_add_y,
    double* px_sub_y,
    double* out_full
) {
    for (int i = 0; i < FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT; i++) {
        out_full[i] = NAN;
    }
    if (!counts || ng <= 0 || !px || !py || !px_add_y || !px_sub_y || !out_full) {
        return 0;
    }

    const size_t matrix_size = (size_t)ng * (size_t)ng;
    long long total_pairs = 0;
    for (size_t idx = 0; idx < matrix_size; idx++) {
        total_pairs += (long long)counts[idx];
    }
    if (total_pairs <= 0) {
        return 0;
    }

    for (int i = 0; i < ng; i++) {
        px[i] = 0.0;
        py[i] = 0.0;
        px_sub_y[i] = 0.0;
    }
    for (int i = 0; i < (2 * ng - 1); i++) {
        px_add_y[i] = 0.0;
    }

    const double inv_total = 1.0 / (double)total_pairs;
    double autocorrelation = 0.0;
    double contrast = 0.0;
    double joint_energy = 0.0;
    double joint_entropy = 0.0;
    double maximum_probability = 0.0;

    for (int i = 0; i < ng; i++) {
        for (int j = 0; j < ng; j++) {
            const int count = counts[(size_t)i * (size_t)ng + (size_t)j];
            if (count <= 0) {
                continue;
            }
            const double p = (double)count * inv_total;
            const double gray_i = (double)(i + 1);
            const double gray_j = (double)(j + 1);
            const double diff = (double)abs(i - j);

            px[i] += p;
            py[j] += p;
            px_add_y[i + j] += p;
            px_sub_y[abs(i - j)] += p;

            autocorrelation += p * gray_i * gray_j;
            contrast += p * diff * diff;
            joint_energy += p * p;
            joint_entropy -= p * log2(p + VOXEL_BATCH_EPS);
            if (p > maximum_probability) {
                maximum_probability = p;
            }
        }
    }

    double joint_average = 0.0;
    double hx = 0.0;
    double hy = 0.0;
    for (int i = 0; i < ng; i++) {
        const double gray = (double)(i + 1);
        joint_average += gray * px[i];
        if (px[i] > 0.0) {
            hx -= px[i] * log2(px[i] + VOXEL_BATCH_EPS);
        }
        if (py[i] > 0.0) {
            hy -= py[i] * log2(py[i] + VOXEL_BATCH_EPS);
        }
    }

    double variance = 0.0;
    for (int i = 0; i < ng; i++) {
        const double centered = (double)(i + 1) - joint_average;
        variance += px[i] * centered * centered;
    }

    double correlation_num = 0.0;
    double cluster_tendency = 0.0;
    double cluster_shade = 0.0;
    double cluster_prominence = 0.0;
    double sum_squares = 0.0;
    double hxy1 = 0.0;

    for (int i = 0; i < ng; i++) {
        for (int j = 0; j < ng; j++) {
            const int count = counts[(size_t)i * (size_t)ng + (size_t)j];
            if (count <= 0) {
                continue;
            }
            const double p = (double)count * inv_total;
            const double gray_i = (double)(i + 1);
            const double gray_j = (double)(j + 1);
            const double centered_i = gray_i - joint_average;
            const double centered_j = gray_j - joint_average;
            const double cluster = gray_i + gray_j - 2.0 * joint_average;
            const double cluster_sq = cluster * cluster;

            sum_squares += p * centered_i * centered_i;
            correlation_num += p * centered_i * centered_j;
            cluster_tendency += p * cluster_sq;
            cluster_shade += p * cluster_sq * cluster;
            cluster_prominence += p * cluster_sq * cluster_sq;

            const double pxy = px[i] * py[j];
            hxy1 -= p * log2(pxy + VOXEL_BATCH_EPS);
        }
    }

    double hxy2 = 0.0;
    for (int i = 0; i < ng; i++) {
        for (int j = 0; j < ng; j++) {
            const double pxy = px[i] * py[j];
            if (pxy > 0.0) {
                hxy2 -= pxy * log2(pxy + VOXEL_BATCH_EPS);
            }
        }
    }

    double difference_average = 0.0;
    double difference_variance = 0.0;
    double difference_entropy = 0.0;
    double idm = 0.0;
    double idmn = 0.0;
    double id = 0.0;
    double idn = 0.0;
    double inverse_variance = 0.0;
    const double ng_sq = (double)ng * (double)ng;
    for (int k = 0; k < ng; k++) {
        const double p = px_sub_y[k];
        const double k_d = (double)k;
        const double k_sq = k_d * k_d;
        difference_average += k_d * p;
        difference_variance += k_sq * p;
        idm += p / (1.0 + k_sq);
        idmn += p / (1.0 + k_sq / ng_sq);
        id += p / (1.0 + k_d);
        idn += p / (1.0 + k_d / (double)ng);
        if (k > 0) {
            inverse_variance += p / k_sq;
        }
        if (p > 0.0) {
            difference_entropy -= p * log2(p + VOXEL_BATCH_EPS);
        }
    }
    difference_variance -= difference_average * difference_average;

    double sum_average = 0.0;
    double sum_entropy = 0.0;
    for (int k = 0; k < (2 * ng - 1); k++) {
        const double p = px_add_y[k];
        sum_average += (double)(k + 2) * p;
        if (p > 0.0) {
            sum_entropy -= p * log2(p + VOXEL_BATCH_EPS);
        }
    }

    const double max_h = (hx > hy) ? hx : hy;
    const double imc1 = (max_h <= 0.0) ? 0.0 : (joint_entropy - hxy1) / max_h;
    double imc2_arg = 0.0;
    if (hxy2 > joint_entropy) {
        imc2_arg = 1.0 - exp(-2.0 * (hxy2 - joint_entropy));
    }
    if (imc2_arg < 0.0) {
        imc2_arg = 0.0;
    }
    const double imc2 = sqrt(imc2_arg);
    const double correlation = (variance <= 0.0) ? 1.0 : (correlation_num / variance);

    out_full[GLCM_FEATURE_AUTOCORRELATION] = autocorrelation;
    out_full[GLCM_FEATURE_JOINT_AVERAGE] = joint_average;
    out_full[GLCM_FEATURE_CLUSTER_PROMINENCE] = cluster_prominence;
    out_full[GLCM_FEATURE_CLUSTER_SHADE] = cluster_shade;
    out_full[GLCM_FEATURE_CLUSTER_TENDENCY] = cluster_tendency;
    out_full[GLCM_FEATURE_CONTRAST] = contrast;
    out_full[GLCM_FEATURE_CORRELATION] = correlation;
    out_full[GLCM_FEATURE_DIFFERENCE_AVERAGE] = difference_average;
    out_full[GLCM_FEATURE_DIFFERENCE_ENTROPY] = difference_entropy;
    out_full[GLCM_FEATURE_DIFFERENCE_VARIANCE] = difference_variance;
    out_full[GLCM_FEATURE_JOINT_ENERGY] = joint_energy;
    out_full[GLCM_FEATURE_JOINT_ENTROPY] = joint_entropy;
    out_full[GLCM_FEATURE_IMC1] = imc1;
    out_full[GLCM_FEATURE_IMC2] = imc2;
    out_full[GLCM_FEATURE_IDM] = idm;
    // MCC computation: find active bins, build Q matrix, Jacobi eigenvalues
    {
        double mcc = 1.0;
        int active_bins[GLCM_MCC_MAX_ACTIVE_BINS];
        int active_count = 0;
        for (int i = 0; i < ng; i++) {
            if (px[i] > 0.0) {
                if (active_count < GLCM_MCC_MAX_ACTIVE_BINS) {
                    active_bins[active_count] = i;
                }
                active_count++;
            }
        }
        if (active_count > GLCM_MCC_MAX_ACTIVE_BINS) {
            mcc = NAN;
        } else if (active_count >= 2) {
            double q[GLCM_MCC_MAX_ACTIVE_BINS * GLCM_MCC_MAX_ACTIVE_BINS];
            double w[GLCM_MCC_MAX_ACTIVE_BINS];
            const double eps = VOXEL_BATCH_EPS;
            for (int ai = 0; ai < active_count; ai++) {
                for (int aj = ai; aj < active_count; aj++) {
                    double acc = 0.0;
                    for (int ak = 0; ak < active_count; ak++) {
                        double p_iak = (double)counts[active_bins[ai] * ng + active_bins[ak]] * inv_total;
                        double p_jak = (double)counts[active_bins[aj] * ng + active_bins[ak]] * inv_total;
                        double v_ak = px[active_bins[ak]];
                        acc += (p_iak * p_jak) / (v_ak + eps);
                    }
                    double denom = sqrt((px[active_bins[ai]] + eps) * (px[active_bins[aj]] + eps));
                    double value = acc / denom;
                    if (!isfinite(value)) {
                        value = 0.0;
                    }
                    q[ai * active_count + aj] = value;
                    q[aj * active_count + ai] = value;
                }
            }
            jacobi_eigenvalues_symmetric_device(q, active_count, w);
            // Insertion sort eigenvalues ascending
            for (int i = 1; i < active_count; i++) {
                const double key = w[i];
                int j = i - 1;
                while (j >= 0 && w[j] > key) {
                    w[j + 1] = w[j];
                    j--;
                }
                w[j + 1] = key;
            }
            double second_eig = w[active_count - 2];
            if (!isfinite(second_eig)) {
                mcc = NAN;
            } else {
                if (second_eig < 0.0 && second_eig > -1e-12) {
                    second_eig = 0.0;
                }
                mcc = (second_eig <= 0.0) ? 0.0 : sqrt(second_eig);
            }
        }
        out_full[GLCM_FEATURE_MCC] = mcc;
    }
    out_full[GLCM_FEATURE_IDMN] = idmn;
    out_full[GLCM_FEATURE_ID] = id;
    out_full[GLCM_FEATURE_IDN] = idn;
    out_full[GLCM_FEATURE_INVERSE_VARIANCE] = inverse_variance;
    out_full[GLCM_FEATURE_MAXIMUM_PROBABILITY] = maximum_probability;
    out_full[GLCM_FEATURE_SUM_AVERAGE] = sum_average;
    out_full[GLCM_FEATURE_SUM_ENTROPY] = sum_entropy;
    out_full[GLCM_FEATURE_SUM_SQUARES] = sum_squares;
    return 1;
}

__global__ void finalize_glcm_features_batched_kernel(
    const int* glcm_counts,
    int batch_size,
    int matrices_per_patch,
    int num_bins,
    const uint8_t* include_features,
    double* scratch_px,
    double* scratch_py,
    double* scratch_px_add_y,
    double* scratch_px_sub_y,
    double* out_features,
    int out_feature_stride
) {
    const int patch_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (patch_idx >= batch_size || !glcm_counts || !out_features ||
        !scratch_px || !scratch_py || !scratch_px_add_y || !scratch_px_sub_y ||
        matrices_per_patch <= 0 || num_bins <= 0) {
        return;
    }

    const size_t matrix_size = (size_t)num_bins * (size_t)num_bins;
    const int* patch_counts =
        glcm_counts + (size_t)patch_idx * (size_t)matrices_per_patch * matrix_size;
    double* px = scratch_px + (size_t)patch_idx * (size_t)num_bins;
    double* py = scratch_py + (size_t)patch_idx * (size_t)num_bins;
    double* px_add_y = scratch_px_add_y + (size_t)patch_idx * (size_t)(2 * num_bins - 1);
    double* px_sub_y = scratch_px_sub_y + (size_t)patch_idx * (size_t)num_bins;

    double matrix_features[FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT];
    double patch_totals[FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT];
    double patch_output[FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT];
    for (int feature_idx = 0; feature_idx < FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT; feature_idx++) {
        patch_totals[feature_idx] = 0.0;
        patch_output[feature_idx] = NAN;
    }

    int valid_matrices = 0;
    for (int matrix_idx = 0; matrix_idx < matrices_per_patch; matrix_idx++) {
        const int* matrix_counts = patch_counts + (size_t)matrix_idx * matrix_size;
        if (!compute_glcm_matrix_features_from_counts_device(
                matrix_counts,
                num_bins,
                px,
                py,
                px_add_y,
                px_sub_y,
                matrix_features)) {
            continue;
        }
        for (int feature_idx = 0; feature_idx < FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT; feature_idx++) {
            patch_totals[feature_idx] += matrix_features[feature_idx];
        }
        valid_matrices++;
    }

    if (valid_matrices > 0) {
        const double denom = (double)valid_matrices;
        for (int feature_idx = 0; feature_idx < FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT; feature_idx++) {
            patch_output[feature_idx] = patch_totals[feature_idx] / denom;
        }
    }

    write_compact_feature_row_device(
        patch_output,
        include_features,
        FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT,
        out_features + (size_t)patch_idx * (size_t)out_feature_stride,
        out_feature_stride
    );
}

__global__ void build_gldm_matrix_batched_kernel(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int* gldm_matrices,
    int batch_size,
    size_t window_voxels,
    int width,
    int height,
    int depth,
    int ndim,
    const int* distances,
    int num_distances,
    int num_dirs,
    int max_dep,
    int num_bins,
    double gldm_a
) {
    const size_t total_voxels = (size_t)batch_size * window_voxels;
    const size_t thread_id =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const int slice_stride = width * height;
    const int dep_len = max_dep + 1;
    const size_t matrix_size = (size_t)num_bins * (size_t)dep_len;
    const double threshold = (gldm_a < 0.0) ? 0.0 : gldm_a;

    for (size_t global_idx = thread_id; global_idx < total_voxels; global_idx += stride) {
        if (mask_windows[global_idx] == 0) {
            continue;
        }
        const int center_raw = discretized_windows[global_idx];
        if (center_raw <= 0 || center_raw > num_bins) {
            continue;
        }

        const int patch_idx = (int)(global_idx / window_voxels);
        const int local_idx = (int)(global_idx - (size_t)patch_idx * window_voxels);
        const size_t patch_offset = (size_t)patch_idx * window_voxels;

        const int x = local_idx % width;
        const int yz = local_idx / width;
        const int y = yz % height;
        const int z = yz / height;

        int dep = 0;
        for (int d = 0; d < num_distances; d++) {
            const int dist = distances[d];
            if (dist <= 0) {
                continue;
            }
            if (ndim == 2) {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + TEXTURE_BATCH_DIRECTIONS_2D[dir][0] * dist;
                    const int ny = y + TEXTURE_BATCH_DIRECTIONS_2D[dir][1] * dist;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    const int n_local = ny * width + nx;
                    const size_t n_global = patch_offset + (size_t)n_local;
                    if (mask_windows[n_global] == 0) {
                        continue;
                    }
                    const int n_raw = discretized_windows[n_global];
                    if (n_raw <= 0 || n_raw > num_bins) {
                        continue;
                    }
                    const int diff = abs(center_raw - n_raw);
                    if ((double)diff <= threshold) {
                        dep++;
                    }
                }
            } else {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + TEXTURE_BATCH_DIRECTIONS_3D[dir][0] * dist;
                    const int ny = y + TEXTURE_BATCH_DIRECTIONS_3D[dir][1] * dist;
                    const int nz = z + TEXTURE_BATCH_DIRECTIONS_3D[dir][2] * dist;
                    if (nx < 0 || nx >= width ||
                        ny < 0 || ny >= height ||
                        nz < 0 || nz >= depth) {
                        continue;
                    }
                    const int n_local = nz * slice_stride + ny * width + nx;
                    const size_t n_global = patch_offset + (size_t)n_local;
                    if (mask_windows[n_global] == 0) {
                        continue;
                    }
                    const int n_raw = discretized_windows[n_global];
                    if (n_raw <= 0 || n_raw > num_bins) {
                        continue;
                    }
                    const int diff = abs(center_raw - n_raw);
                    if ((double)diff <= threshold) {
                        dep++;
                    }
                }
            }
        }

        if (dep < 0) dep = 0;
        if (dep > max_dep) dep = max_dep;
        const size_t matrix_base = (size_t)patch_idx * matrix_size;
        const size_t entry =
            matrix_base +
            (size_t)(center_raw - 1) * (size_t)dep_len +
            (size_t)dep;
        atomicAdd(&gldm_matrices[entry], 1);
    }
}

__global__ void build_ngtdm_vectors_batched_kernel(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    double* n_i,
    double* s_i,
    int batch_size,
    size_t window_voxels,
    int width,
    int height,
    int depth,
    int ndim,
    const int* distances,
    int num_distances,
    int num_dirs,
    int num_bins
) {
    const size_t total_voxels = (size_t)batch_size * window_voxels;
    const size_t thread_id =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const int slice_stride = width * height;

    for (size_t global_idx = thread_id; global_idx < total_voxels; global_idx += stride) {
        if (mask_windows[global_idx] == 0) {
            continue;
        }
        const int center_raw = discretized_windows[global_idx];
        if (center_raw <= 0 || center_raw > num_bins) {
            continue;
        }

        const int patch_idx = (int)(global_idx / window_voxels);
        const int local_idx = (int)(global_idx - (size_t)patch_idx * window_voxels);
        const size_t patch_offset = (size_t)patch_idx * window_voxels;
        const size_t vector_offset = (size_t)patch_idx * (size_t)num_bins;

        const int x = local_idx % width;
        const int yz = local_idx / width;
        const int y = yz % height;
        const int z = yz / height;

        double sum_neighbors = 0.0;
        int count_neighbors = 0;
        for (int d = 0; d < num_distances; d++) {
            const int dist = distances[d];
            if (dist <= 0) {
                continue;
            }
            if (ndim == 2) {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + TEXTURE_BATCH_DIRECTIONS_2D[dir][0] * dist;
                    const int ny = y + TEXTURE_BATCH_DIRECTIONS_2D[dir][1] * dist;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    const int n_local = ny * width + nx;
                    const size_t n_global = patch_offset + (size_t)n_local;
                    if (mask_windows[n_global] == 0) {
                        continue;
                    }
                    const int n_raw = discretized_windows[n_global];
                    if (n_raw <= 0 || n_raw > num_bins) {
                        continue;
                    }
                    sum_neighbors += (double)n_raw;
                    count_neighbors++;
                }
            } else {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + TEXTURE_BATCH_DIRECTIONS_3D[dir][0] * dist;
                    const int ny = y + TEXTURE_BATCH_DIRECTIONS_3D[dir][1] * dist;
                    const int nz = z + TEXTURE_BATCH_DIRECTIONS_3D[dir][2] * dist;
                    if (nx < 0 || nx >= width ||
                        ny < 0 || ny >= height ||
                        nz < 0 || nz >= depth) {
                        continue;
                    }
                    const int n_local = nz * slice_stride + ny * width + nx;
                    const size_t n_global = patch_offset + (size_t)n_local;
                    if (mask_windows[n_global] == 0) {
                        continue;
                    }
                    const int n_raw = discretized_windows[n_global];
                    if (n_raw <= 0 || n_raw > num_bins) {
                        continue;
                    }
                    sum_neighbors += (double)n_raw;
                    count_neighbors++;
                }
            }
        }

        if (count_neighbors <= 0) {
            continue;
        }

        const double avg = sum_neighbors / (double)count_neighbors;
        const double diff = fabs((double)center_raw - avg);
        const int g_idx = center_raw - 1;
        atomicAdd(&n_i[vector_offset + (size_t)g_idx], 1.0);
        atomicAdd(&s_i[vector_offset + (size_t)g_idx], diff);
    }
}

__global__ void build_ngtdm_vectors_batched_patch_kernel(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    double* n_i,
    double* s_i,
    int batch_size,
    size_t window_voxels,
    int width,
    int height,
    int depth,
    int ndim,
    const int* distances,
    int num_distances,
    int num_dirs,
    int num_bins
) {
    const int patch_idx = (int)blockIdx.x;
    if (patch_idx >= batch_size) {
        return;
    }

    extern __shared__ double shared_vectors[];
    double* shared_n_i = shared_vectors;
    double* shared_s_i = shared_vectors + num_bins;

    for (int bin_idx = (int)threadIdx.x; bin_idx < num_bins; bin_idx += (int)blockDim.x) {
        shared_n_i[bin_idx] = 0.0;
        shared_s_i[bin_idx] = 0.0;
    }
    __syncthreads();

    const int slice_stride = width * height;
    const size_t patch_offset = (size_t)patch_idx * window_voxels;
    for (size_t local_idx = (size_t)threadIdx.x;
         local_idx < window_voxels;
         local_idx += (size_t)blockDim.x) {
        const size_t global_idx = patch_offset + local_idx;
        if (mask_windows[global_idx] == 0) {
            continue;
        }

        const int center_raw = discretized_windows[global_idx];
        if (center_raw <= 0 || center_raw > num_bins) {
            continue;
        }

        const int local_pos = (int)local_idx;
        const int x = local_pos % width;
        const int yz = local_pos / width;
        const int y = yz % height;
        const int z = yz / height;

        double sum_neighbors = 0.0;
        int count_neighbors = 0;
        for (int d = 0; d < num_distances; d++) {
            const int dist = distances[d];
            if (dist <= 0) {
                continue;
            }

            if (ndim == 2) {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + TEXTURE_BATCH_DIRECTIONS_2D[dir][0] * dist;
                    const int ny = y + TEXTURE_BATCH_DIRECTIONS_2D[dir][1] * dist;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    const int n_local = ny * width + nx;
                    const size_t n_global = patch_offset + (size_t)n_local;
                    if (mask_windows[n_global] == 0) {
                        continue;
                    }
                    const int n_raw = discretized_windows[n_global];
                    if (n_raw <= 0 || n_raw > num_bins) {
                        continue;
                    }
                    sum_neighbors += (double)n_raw;
                    count_neighbors++;
                }
            } else {
                for (int dir = 0; dir < num_dirs; dir++) {
                    const int nx = x + TEXTURE_BATCH_DIRECTIONS_3D[dir][0] * dist;
                    const int ny = y + TEXTURE_BATCH_DIRECTIONS_3D[dir][1] * dist;
                    const int nz = z + TEXTURE_BATCH_DIRECTIONS_3D[dir][2] * dist;
                    if (nx < 0 || nx >= width ||
                        ny < 0 || ny >= height ||
                        nz < 0 || nz >= depth) {
                        continue;
                    }
                    const int n_local = nz * slice_stride + ny * width + nx;
                    const size_t n_global = patch_offset + (size_t)n_local;
                    if (mask_windows[n_global] == 0) {
                        continue;
                    }
                    const int n_raw = discretized_windows[n_global];
                    if (n_raw <= 0 || n_raw > num_bins) {
                        continue;
                    }
                    sum_neighbors += (double)n_raw;
                    count_neighbors++;
                }
            }
        }

        if (count_neighbors <= 0) {
            continue;
        }

        const double avg = sum_neighbors / (double)count_neighbors;
        const double diff = fabs((double)center_raw - avg);
        const int g_idx = center_raw - 1;
        atomicAdd(&shared_n_i[g_idx], 1.0);
        atomicAdd(&shared_s_i[g_idx], diff);
    }
    __syncthreads();

    double* patch_n_i = n_i + (size_t)patch_idx * (size_t)num_bins;
    double* patch_s_i = s_i + (size_t)patch_idx * (size_t)num_bins;
    for (int bin_idx = (int)threadIdx.x; bin_idx < num_bins; bin_idx += (int)blockDim.x) {
        patch_n_i[bin_idx] = shared_n_i[bin_idx];
        patch_s_i[bin_idx] = shared_s_i[bin_idx];
    }
}

__global__ void finalize_gldm_features_batched_kernel(
    const int* matrix_counts,
    int batch_size,
    int num_bins,
    int dep_len,
    const uint8_t* include_features,
    double* out_features,
    int out_feature_stride
) {
    const int patch_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (patch_idx >= batch_size || !matrix_counts || !out_features || num_bins <= 0 || dep_len <= 0) {
        return;
    }

    const size_t matrix_size = (size_t)num_bins * (size_t)dep_len;
    const int* patch_matrix = matrix_counts + (size_t)patch_idx * matrix_size;
    double patch_output[FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT];
    for (int i = 0; i < FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT; i++) {
        patch_output[i] = NAN;
    }

    double Nz = 0.0;
    for (int i = 0; i < num_bins; i++) {
        for (int j = 0; j < dep_len; j++) {
            Nz += (double)patch_matrix[(size_t)i * (size_t)dep_len + (size_t)j];
        }
    }
    if (Nz <= 0.0) {
        write_full_index_feature_row_device(
            patch_output,
            include_features,
            FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT,
            out_features + (size_t)patch_idx * (size_t)out_feature_stride,
            out_feature_stride
        );
        return;
    }

    double mean_g = 0.0;
    for (int i = 0; i < num_bins; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < dep_len; j++) {
            row_sum += (double)patch_matrix[(size_t)i * (size_t)dep_len + (size_t)j];
        }
        mean_g += (double)(i + 1) * row_sum;
    }
    mean_g /= Nz;

    double mean_d = 0.0;
    for (int j = 0; j < dep_len; j++) {
        double col_sum = 0.0;
        for (int i = 0; i < num_bins; i++) {
            col_sum += (double)patch_matrix[(size_t)i * (size_t)dep_len + (size_t)j];
        }
        mean_d += (double)(j + 1) * col_sum;
    }
    mean_d /= Nz;

    double sde = 0.0;
    double lde = 0.0;
    double gln = 0.0;
    double dn = 0.0;
    double glv = 0.0;
    double dv = 0.0;
    double de = 0.0;
    double lgle = 0.0;
    double hgle = 0.0;
    double sdlgle = 0.0;
    double sdhgle = 0.0;
    double ldlgle = 0.0;
    double ldhgle = 0.0;

    for (int i = 0; i < num_bins; i++) {
        const double i_val = (double)(i + 1);
        const double i_sq = i_val * i_val;
        double row_sum = 0.0;
        for (int j = 0; j < dep_len; j++) {
            row_sum += (double)patch_matrix[(size_t)i * (size_t)dep_len + (size_t)j];
        }
        const double p_row = row_sum / Nz;
        gln += row_sum * row_sum;
        glv += p_row * (i_val - mean_g) * (i_val - mean_g);
        lgle += p_row / i_sq;
        hgle += p_row * i_sq;
    }

    for (int j = 0; j < dep_len; j++) {
        const double j_val = (double)(j + 1);
        const double j_sq = j_val * j_val;
        double col_sum = 0.0;
        for (int i = 0; i < num_bins; i++) {
            col_sum += (double)patch_matrix[(size_t)i * (size_t)dep_len + (size_t)j];
        }
        const double p_col = col_sum / Nz;
        dn += col_sum * col_sum;
        sde += p_col / j_sq;
        lde += p_col * j_sq;
        dv += p_col * (j_val - mean_d) * (j_val - mean_d);
    }

    for (int i = 0; i < num_bins; i++) {
        const double i_val = (double)(i + 1);
        const double i_sq = i_val * i_val;
        for (int j = 0; j < dep_len; j++) {
            const double v = (double)patch_matrix[(size_t)i * (size_t)dep_len + (size_t)j];
            if (v <= 0.0) {
                continue;
            }
            const double j_val = (double)(j + 1);
            const double j_sq = j_val * j_val;
            const double p = v / Nz;
            de -= p * log2(p + VOXEL_BATCH_EPS);
            sdlgle += p / (i_sq * j_sq);
            sdhgle += p * i_sq / j_sq;
            ldlgle += p * j_sq / i_sq;
            ldhgle += p * i_sq * j_sq;
        }
    }

    patch_output[0] = sde;
    patch_output[1] = lde;
    patch_output[2] = gln / Nz;
    patch_output[3] = dn / Nz;
    patch_output[4] = dn / (Nz * Nz);
    patch_output[5] = glv;
    patch_output[6] = dv;
    patch_output[7] = de;
    patch_output[8] = lgle;
    patch_output[9] = hgle;
    patch_output[10] = sdlgle;
    patch_output[11] = sdhgle;
    patch_output[12] = ldlgle;
    patch_output[13] = ldhgle;

    write_full_index_feature_row_device(
        patch_output,
        include_features,
        FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT,
        out_features + (size_t)patch_idx * (size_t)out_feature_stride,
        out_feature_stride
    );
}

__global__ void finalize_ngtdm_features_batched_kernel(
    const double* n_i,
    const double* s_i,
    int batch_size,
    int num_bins,
    const uint8_t* include_features,
    double* out_features,
    int out_feature_stride
) {
    const int patch_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (patch_idx >= batch_size || !n_i || !s_i || !out_features || num_bins <= 0) {
        return;
    }

    const double* patch_n_i = n_i + (size_t)patch_idx * (size_t)num_bins;
    const double* patch_s_i = s_i + (size_t)patch_idx * (size_t)num_bins;
    double patch_output[FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT];
    for (int i = 0; i < FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT; i++) {
        patch_output[i] = NAN;
    }

    double Nvp = 0.0;
    int Ngp = 0;
    for (int i = 0; i < num_bins; i++) {
        const double count = patch_n_i[i];
        Nvp += count;
        if (count > 0.0) {
            Ngp++;
        }
    }
    if (Nvp <= 0.0) {
        write_full_index_feature_row_device(
            patch_output,
            include_features,
            FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT,
            out_features + (size_t)patch_idx * (size_t)out_feature_stride,
            out_feature_stride
        );
        return;
    }

    double sum_p_s = 0.0;
    double sum_s_all = 0.0;
    for (int i = 0; i < num_bins; i++) {
        if (patch_n_i[i] <= 0.0) {
            continue;
        }
        const double p = patch_n_i[i] / Nvp;
        sum_p_s += p * patch_s_i[i];
        sum_s_all += patch_s_i[i];
    }

    patch_output[0] = (sum_p_s != 0.0) ? (1.0 / sum_p_s) : 1e6;

    if (Ngp > 1) {
        double sum_ij = 0.0;
        for (int i = 0; i < num_bins; i++) {
            if (patch_n_i[i] <= 0.0) {
                continue;
            }
            const double p_ii = patch_n_i[i] / Nvp;
            const double i_val = (double)(i + 1);
            for (int j = i + 1; j < num_bins; j++) {
                if (patch_n_i[j] <= 0.0) {
                    continue;
                }
                const double p_jj = patch_n_i[j] / Nvp;
                const double j_val = (double)(j + 1);
                const double diff = i_val - j_val;
                sum_ij += 2.0 * p_ii * p_jj * diff * diff;
            }
        }
        patch_output[1] =
            (sum_ij / ((double)Ngp * (double)(Ngp - 1))) * (sum_s_all / Nvp);
    }

    double busyness = 0.0;
    patch_output[2] = 0.0;
    {
        double denom_busy = 0.0;
        for (int i = 0; i < num_bins; i++) {
            if (patch_n_i[i] <= 0.0) {
                continue;
            }
            const double i_pi = (double)(i + 1) * (patch_n_i[i] / Nvp);
            for (int j = i + 1; j < num_bins; j++) {
                if (patch_n_i[j] <= 0.0) {
                    continue;
                }
                const double j_pi = (double)(j + 1) * (patch_n_i[j] / Nvp);
                denom_busy += 2.0 * fabs(i_pi - j_pi);
            }
        }
        if (denom_busy != 0.0) {
            busyness = sum_p_s / denom_busy;
        }
    }
    patch_output[2] = busyness;

    double complexity = 0.0;
    for (int i = 0; i < num_bins; i++) {
        if (patch_n_i[i] <= 0.0) {
            continue;
        }
        const double p_ii = patch_n_i[i] / Nvp;
        const double s_ii = patch_s_i[i];
        const double i_val = (double)(i + 1);
        for (int j = i + 1; j < num_bins; j++) {
            if (patch_n_i[j] <= 0.0) {
                continue;
            }
            const double p_jj = patch_n_i[j] / Nvp;
            const double s_jj = patch_s_i[j];
            const double j_val = (double)(j + 1);
            const double denom = p_ii + p_jj;
            if (denom > 0.0) {
                complexity += 2.0 * (fabs(i_val - j_val) / denom) *
                              (p_ii * s_ii + p_jj * s_jj);
            }
        }
    }
    patch_output[3] = complexity / Nvp;

    if (sum_s_all != 0.0) {
        double sum_strength = 0.0;
        for (int i = 0; i < num_bins; i++) {
            if (patch_n_i[i] <= 0.0) {
                continue;
            }
            const double p_ii = patch_n_i[i] / Nvp;
            const double i_val = (double)(i + 1);
            for (int j = i + 1; j < num_bins; j++) {
                if (patch_n_i[j] <= 0.0) {
                    continue;
                }
                const double p_jj = patch_n_i[j] / Nvp;
                const double j_val = (double)(j + 1);
                const double diff = i_val - j_val;
                sum_strength += 2.0 * (p_ii + p_jj) * diff * diff;
            }
        }
        patch_output[4] = sum_strength / sum_s_all;
    }

    write_full_index_feature_row_device(
        patch_output,
        include_features,
        FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT,
        out_features + (size_t)patch_idx * (size_t)out_feature_stride,
        out_feature_stride
    );
}

__global__ void ccl_init_batched_kernel(
    const uint8_t* mask_windows,
    int* labels,
    size_t total_voxels
) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    for (size_t i = idx; i < total_voxels; i += stride) {
        if (mask_windows[i] != 0) {
            labels[i] = (int)i;
        } else {
            labels[i] = -1;
        }
    }
}

__global__ void ccl_merge_batched_kernel(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int* labels,
    int* changed,
    size_t total_voxels,
    size_t window_voxels,
    int width,
    int height,
    int depth,
    int ndim
) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const int slice_stride = width * height;

    for (size_t global_idx = idx; global_idx < total_voxels; global_idx += stride) {
        if (mask_windows[global_idx] == 0) {
            continue;
        }
        const int my_gray = discretized_windows[global_idx];
        if (my_gray <= 0) {
            continue;
        }
        const int my_label = labels[global_idx];
        if (my_label < 0) {
            continue;
        }

        const int patch_idx = (int)(global_idx / window_voxels);
        const int local_idx = (int)(global_idx - (size_t)patch_idx * window_voxels);
        const size_t patch_offset = (size_t)patch_idx * window_voxels;

        const int x = local_idx % width;
        const int yz = local_idx / width;
        const int y = yz % height;
        const int z = yz / height;

        int min_label = my_label;
        if (ndim == 2) {
            const int dx4[4] = {1, 0, -1, 0};
            const int dy4[4] = {0, 1, 0, -1};
            for (int d = 0; d < 4; d++) {
                const int nx = x + dx4[d];
                const int ny = y + dy4[d];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }
                const int n_local = ny * width + nx;
                const size_t n_global = patch_offset + (size_t)n_local;
                if (mask_windows[n_global] == 0) {
                    continue;
                }
                if (discretized_windows[n_global] != my_gray) {
                    continue;
                }
                const int n_label = labels[n_global];
                if (n_label >= 0 && n_label < min_label) {
                    min_label = n_label;
                }
            }
        } else {
            for (int dz = -1; dz <= 1; dz++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx3 = -1; dx3 <= 1; dx3++) {
                        if (dx3 == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        const int nx = x + dx3;
                        const int ny = y + dy;
                        const int nz = z + dz;
                        if (nx < 0 || nx >= width ||
                            ny < 0 || ny >= height ||
                            nz < 0 || nz >= depth) {
                            continue;
                        }
                        const int n_local = nz * slice_stride + ny * width + nx;
                        const size_t n_global = patch_offset + (size_t)n_local;
                        if (mask_windows[n_global] == 0) {
                            continue;
                        }
                        if (discretized_windows[n_global] != my_gray) {
                            continue;
                        }
                        const int n_label = labels[n_global];
                        if (n_label >= 0 && n_label < min_label) {
                            min_label = n_label;
                        }
                    }
                }
            }
        }

        if (min_label < my_label) {
            atomicMin(&labels[global_idx], min_label);
            *changed = 1;
        }
    }
}

__global__ void ccl_flatten_batched_kernel(int* labels, size_t total_voxels) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    for (size_t i = idx; i < total_voxels; i += stride) {
        const int label = labels[i];
        if (label < 0) {
            continue;
        }
        int root = label;
        while (root >= 0 && root != labels[root]) {
            root = labels[root];
        }
        int cursor = label;
        while (cursor >= 0 && cursor != root) {
            const int parent = labels[cursor];
            if (parent < 0) {
                break;
            }
            labels[cursor] = root;
            cursor = parent;
        }
        labels[i] = root;
    }
}

__global__ void compute_zone_sizes_batched_kernel(
    const int* labels,
    const uint8_t* mask_windows,
    int* zone_sizes,
    size_t total_voxels
) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    for (size_t i = idx; i < total_voxels; i += stride) {
        if (mask_windows[i] == 0) {
            continue;
        }
        const int label = labels[i];
        if (label >= 0) {
            atomicAdd(&zone_sizes[label], 1);
        }
    }
}

__global__ void mark_zone_representatives_batched_kernel(
    const int* labels,
    const uint8_t* mask_windows,
    int* zone_representatives,
    size_t total_voxels
) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    for (size_t i = idx; i < total_voxels; i += stride) {
        if (mask_windows[i] == 0) {
            continue;
        }
        const int label = labels[i];
        if (label >= 0) {
            atomicMin(&zone_representatives[label], (int)i);
        }
    }
}

__global__ void build_glszm_counts_batched_kernel(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    const int* labels,
    const int* zone_sizes,
    const int* zone_representatives,
    int* glszm_counts,
    size_t total_voxels,
    size_t window_voxels,
    int num_bins,
    int max_zone_size
) {
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
    const size_t matrix_size = (size_t)num_bins * (size_t)max_zone_size;

    for (size_t i = idx; i < total_voxels; i += stride) {
        if (mask_windows[i] == 0) {
            continue;
        }
        const int label = labels[i];
        if (label < 0) {
            continue;
        }
        if (zone_representatives[label] != (int)i) {
            continue;
        }

        const int raw = discretized_windows[i];
        if (raw <= 0 || raw > num_bins) {
            continue;
        }
        const int zone_size = zone_sizes[label];
        if (zone_size <= 0 || zone_size > max_zone_size) {
            continue;
        }

        const int patch_idx = (int)(i / window_voxels);
        const size_t base = (size_t)patch_idx * matrix_size;
        const size_t entry =
            base + (size_t)(raw - 1) * (size_t)max_zone_size + (size_t)(zone_size - 1);
        atomicAdd(&glszm_counts[entry], 1);
    }
}

__global__ void finalize_glrlm_features_batched_kernel(
    const int* counts,
    const int* dir_has_multi_element,
    int batch_size,
    int num_directions,
    int num_bins,
    int max_run_length,
    const uint8_t* include_features,
    double* out_features,
    int out_feature_stride
) {
    const int patch_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (patch_idx >= batch_size || !counts || !out_features ||
        num_directions <= 0 || num_bins <= 0 || max_run_length <= 0) {
        return;
    }

    const size_t matrix_size = (size_t)num_bins * (size_t)max_run_length;
    const size_t counts_per_patch = (size_t)num_directions * matrix_size;
    const int* patch_counts = counts + (size_t)patch_idx * counts_per_patch;
    double patch_output[FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT];
    for (int i = 0; i < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT; i++) {
        patch_output[i] = NAN;
    }

    double patch_totals[FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT];
    for (int i = 0; i < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT; i++) {
        patch_totals[i] = 0.0;
    }

    int valid_dirs = 0;
    for (int dir = 0; dir < num_directions; dir++) {
        if (dir_has_multi_element) {
            const size_t flag_idx =
                (size_t)patch_idx * (size_t)num_directions + (size_t)dir;
            if (dir_has_multi_element[flag_idx] == 0) {
                continue;
            }
        }
        const int* dir_counts = patch_counts + (size_t)dir * matrix_size;
        double Nr = 0.0;
        for (int i = 0; i < num_bins; i++) {
            for (int j = 0; j < max_run_length; j++) {
                const double c = (double)dir_counts[(size_t)i * (size_t)max_run_length + (size_t)j];
                if (c > 0.0) {
                    Nr += c;
                }
            }
        }
        if (Nr <= 0.0) {
            continue;
        }

        double sre = 0.0;
        double lre = 0.0;
        double gln_sum = 0.0;
        double rln_sum = 0.0;
        double Np = 0.0;
        double mean_i = 0.0;
        double mean_j = 0.0;
        double lglre = 0.0;
        double hglre = 0.0;
        double srlgle = 0.0;
        double srhgle = 0.0;
        double lrlgle = 0.0;
        double lrhgle = 0.0;
        double re = 0.0;

        for (int j = 0; j < max_run_length; j++) {
            const double j_val = (double)(j + 1);
            const double j_sq = j_val * j_val;
            double col_sum = 0.0;
            for (int i = 0; i < num_bins; i++) {
                col_sum += (double)dir_counts[(size_t)i * (size_t)max_run_length + (size_t)j];
            }
            sre += col_sum / j_sq;
            lre += col_sum * j_sq;
            rln_sum += col_sum * col_sum;
            Np += col_sum * j_val;
            mean_j += col_sum * j_val;
        }
        mean_j /= Nr;

        for (int i = 0; i < num_bins; i++) {
            const double i_val = (double)(i + 1);
            const double i_sq = i_val * i_val;
            double row_sum = 0.0;
            for (int j = 0; j < max_run_length; j++) {
                row_sum += (double)dir_counts[(size_t)i * (size_t)max_run_length + (size_t)j];
            }
            gln_sum += row_sum * row_sum;
            mean_i += row_sum * i_val;
            lglre += row_sum / i_sq;
            hglre += row_sum * i_sq;
        }
        mean_i /= Nr;

        double glv = 0.0;
        for (int i = 0; i < num_bins; i++) {
            const double i_val = (double)(i + 1);
            const double diff = i_val - mean_i;
            double row_sum = 0.0;
            for (int j = 0; j < max_run_length; j++) {
                row_sum += (double)dir_counts[(size_t)i * (size_t)max_run_length + (size_t)j];
            }
            glv += row_sum * diff * diff;
        }
        glv /= Nr;

        double rv = 0.0;
        for (int j = 0; j < max_run_length; j++) {
            const double j_val = (double)(j + 1);
            const double diff = j_val - mean_j;
            double col_sum = 0.0;
            for (int i = 0; i < num_bins; i++) {
                col_sum += (double)dir_counts[(size_t)i * (size_t)max_run_length + (size_t)j];
            }
            rv += col_sum * diff * diff;
        }
        rv /= Nr;

        for (int i = 0; i < num_bins; i++) {
            const double i_val = (double)(i + 1);
            const double i_sq = i_val * i_val;
            for (int j = 0; j < max_run_length; j++) {
                const double c = (double)dir_counts[(size_t)i * (size_t)max_run_length + (size_t)j];
                if (c <= 0.0) {
                    continue;
                }
                const double j_val = (double)(j + 1);
                const double j_sq = j_val * j_val;
                const double p = c / Nr;
                re -= p * log2(p + VOXEL_BATCH_EPS);
                srlgle += c / (i_sq * j_sq);
                srhgle += c * i_sq / j_sq;
                lrlgle += c * j_sq / i_sq;
                lrhgle += c * i_sq * j_sq;
            }
        }

        patch_totals[0] += sre / Nr;
        patch_totals[1] += lre / Nr;
        patch_totals[2] += gln_sum / Nr;
        patch_totals[3] += gln_sum / (Nr * Nr);
        patch_totals[4] += rln_sum / Nr;
        patch_totals[5] += rln_sum / (Nr * Nr);
        patch_totals[6] += (Np > 0.0) ? (Nr / Np) : 0.0;
        patch_totals[7] += glv;
        patch_totals[8] += rv;
        patch_totals[9] += re;
        patch_totals[10] += lglre / Nr;
        patch_totals[11] += hglre / Nr;
        patch_totals[12] += srlgle / Nr;
        patch_totals[13] += srhgle / Nr;
        patch_totals[14] += lrlgle / Nr;
        patch_totals[15] += lrhgle / Nr;
        valid_dirs++;
    }

    if (valid_dirs > 0) {
        const double denom = (double)valid_dirs;
        for (int i = 0; i < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT; i++) {
            patch_output[i] = patch_totals[i] / denom;
        }
    }

    write_full_index_feature_row_device(
        patch_output,
        include_features,
        FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT,
        out_features + (size_t)patch_idx * (size_t)out_feature_stride,
        out_feature_stride
    );
}

__global__ void finalize_glszm_features_batched_kernel(
    const int* matrix_counts,
    const uint8_t* mask_windows,
    int batch_size,
    size_t window_voxels,
    int num_bins,
    int max_zone_size,
    const uint8_t* include_features,
    double* out_features,
    int out_feature_stride
) {
    const int patch_idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (patch_idx >= batch_size || !matrix_counts || !mask_windows || !out_features ||
        num_bins <= 0 || max_zone_size <= 0) {
        return;
    }

    const size_t matrix_size = (size_t)num_bins * (size_t)max_zone_size;
    const int* patch_counts = matrix_counts + (size_t)patch_idx * matrix_size;
    const uint8_t* patch_mask = mask_windows + (size_t)patch_idx * window_voxels;
    double patch_output[FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT];
    for (int i = 0; i < FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT; i++) {
        patch_output[i] = 0.0;
    }

    int roi_voxels = 0;
    for (size_t i = 0; i < window_voxels; i++) {
        if (patch_mask[i] != 0) {
            roi_voxels++;
        }
    }

    double total = 0.0;
    double sae = 0.0;
    double lae = 0.0;
    double lglze = 0.0;
    double hglze = 0.0;
    double salgle = 0.0;
    double sahgle = 0.0;
    double lalgle = 0.0;
    double lahgle = 0.0;
    double gl_weighted = 0.0;
    double gl2_weighted = 0.0;
    double sz_weighted = 0.0;
    double sz2_weighted = 0.0;
    double p_log_p = 0.0;

    for (int i = 0; i < num_bins; i++) {
        const double gl = (double)(i + 1);
        const double gl_sq = gl * gl;
        for (int j = 0; j < max_zone_size; j++) {
            const int count = patch_counts[(size_t)i * (size_t)max_zone_size + (size_t)j];
            if (count <= 0) {
                continue;
            }
            const double p_ij = (double)count;
            const double sz = (double)(j + 1);
            const double sz_sq = sz * sz;

            total += p_ij;
            sae += p_ij / sz_sq;
            lae += p_ij * sz_sq;
            lglze += p_ij / gl_sq;
            hglze += p_ij * gl_sq;
            salgle += p_ij / (gl_sq * sz_sq);
            sahgle += p_ij * gl_sq / sz_sq;
            lalgle += p_ij * sz_sq / gl_sq;
            lahgle += p_ij * gl_sq * sz_sq;
            gl_weighted += gl * p_ij;
            gl2_weighted += gl_sq * p_ij;
            sz_weighted += sz * p_ij;
            sz2_weighted += sz_sq * p_ij;
            p_log_p += p_ij * log2(p_ij);
        }
    }

    patch_output[12] = (roi_voxels > 0) ? (total / (double)roi_voxels) : 0.0;
    if (total > VOXEL_BATCH_EPS) {
        double gln_numerator = 0.0;
        for (int i = 0; i < num_bins; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < max_zone_size; j++) {
                row_sum += (double)patch_counts[(size_t)i * (size_t)max_zone_size + (size_t)j];
            }
            gln_numerator += row_sum * row_sum;
        }

        double szn_numerator = 0.0;
        for (int j = 0; j < max_zone_size; j++) {
            double col_sum = 0.0;
            for (int i = 0; i < num_bins; i++) {
                col_sum += (double)patch_counts[(size_t)i * (size_t)max_zone_size + (size_t)j];
            }
            szn_numerator += col_sum * col_sum;
        }

        patch_output[0] = sae / total;
        patch_output[1] = lae / total;
        patch_output[2] = lglze / total;
        patch_output[3] = hglze / total;
        patch_output[4] = salgle / total;
        patch_output[5] = sahgle / total;
        patch_output[6] = lalgle / total;
        patch_output[7] = lahgle / total;
        patch_output[8] = gln_numerator / total;
        patch_output[9] = gln_numerator / (total * total);
        patch_output[10] = szn_numerator / total;
        patch_output[11] = szn_numerator / (total * total);

        const double gl_mean = gl_weighted / total;
        const double gl_second = gl2_weighted / total;
        double gl_var = gl_second - gl_mean * gl_mean;
        if (gl_var < 0.0 && gl_var > -1e-12) {
            gl_var = 0.0;
        }
        patch_output[13] = gl_var;

        const double sz_mean = sz_weighted / total;
        const double sz_second = sz2_weighted / total;
        double sz_var = sz_second - sz_mean * sz_mean;
        if (sz_var < 0.0 && sz_var > -1e-12) {
            sz_var = 0.0;
        }
        patch_output[14] = sz_var;
        patch_output[15] = log2(total) - (p_log_p / total);
    }

    write_compact_feature_row_device(
        patch_output,
        include_features,
        FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT,
        out_features + (size_t)patch_idx * (size_t)out_feature_stride,
        out_feature_stride
    );
}

}  // namespace

extern "C" int flash_radiomics_glcm_cuda_discretized_voxel_batch(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    int* distances,
    int num_distances,
    const uint8_t* include_features,
    int ng,
    double* out_features,
    int out_feature_stride
) {
    WindowSpec spec;
    if (!discretized_windows || !mask_windows || !distances || !out_features ||
        batch_size <= 0 || num_distances <= 0 || ng <= 0 ||
        !init_window_spec(ndim, window_dims, &spec)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    const int enabled_count =
        count_enabled_features(include_features, FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT);
    if (enabled_count <= 0 || out_feature_stride < enabled_count) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    CudaContext* ctx = get_cached_voxel_cuda_context();
    if (!ctx) {
        return FLASH_RADIOMICS_ERROR_BACKEND_UNAVAILABLE;
    }

    int status = FLASH_RADIOMICS_SUCCESS;
    const int log_timing = flash_radiomics_env_truthy("FLASH_RADIOMICS_VOXEL_CUDA_TIMING");
    double total_h2d_ms = 0.0;
    double total_device_build_ms = 0.0;
    double total_device_finalize_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_host_finalize_ms = 0.0;
    const int num_directions = (spec.ndim == 2) ? 4 : 13;
    const size_t matrices_per_patch =
        (size_t)num_distances * (size_t)num_directions;

    size_t matrix_size = 0;
    if (!safe_mul_size_t((size_t)ng, (size_t)ng, &matrix_size)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    size_t counts_per_patch = 0;
    if (!safe_mul_size_t(matrices_per_patch, matrix_size, &counts_per_patch)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    int* d_distances = NULL;
    int* d_discretized = NULL;
    uint8_t* d_mask = NULL;
    int* d_glcm_counts = NULL;
    uint8_t* d_include_features = NULL;
    double* d_feature_rows = NULL;
    double* d_glcm_px = NULL;
    double* d_glcm_py = NULL;
    double* d_glcm_px_add_y = NULL;
    double* d_glcm_px_sub_y = NULL;

    size_t chunk_batch = (size_t)batch_size;
    const size_t target_counts_bytes = choose_chunk_target_bytes(
        128ull * 1024ull * 1024ull,
        spec.window_voxels,
        ng
    );
    size_t counts_bytes_per_patch = 0;
    size_t allocated_chunk_discretized_bytes = 0;
    size_t allocated_chunk_mask_bytes = 0;
    size_t allocated_chunk_counts_bytes = 0;
    size_t allocated_chunk_output_bytes = 0;
    size_t allocated_chunk_finalize_scratch_bytes = 0;
    const int build_block_size = choose_voxel_build_block_size(spec.window_voxels, ng);

    cudaError_t err = cudaMalloc(&d_distances, (size_t)num_distances * sizeof(int));
    if (err != cudaSuccess) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto glcm_cleanup;
    }
    err = cudaMemcpyAsync(
        d_distances,
        distances,
        (size_t)num_distances * sizeof(int),
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (err != cudaSuccess) {
        status = FLASH_RADIOMICS_ERROR_IO_FAILED;
        goto glcm_cleanup;
    }
    if (include_features) {
        err = cudaMalloc(
            &d_include_features,
            (size_t)FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT * sizeof(uint8_t)
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            goto glcm_cleanup;
        }
        err = cudaMemcpyAsync(
            d_include_features,
            include_features,
            (size_t)FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            goto glcm_cleanup;
        }
    }

    if (!safe_mul_size_t(counts_per_patch, sizeof(int), &counts_bytes_per_patch)) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto glcm_cleanup;
    }
    chunk_batch = cap_chunk_batch_by_target(
        chunk_batch,
        counts_bytes_per_patch,
        target_counts_bytes
    );

    while (chunk_batch > 0) {
        size_t chunk_voxels = 0;
        size_t chunk_matrices = 0;
        size_t chunk_counts = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_counts_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;
        size_t px_entries = 0;
        size_t px_bytes = 0;
        size_t px_add_y_entries = 0;
        size_t px_add_y_bytes = 0;
        size_t scratch_three_px_bytes = 0;
        size_t scratch_bytes = 0;

        if (!safe_mul_size_t(chunk_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(chunk_batch, matrices_per_patch, &chunk_matrices) ||
            !safe_mul_size_t(chunk_matrices, matrix_size, &chunk_counts) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_counts, sizeof(int), &chunk_counts_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)ng, &px_entries) ||
            !safe_mul_size_t(px_entries, sizeof(double), &px_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)(2 * ng - 1), &px_add_y_entries) ||
            !safe_mul_size_t(px_add_y_entries, sizeof(double), &px_add_y_bytes) ||
            !safe_mul_size_t(px_bytes, 3u, &scratch_three_px_bytes) ||
            !safe_add_size_t(scratch_three_px_bytes, px_add_y_bytes, &scratch_bytes)) {
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_discretized, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            if (d_discretized) cudaFree(d_discretized);
            d_discretized = NULL;
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_mask, chunk_mask_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            d_discretized = NULL;
            d_mask = NULL;
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_glcm_counts, chunk_counts_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            d_discretized = NULL;
            d_mask = NULL;
            d_glcm_counts = NULL;
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_feature_rows, chunk_output_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_glcm_counts);
            d_discretized = NULL;
            d_mask = NULL;
            d_glcm_counts = NULL;
            d_feature_rows = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_glcm_px, px_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_glcm_counts);
            cudaFree(d_feature_rows);
            d_discretized = NULL;
            d_mask = NULL;
            d_glcm_counts = NULL;
            d_feature_rows = NULL;
            d_glcm_px = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_glcm_py, px_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_glcm_counts);
            cudaFree(d_feature_rows);
            cudaFree(d_glcm_px);
            d_discretized = NULL;
            d_mask = NULL;
            d_glcm_counts = NULL;
            d_feature_rows = NULL;
            d_glcm_px = NULL;
            d_glcm_py = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_glcm_px_add_y, px_add_y_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_glcm_counts);
            cudaFree(d_feature_rows);
            cudaFree(d_glcm_px);
            cudaFree(d_glcm_py);
            d_discretized = NULL;
            d_mask = NULL;
            d_glcm_counts = NULL;
            d_feature_rows = NULL;
            d_glcm_px = NULL;
            d_glcm_py = NULL;
            d_glcm_px_add_y = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_glcm_px_sub_y, px_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_glcm_counts);
            cudaFree(d_feature_rows);
            cudaFree(d_glcm_px);
            cudaFree(d_glcm_py);
            cudaFree(d_glcm_px_add_y);
            d_discretized = NULL;
            d_mask = NULL;
            d_glcm_counts = NULL;
            d_feature_rows = NULL;
            d_glcm_px = NULL;
            d_glcm_py = NULL;
            d_glcm_px_add_y = NULL;
            d_glcm_px_sub_y = NULL;
            chunk_batch /= 2;
            continue;
        }
        allocated_chunk_discretized_bytes = chunk_discretized_bytes;
        allocated_chunk_mask_bytes = chunk_mask_bytes;
        allocated_chunk_counts_bytes = chunk_counts_bytes;
        allocated_chunk_output_bytes = chunk_output_bytes;
        allocated_chunk_finalize_scratch_bytes = scratch_bytes;
        break;
    }

    if (chunk_batch == 0) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto glcm_cleanup;
    }


    for (size_t batch_offset = 0; batch_offset < (size_t)batch_size; batch_offset += chunk_batch) {
        size_t current_batch = (size_t)batch_size - batch_offset;
        if (current_batch > chunk_batch) {
            current_batch = chunk_batch;
        }

        size_t chunk_voxels = 0;
        size_t chunk_matrices = 0;
        size_t chunk_counts = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_counts_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;
        if (!safe_mul_size_t(current_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(current_batch, matrices_per_patch, &chunk_matrices) ||
            !safe_mul_size_t(chunk_matrices, matrix_size, &chunk_counts) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_counts, sizeof(int), &chunk_counts_bytes) ||
            !safe_mul_size_t(current_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            break;
        }

        const int* chunk_discretized =
            discretized_windows + batch_offset * spec.window_voxels;
        const uint8_t* chunk_mask =
            mask_windows + batch_offset * spec.window_voxels;

        const std::chrono::steady_clock::time_point h2d_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            d_discretized,
            chunk_discretized,
            chunk_discretized_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaMemcpyAsync(
            d_mask,
            chunk_mask,
            chunk_mask_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_h2d_ms += elapsed_ms(h2d_start, std::chrono::steady_clock::now());
        }

        const std::chrono::steady_clock::time_point device_build_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemsetAsync(d_glcm_counts, 0, chunk_counts_bytes, ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        const int grid_size = compute_grid_size_1d(chunk_voxels, build_block_size, 4096);

        build_glcm_counts_batched_kernel<<<grid_size, build_block_size, 0, ctx->stream>>>(
            d_discretized,
            d_mask,
            d_glcm_counts,
            (int)current_batch,
            spec.window_voxels,
            spec.dims[0],
            spec.dims[1],
            spec.dims[2],
            spec.ndim,
            d_distances,
            num_distances,
            num_directions,
            ng
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_build_ms += elapsed_ms(
                device_build_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point device_finalize_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        const int finalize_block_size = choose_voxel_finalize_block_size(current_batch);
        const int finalize_grid_size =
            compute_grid_size_1d(current_batch, finalize_block_size, 4096);
        finalize_glcm_features_batched_kernel<<<finalize_grid_size, finalize_block_size, 0, ctx->stream>>>(
            d_glcm_counts,
            (int)current_batch,
            (int)matrices_per_patch,
            ng,
            d_include_features,
            d_glcm_px,
            d_glcm_py,
            d_glcm_px_add_y,
            d_glcm_px_sub_y,
            d_feature_rows,
            out_feature_stride
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_finalize_ms += elapsed_ms(
                device_finalize_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point d2h_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            out_features + batch_offset * (size_t)out_feature_stride,
            d_feature_rows,
            chunk_output_bytes,
            cudaMemcpyDeviceToHost,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaStreamSynchronize(ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            total_d2h_ms += elapsed_ms(d2h_start, std::chrono::steady_clock::now());
        }
    }

glcm_cleanup:
    if (d_distances) cudaFree(d_distances);
    if (d_discretized) cudaFree(d_discretized);
    if (d_mask) cudaFree(d_mask);
    if (d_glcm_counts) cudaFree(d_glcm_counts);
    if (d_include_features) cudaFree(d_include_features);
    if (d_feature_rows) cudaFree(d_feature_rows);
    if (d_glcm_px) cudaFree(d_glcm_px);
    if (d_glcm_py) cudaFree(d_glcm_py);
    if (d_glcm_px_add_y) cudaFree(d_glcm_px_add_y);
    if (d_glcm_px_sub_y) cudaFree(d_glcm_px_sub_y);

    if (log_timing) {
        log_voxel_cuda_stage_summary(
            "glcm",
            batch_size,
            chunk_batch,
            spec.window_voxels,
            allocated_chunk_output_bytes,
            allocated_chunk_discretized_bytes + allocated_chunk_mask_bytes,
            allocated_chunk_counts_bytes + allocated_chunk_output_bytes +
                allocated_chunk_finalize_scratch_bytes + (size_t)num_distances * sizeof(int),
            0,
            total_h2d_ms,
            total_device_build_ms,
            total_device_finalize_ms,
            total_d2h_ms,
            total_host_finalize_ms
        );
    }
    return status;
}

extern "C" int flash_radiomics_glrlm_cuda_discretized_voxel_batch(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    const uint8_t* include_features,
    int ng,
    double* out_features,
    int out_feature_stride
) {
    WindowSpec spec;
    if (!discretized_windows || !mask_windows || !out_features ||
        batch_size <= 0 || ng <= 0 ||
        !init_window_spec(ndim, window_dims, &spec)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    const int enabled_count =
        count_enabled_features(include_features, FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT);
    if (enabled_count <= 0 || out_feature_stride < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    CudaContext* ctx = get_cached_voxel_cuda_context();
    if (!ctx) {
        return FLASH_RADIOMICS_ERROR_BACKEND_UNAVAILABLE;
    }

    int status = FLASH_RADIOMICS_SUCCESS;
    const int log_timing = flash_radiomics_env_truthy("FLASH_RADIOMICS_VOXEL_CUDA_TIMING");
    double total_h2d_ms = 0.0;
    double total_device_build_ms = 0.0;
    double total_device_finalize_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_host_finalize_ms = 0.0;
    const int num_directions = (spec.ndim == 2) ? 4 : 13;
    int max_run_length = spec.dims[0];
    if (spec.dims[1] > max_run_length) max_run_length = spec.dims[1];
    if (spec.dims[2] > max_run_length) max_run_length = spec.dims[2];

    size_t matrix_size = 0;
    if (!safe_mul_size_t((size_t)ng, (size_t)max_run_length, &matrix_size)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    size_t counts_per_patch = 0;
    if (!safe_mul_size_t((size_t)num_directions, matrix_size, &counts_per_patch)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    int* d_discretized = NULL;
    uint8_t* d_mask = NULL;
    int* d_counts = NULL;
    int* d_dir_has_multi_element = NULL;
    uint8_t* d_include_features = NULL;
    double* d_feature_rows = NULL;

    const size_t target_counts_bytes = choose_chunk_target_bytes(
        128ull * 1024ull * 1024ull,
        spec.window_voxels,
        ng
    );
    size_t counts_bytes_per_patch = 0;
    size_t chunk_batch = (size_t)batch_size;
    size_t allocated_chunk_discretized_bytes = 0;
    size_t allocated_chunk_mask_bytes = 0;
    size_t allocated_chunk_counts_bytes = 0;
    size_t allocated_chunk_dir_flags_bytes = 0;
    size_t allocated_chunk_output_bytes = 0;
    const int build_block_size = choose_voxel_build_block_size(spec.window_voxels, ng);

    if (!safe_mul_size_t(counts_per_patch, sizeof(int), &counts_bytes_per_patch)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    chunk_batch = cap_chunk_batch_by_target(
        chunk_batch,
        counts_bytes_per_patch,
        target_counts_bytes
    );

    cudaError_t err = cudaSuccess;
    if (include_features) {
        cudaError_t include_err = cudaMalloc(
            &d_include_features,
            (size_t)FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT * sizeof(uint8_t)
        );
        if (include_err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            goto glrlm_cleanup;
        }
        include_err = cudaMemcpyAsync(
            d_include_features,
            include_features,
            (size_t)FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (include_err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            goto glrlm_cleanup;
        }
    }
    while (chunk_batch > 0) {
        size_t chunk_voxels = 0;
        size_t chunk_counts = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_counts_bytes = 0;
        size_t chunk_dir_flags = 0;
        size_t chunk_dir_flags_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;

        if (!safe_mul_size_t(chunk_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(chunk_batch, counts_per_patch, &chunk_counts) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_counts, sizeof(int), &chunk_counts_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)num_directions, &chunk_dir_flags) ||
            !safe_mul_size_t(chunk_dir_flags, sizeof(int), &chunk_dir_flags_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_discretized, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            d_discretized = NULL;
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_mask, chunk_mask_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            d_discretized = NULL;
            d_mask = NULL;
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_counts, chunk_counts_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            d_discretized = NULL;
            d_mask = NULL;
            d_counts = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_dir_has_multi_element, chunk_dir_flags_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_counts);
            d_discretized = NULL;
            d_mask = NULL;
            d_counts = NULL;
            d_dir_has_multi_element = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_feature_rows, chunk_output_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_counts);
            cudaFree(d_dir_has_multi_element);
            d_discretized = NULL;
            d_mask = NULL;
            d_counts = NULL;
            d_dir_has_multi_element = NULL;
            d_feature_rows = NULL;
            chunk_batch /= 2;
            continue;
        }
        allocated_chunk_discretized_bytes = chunk_discretized_bytes;
        allocated_chunk_mask_bytes = chunk_mask_bytes;
        allocated_chunk_counts_bytes = chunk_counts_bytes;
        allocated_chunk_dir_flags_bytes = chunk_dir_flags_bytes;
        allocated_chunk_output_bytes = chunk_output_bytes;
        break;
    }

    if (chunk_batch == 0) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto glrlm_cleanup;
    }

    for (size_t batch_offset = 0; batch_offset < (size_t)batch_size; batch_offset += chunk_batch) {
        size_t current_batch = (size_t)batch_size - batch_offset;
        if (current_batch > chunk_batch) {
            current_batch = chunk_batch;
        }

        size_t chunk_voxels = 0;
        size_t chunk_counts = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_counts_bytes = 0;
        size_t chunk_dir_flags = 0;
        size_t chunk_dir_flags_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;
        if (!safe_mul_size_t(current_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(current_batch, counts_per_patch, &chunk_counts) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_counts, sizeof(int), &chunk_counts_bytes) ||
            !safe_mul_size_t(current_batch, (size_t)num_directions, &chunk_dir_flags) ||
            !safe_mul_size_t(chunk_dir_flags, sizeof(int), &chunk_dir_flags_bytes) ||
            !safe_mul_size_t(current_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            break;
        }

        const int* chunk_discretized = discretized_windows + batch_offset * spec.window_voxels;
        const uint8_t* chunk_mask = mask_windows + batch_offset * spec.window_voxels;

        const std::chrono::steady_clock::time_point h2d_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            d_discretized,
            chunk_discretized,
            chunk_discretized_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaMemcpyAsync(
            d_mask,
            chunk_mask,
            chunk_mask_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_h2d_ms += elapsed_ms(h2d_start, std::chrono::steady_clock::now());
        }

        const std::chrono::steady_clock::time_point device_build_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemsetAsync(d_counts, 0, chunk_counts_bytes, ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        err = cudaMemsetAsync(d_dir_has_multi_element, 0, chunk_dir_flags_bytes, ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        size_t total_tasks = 0;
        if (!safe_mul_size_t(current_batch, (size_t)num_directions, &total_tasks) ||
            !safe_mul_size_t(total_tasks, spec.window_voxels, &total_tasks)) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            break;
        }
        const int grid_size = compute_grid_size_1d(total_tasks, build_block_size, 4096);
        build_glrlm_counts_batched_kernel<<<grid_size, build_block_size, 0, ctx->stream>>>(
            d_discretized,
            d_mask,
            d_counts,
            d_dir_has_multi_element,
            (int)current_batch,
            spec.window_voxels,
            spec.dims[0],
            spec.dims[1],
            spec.dims[2],
            spec.ndim,
            num_directions,
            ng,
            max_run_length
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_build_ms += elapsed_ms(
                device_build_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point device_finalize_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        const int finalize_block_size = choose_voxel_finalize_block_size(current_batch);
        const int finalize_grid_size =
            compute_grid_size_1d(current_batch, finalize_block_size, 4096);
        finalize_glrlm_features_batched_kernel<<<finalize_grid_size, finalize_block_size, 0, ctx->stream>>>(
            d_counts,
            d_dir_has_multi_element,
            (int)current_batch,
            num_directions,
            ng,
            max_run_length,
            d_include_features,
            d_feature_rows,
            out_feature_stride
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_finalize_ms += elapsed_ms(
                device_finalize_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point d2h_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            out_features + batch_offset * (size_t)out_feature_stride,
            d_feature_rows,
            chunk_output_bytes,
            cudaMemcpyDeviceToHost,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaStreamSynchronize(ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            total_d2h_ms += elapsed_ms(d2h_start, std::chrono::steady_clock::now());
        }
    }

glrlm_cleanup:
    if (d_discretized) cudaFree(d_discretized);
    if (d_mask) cudaFree(d_mask);
    if (d_counts) cudaFree(d_counts);
    if (d_dir_has_multi_element) cudaFree(d_dir_has_multi_element);
    if (d_include_features) cudaFree(d_include_features);
    if (d_feature_rows) cudaFree(d_feature_rows);
    if (log_timing) {
        log_voxel_cuda_stage_summary(
            "glrlm",
            batch_size,
            chunk_batch,
            spec.window_voxels,
            allocated_chunk_output_bytes,
            allocated_chunk_discretized_bytes + allocated_chunk_mask_bytes,
            allocated_chunk_counts_bytes + allocated_chunk_dir_flags_bytes +
                allocated_chunk_output_bytes,
            0,
            total_h2d_ms,
            total_device_build_ms,
            total_device_finalize_ms,
            total_d2h_ms,
            total_host_finalize_ms
        );
    }
    return status;
}

extern "C" int flash_radiomics_glszm_cuda_discretized_voxel_batch(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    const uint8_t* include_features,
    int ng,
    double* out_features,
    int out_feature_stride
) {
    WindowSpec spec;
    if (!discretized_windows || !mask_windows || !out_features ||
        batch_size <= 0 || ng <= 0 ||
        !init_window_spec(ndim, window_dims, &spec)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    const int enabled_count =
        count_enabled_features(include_features, FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT);
    if (enabled_count <= 0 || out_feature_stride < enabled_count) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    CudaContext* ctx = get_cached_voxel_cuda_context();
    if (!ctx) {
        return FLASH_RADIOMICS_ERROR_BACKEND_UNAVAILABLE;
    }

    int status = FLASH_RADIOMICS_SUCCESS;
    const int log_timing = flash_radiomics_env_truthy("FLASH_RADIOMICS_VOXEL_CUDA_TIMING");
    double total_h2d_ms = 0.0;
    double total_device_build_ms = 0.0;
    double total_device_finalize_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_host_finalize_ms = 0.0;
    int* d_discretized = NULL;
    uint8_t* d_mask = NULL;
    int* d_labels = NULL;
    int* d_zone_sizes = NULL;
    int* d_zone_representatives = NULL;
    int* d_glszm_counts = NULL;
    int* d_changed = NULL;
    uint8_t* d_include_features = NULL;
    double* d_feature_rows = NULL;

    const int max_zone_size = (int)spec.window_voxels;
    size_t matrix_size = 0;
    if (!safe_mul_size_t((size_t)ng, (size_t)max_zone_size, &matrix_size)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    size_t counts_bytes_per_patch = 0;
    if (!safe_mul_size_t(matrix_size, sizeof(int), &counts_bytes_per_patch)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    size_t chunk_batch = (size_t)batch_size;
    const size_t target_counts_bytes = 128ull * 1024ull * 1024ull;
    size_t allocated_chunk_discretized_bytes = 0;
    size_t allocated_chunk_mask_bytes = 0;
    size_t allocated_chunk_counts_bytes = 0;
    size_t allocated_chunk_output_bytes = 0;
    if (counts_bytes_per_patch > 0) {
        const size_t max_by_target = target_counts_bytes / counts_bytes_per_patch;
        if (max_by_target > 0 && chunk_batch > max_by_target) {
            chunk_batch = max_by_target;
        }
    }
    if (chunk_batch == 0) {
        chunk_batch = 1;
    }

    cudaError_t err = cudaSuccess;
    if (include_features) {
        err = cudaMalloc(
            &d_include_features,
            (size_t)FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT * sizeof(uint8_t)
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            goto glszm_cleanup;
        }
        err = cudaMemcpyAsync(
            d_include_features,
            include_features,
            (size_t)FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            goto glszm_cleanup;
        }
    }

    while (chunk_batch > 0) {
        size_t chunk_voxels = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_matrix_entries = 0;
        size_t chunk_counts_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;

        if (!safe_mul_size_t(chunk_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_batch, matrix_size, &chunk_matrix_entries) ||
            !safe_mul_size_t(chunk_matrix_entries, sizeof(int), &chunk_counts_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_discretized, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            d_discretized = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_mask, chunk_mask_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            d_discretized = NULL;
            d_mask = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_labels, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            d_discretized = NULL;
            d_mask = NULL;
            d_labels = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_zone_sizes, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_labels);
            d_discretized = NULL;
            d_mask = NULL;
            d_labels = NULL;
            d_zone_sizes = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_zone_representatives, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_labels);
            cudaFree(d_zone_sizes);
            d_discretized = NULL;
            d_mask = NULL;
            d_labels = NULL;
            d_zone_sizes = NULL;
            d_zone_representatives = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_glszm_counts, chunk_counts_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_labels);
            cudaFree(d_zone_sizes);
            cudaFree(d_zone_representatives);
            d_discretized = NULL;
            d_mask = NULL;
            d_labels = NULL;
            d_zone_sizes = NULL;
            d_zone_representatives = NULL;
            d_glszm_counts = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_changed, sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_labels);
            cudaFree(d_zone_sizes);
            cudaFree(d_zone_representatives);
            cudaFree(d_glszm_counts);
            d_discretized = NULL;
            d_mask = NULL;
            d_labels = NULL;
            d_zone_sizes = NULL;
            d_zone_representatives = NULL;
            d_glszm_counts = NULL;
            d_changed = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_feature_rows, chunk_output_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_labels);
            cudaFree(d_zone_sizes);
            cudaFree(d_zone_representatives);
            cudaFree(d_glszm_counts);
            cudaFree(d_changed);
            d_discretized = NULL;
            d_mask = NULL;
            d_labels = NULL;
            d_zone_sizes = NULL;
            d_zone_representatives = NULL;
            d_glszm_counts = NULL;
            d_changed = NULL;
            d_feature_rows = NULL;
            chunk_batch /= 2;
            continue;
        }
        allocated_chunk_discretized_bytes = chunk_discretized_bytes;
        allocated_chunk_mask_bytes = chunk_mask_bytes;
        allocated_chunk_counts_bytes = chunk_counts_bytes;
        allocated_chunk_output_bytes = chunk_output_bytes;
        break;
    }

    if (chunk_batch == 0) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto glszm_cleanup;
    }

    for (size_t batch_offset = 0; batch_offset < (size_t)batch_size; batch_offset += chunk_batch) {
        size_t current_batch = (size_t)batch_size - batch_offset;
        if (current_batch > chunk_batch) {
            current_batch = chunk_batch;
        }

        size_t chunk_voxels = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_matrix_entries = 0;
        size_t chunk_counts_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;

        if (!safe_mul_size_t(current_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(current_batch, matrix_size, &chunk_matrix_entries) ||
            !safe_mul_size_t(chunk_matrix_entries, sizeof(int), &chunk_counts_bytes) ||
            !safe_mul_size_t(current_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            break;
        }

        const int* chunk_discretized = discretized_windows + batch_offset * spec.window_voxels;
        const uint8_t* chunk_mask = mask_windows + batch_offset * spec.window_voxels;

        const std::chrono::steady_clock::time_point h2d_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            d_discretized,
            chunk_discretized,
            chunk_discretized_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaMemcpyAsync(
            d_mask,
            chunk_mask,
            chunk_mask_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_h2d_ms += elapsed_ms(h2d_start, std::chrono::steady_clock::now());
        }

        const int block_size = 256;
        const int grid_size = compute_grid_size_1d(chunk_voxels, block_size, 4096);
        const std::chrono::steady_clock::time_point device_build_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();

        ccl_init_batched_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
            d_mask,
            d_labels,
            chunk_voxels
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        int changed = 1;
        const int max_iterations = 128;
        for (int iter = 0; changed && iter < max_iterations; iter++) {
            changed = 0;
            err = cudaMemcpyAsync(
                d_changed,
                &changed,
                sizeof(int),
                cudaMemcpyHostToDevice,
                ctx->stream
            );
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_IO_FAILED;
                break;
            }

            ccl_merge_batched_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
                d_discretized,
                d_mask,
                d_labels,
                d_changed,
                chunk_voxels,
                spec.window_voxels,
                spec.dims[0],
                spec.dims[1],
                spec.dims[2],
                spec.ndim
            );
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }

            err = cudaMemcpyAsync(
                &changed,
                d_changed,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                ctx->stream
            );
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_IO_FAILED;
                break;
            }
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
        }
        if (status != FLASH_RADIOMICS_SUCCESS) {
            break;
        }
        ccl_flatten_batched_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
            d_labels,
            chunk_voxels
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        err = cudaMemsetAsync(d_zone_sizes, 0, chunk_discretized_bytes, ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        fill_int_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
            d_zone_representatives,
            INT_MAX,
            chunk_voxels
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        compute_zone_sizes_batched_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
            d_labels,
            d_mask,
            d_zone_sizes,
            chunk_voxels
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        mark_zone_representatives_batched_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
            d_labels,
            d_mask,
            d_zone_representatives,
            chunk_voxels
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        err = cudaMemsetAsync(d_glszm_counts, 0, chunk_counts_bytes, ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        build_glszm_counts_batched_kernel<<<grid_size, block_size, 0, ctx->stream>>>(
            d_discretized,
            d_mask,
            d_labels,
            d_zone_sizes,
            d_zone_representatives,
            d_glszm_counts,
            chunk_voxels,
            spec.window_voxels,
            ng,
            max_zone_size
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_build_ms += elapsed_ms(
                device_build_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point device_finalize_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        const int finalize_block_size = choose_voxel_finalize_block_size(current_batch);
        const int finalize_grid_size =
            compute_grid_size_1d(current_batch, finalize_block_size, 4096);
        finalize_glszm_features_batched_kernel<<<finalize_grid_size, finalize_block_size, 0, ctx->stream>>>(
            d_glszm_counts,
            d_mask,
            (int)current_batch,
            spec.window_voxels,
            ng,
            max_zone_size,
            d_include_features,
            d_feature_rows,
            out_feature_stride
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_finalize_ms += elapsed_ms(
                device_finalize_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point d2h_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            out_features + batch_offset * (size_t)out_feature_stride,
            d_feature_rows,
            chunk_output_bytes,
            cudaMemcpyDeviceToHost,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaStreamSynchronize(ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            total_d2h_ms += elapsed_ms(d2h_start, std::chrono::steady_clock::now());
        }
    }

glszm_cleanup:
    if (d_discretized) cudaFree(d_discretized);
    if (d_mask) cudaFree(d_mask);
    if (d_labels) cudaFree(d_labels);
    if (d_zone_sizes) cudaFree(d_zone_sizes);
    if (d_zone_representatives) cudaFree(d_zone_representatives);
    if (d_glszm_counts) cudaFree(d_glszm_counts);
    if (d_changed) cudaFree(d_changed);
    if (d_include_features) cudaFree(d_include_features);
    if (d_feature_rows) cudaFree(d_feature_rows);
    if (log_timing) {
        log_voxel_cuda_stage_summary(
            "glszm",
            batch_size,
            chunk_batch,
            spec.window_voxels,
            allocated_chunk_output_bytes,
            allocated_chunk_discretized_bytes + allocated_chunk_mask_bytes,
            allocated_chunk_counts_bytes + allocated_chunk_output_bytes +
                3u * allocated_chunk_discretized_bytes + sizeof(int),
            0,
            total_h2d_ms,
            total_device_build_ms,
            total_device_finalize_ms,
            total_d2h_ms,
            total_host_finalize_ms
        );
    }
    return status;
}

extern "C" int flash_radiomics_gldm_cuda_discretized_voxel_batch(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    const int* distances,
    int num_distances,
    const uint8_t* include_features,
    int ng,
    double gldm_a,
    double* out_features,
    int out_feature_stride
) {
    WindowSpec spec;
    if (!discretized_windows || !mask_windows || !distances || !out_features ||
        batch_size <= 0 || num_distances <= 0 || ng <= 0 ||
        !init_window_spec(ndim, window_dims, &spec)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    const int enabled_count =
        count_enabled_features(include_features, FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT);
    if (enabled_count <= 0 || out_feature_stride < FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    CudaContext* ctx = get_cached_voxel_cuda_context();
    if (!ctx) {
        return FLASH_RADIOMICS_ERROR_BACKEND_UNAVAILABLE;
    }

    int status = FLASH_RADIOMICS_SUCCESS;
    const int log_timing = flash_radiomics_env_truthy("FLASH_RADIOMICS_VOXEL_CUDA_TIMING");
    double total_h2d_ms = 0.0;
    double total_device_build_ms = 0.0;
    double total_device_finalize_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_host_finalize_ms = 0.0;
    const int num_dirs = (spec.ndim == 2) ? 8 : 26;
    const int max_dep = num_dirs * num_distances;
    const int dep_len = max_dep + 1;

    size_t matrix_size = 0;
    if (!safe_mul_size_t((size_t)ng, (size_t)dep_len, &matrix_size)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    int* d_distances = NULL;
    int* d_discretized = NULL;
    uint8_t* d_mask = NULL;
    int* d_matrix = NULL;
    uint8_t* d_include_features = NULL;
    double* d_feature_rows = NULL;

    size_t chunk_batch = (size_t)batch_size;
    const size_t target_counts_bytes = choose_chunk_target_bytes(
        128ull * 1024ull * 1024ull,
        spec.window_voxels,
        ng
    );
    size_t counts_bytes_per_patch = 0;
    size_t allocated_chunk_discretized_bytes = 0;
    size_t allocated_chunk_mask_bytes = 0;
    size_t allocated_chunk_matrix_bytes = 0;
    size_t allocated_chunk_output_bytes = 0;
    const int build_block_size = choose_voxel_build_block_size(spec.window_voxels, ng);
    if (!safe_mul_size_t(matrix_size, sizeof(int), &counts_bytes_per_patch)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    chunk_batch = cap_chunk_batch_by_target(
        chunk_batch,
        counts_bytes_per_patch,
        target_counts_bytes
    );

    cudaError_t err = cudaSuccess;
    if (include_features) {
        cudaError_t include_err = cudaMalloc(
            &d_include_features,
            (size_t)FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT * sizeof(uint8_t)
        );
        if (include_err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            goto gldm_cleanup;
        }
        include_err = cudaMemcpyAsync(
            d_include_features,
            include_features,
            (size_t)FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (include_err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            goto gldm_cleanup;
        }
    }

    err = cudaMalloc(&d_distances, (size_t)num_distances * sizeof(int));
    if (err != cudaSuccess) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto gldm_cleanup;
    }
    err = cudaMemcpyAsync(
        d_distances,
        distances,
        (size_t)num_distances * sizeof(int),
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (err != cudaSuccess) {
        status = FLASH_RADIOMICS_ERROR_IO_FAILED;
        goto gldm_cleanup;
    }

    while (chunk_batch > 0) {
        size_t chunk_voxels = 0;
        size_t chunk_matrix_entries = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_matrix_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;

        if (!safe_mul_size_t(chunk_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(chunk_batch, matrix_size, &chunk_matrix_entries) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_matrix_entries, sizeof(int), &chunk_matrix_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_discretized, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            d_discretized = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_mask, chunk_mask_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            d_discretized = NULL;
            d_mask = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_matrix, chunk_matrix_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            d_discretized = NULL;
            d_mask = NULL;
            d_matrix = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_feature_rows, chunk_output_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_matrix);
            d_discretized = NULL;
            d_mask = NULL;
            d_matrix = NULL;
            d_feature_rows = NULL;
            chunk_batch /= 2;
            continue;
        }
        allocated_chunk_discretized_bytes = chunk_discretized_bytes;
        allocated_chunk_mask_bytes = chunk_mask_bytes;
        allocated_chunk_matrix_bytes = chunk_matrix_bytes;
        allocated_chunk_output_bytes = chunk_output_bytes;
        break;
    }

    if (chunk_batch == 0) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto gldm_cleanup;
    }

    for (size_t batch_offset = 0; batch_offset < (size_t)batch_size; batch_offset += chunk_batch) {
        size_t current_batch = (size_t)batch_size - batch_offset;
        if (current_batch > chunk_batch) {
            current_batch = chunk_batch;
        }

        size_t chunk_voxels = 0;
        size_t chunk_matrix_entries = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_matrix_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;
        if (!safe_mul_size_t(current_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(current_batch, matrix_size, &chunk_matrix_entries) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_matrix_entries, sizeof(int), &chunk_matrix_bytes) ||
            !safe_mul_size_t(current_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            break;
        }

        const int* chunk_discretized = discretized_windows + batch_offset * spec.window_voxels;
        const uint8_t* chunk_mask = mask_windows + batch_offset * spec.window_voxels;

        const std::chrono::steady_clock::time_point h2d_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            d_discretized,
            chunk_discretized,
            chunk_discretized_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaMemcpyAsync(
            d_mask,
            chunk_mask,
            chunk_mask_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_h2d_ms += elapsed_ms(h2d_start, std::chrono::steady_clock::now());
        }

        const std::chrono::steady_clock::time_point device_build_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemsetAsync(d_matrix, 0, chunk_matrix_bytes, ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        const int grid_size = compute_grid_size_1d(chunk_voxels, build_block_size, 4096);
        build_gldm_matrix_batched_kernel<<<grid_size, build_block_size, 0, ctx->stream>>>(
            d_discretized,
            d_mask,
            d_matrix,
            (int)current_batch,
            spec.window_voxels,
            spec.dims[0],
            spec.dims[1],
            spec.dims[2],
            spec.ndim,
            d_distances,
            num_distances,
            num_dirs,
            max_dep,
            ng,
            gldm_a
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_build_ms += elapsed_ms(
                device_build_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point device_finalize_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        const int finalize_block_size = choose_voxel_finalize_block_size(current_batch);
        const int finalize_grid_size =
            compute_grid_size_1d(current_batch, finalize_block_size, 4096);
        finalize_gldm_features_batched_kernel<<<finalize_grid_size, finalize_block_size, 0, ctx->stream>>>(
            d_matrix,
            (int)current_batch,
            ng,
            dep_len,
            d_include_features,
            d_feature_rows,
            out_feature_stride
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_finalize_ms += elapsed_ms(
                device_finalize_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point d2h_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            out_features + batch_offset * (size_t)out_feature_stride,
            d_feature_rows,
            chunk_output_bytes,
            cudaMemcpyDeviceToHost,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaStreamSynchronize(ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            total_d2h_ms += elapsed_ms(d2h_start, std::chrono::steady_clock::now());
        }
    }

gldm_cleanup:
    if (d_distances) cudaFree(d_distances);
    if (d_discretized) cudaFree(d_discretized);
    if (d_mask) cudaFree(d_mask);
    if (d_matrix) cudaFree(d_matrix);
    if (d_include_features) cudaFree(d_include_features);
    if (d_feature_rows) cudaFree(d_feature_rows);
    if (log_timing) {
        log_voxel_cuda_stage_summary(
            "gldm",
            batch_size,
            chunk_batch,
            spec.window_voxels,
            allocated_chunk_output_bytes,
            allocated_chunk_discretized_bytes + allocated_chunk_mask_bytes,
            allocated_chunk_matrix_bytes + allocated_chunk_output_bytes + (size_t)num_distances * sizeof(int),
            0,
            total_h2d_ms,
            total_device_build_ms,
            total_device_finalize_ms,
            total_d2h_ms,
            total_host_finalize_ms
        );
    }
    return status;
}

extern "C" int flash_radiomics_ngtdm_cuda_discretized_voxel_batch(
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    const int* distances,
    int num_distances,
    const uint8_t* include_features,
    int ng,
    double* out_features,
    int out_feature_stride
) {
    WindowSpec spec;
    if (!discretized_windows || !mask_windows || !distances || !out_features ||
        batch_size <= 0 || num_distances <= 0 || ng <= 0 ||
        !init_window_spec(ndim, window_dims, &spec)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    const int enabled_count =
        count_enabled_features(include_features, FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT);
    if (enabled_count <= 0 || out_feature_stride < FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    CudaContext* ctx = get_cached_voxel_cuda_context();
    if (!ctx) {
        return FLASH_RADIOMICS_ERROR_BACKEND_UNAVAILABLE;
    }

    int status = FLASH_RADIOMICS_SUCCESS;
    const int log_timing = flash_radiomics_env_truthy("FLASH_RADIOMICS_VOXEL_CUDA_TIMING");
    double total_h2d_ms = 0.0;
    double total_device_build_ms = 0.0;
    double total_device_finalize_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_host_finalize_ms = 0.0;
    const int num_dirs = (spec.ndim == 2) ? 8 : 26;
    size_t ngtdm_shared_bytes = 0;
    int use_patch_kernel = 0;
    if (safe_mul_size_t((size_t)ng, sizeof(double), &ngtdm_shared_bytes) &&
        safe_mul_size_t(ngtdm_shared_bytes, 2u, &ngtdm_shared_bytes) &&
        ngtdm_shared_bytes <= 48ull * 1024ull) {
        use_patch_kernel = 1;
    }

    int* d_distances = NULL;
    int* d_discretized = NULL;
    uint8_t* d_mask = NULL;
    double* d_n_i = NULL;
    double* d_s_i = NULL;
    uint8_t* d_include_features = NULL;
    double* d_feature_rows = NULL;

    size_t chunk_batch = (size_t)batch_size;
    const size_t target_vectors_bytes = choose_chunk_target_bytes(
        64ull * 1024ull * 1024ull,
        spec.window_voxels,
        ng
    );
    size_t vectors_bytes_per_patch = 0;
    size_t allocated_chunk_discretized_bytes = 0;
    size_t allocated_chunk_mask_bytes = 0;
    size_t allocated_chunk_vector_bytes = 0;
    size_t allocated_chunk_output_bytes = 0;
    const int build_block_size = choose_voxel_build_block_size(spec.window_voxels, ng);
    if (!safe_mul_size_t((size_t)ng, sizeof(double), &vectors_bytes_per_patch) ||
        !safe_mul_size_t(vectors_bytes_per_patch, 2u, &vectors_bytes_per_patch)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    chunk_batch = cap_chunk_batch_by_target(
        chunk_batch,
        vectors_bytes_per_patch,
        target_vectors_bytes
    );

    cudaError_t err = cudaSuccess;
    if (include_features) {
        cudaError_t include_err = cudaMalloc(
            &d_include_features,
            (size_t)FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT * sizeof(uint8_t)
        );
        if (include_err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            goto ngtdm_cleanup;
        }
        include_err = cudaMemcpyAsync(
            d_include_features,
            include_features,
            (size_t)FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (include_err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            goto ngtdm_cleanup;
        }
    }

    err = cudaMalloc(&d_distances, (size_t)num_distances * sizeof(int));
    if (err != cudaSuccess) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto ngtdm_cleanup;
    }
    err = cudaMemcpyAsync(
        d_distances,
        distances,
        (size_t)num_distances * sizeof(int),
        cudaMemcpyHostToDevice,
        ctx->stream
    );
    if (err != cudaSuccess) {
        status = FLASH_RADIOMICS_ERROR_IO_FAILED;
        goto ngtdm_cleanup;
    }

    while (chunk_batch > 0) {
        size_t chunk_voxels = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_vector_entries = 0;
        size_t chunk_vector_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;

        if (!safe_mul_size_t(chunk_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)ng, &chunk_vector_entries) ||
            !safe_mul_size_t(chunk_vector_entries, sizeof(double), &chunk_vector_bytes) ||
            !safe_mul_size_t(chunk_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            chunk_batch /= 2;
            continue;
        }

        err = cudaMalloc(&d_discretized, chunk_discretized_bytes);
        if (err != cudaSuccess) {
            d_discretized = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_mask, chunk_mask_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            d_discretized = NULL;
            d_mask = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_n_i, chunk_vector_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            d_discretized = NULL;
            d_mask = NULL;
            d_n_i = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_s_i, chunk_vector_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_n_i);
            d_discretized = NULL;
            d_mask = NULL;
            d_n_i = NULL;
            d_s_i = NULL;
            chunk_batch /= 2;
            continue;
        }
        err = cudaMalloc(&d_feature_rows, chunk_output_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_discretized);
            cudaFree(d_mask);
            cudaFree(d_n_i);
            cudaFree(d_s_i);
            d_discretized = NULL;
            d_mask = NULL;
            d_n_i = NULL;
            d_s_i = NULL;
            d_feature_rows = NULL;
            chunk_batch /= 2;
            continue;
        }
        allocated_chunk_discretized_bytes = chunk_discretized_bytes;
        allocated_chunk_mask_bytes = chunk_mask_bytes;
        allocated_chunk_vector_bytes = chunk_vector_bytes;
        allocated_chunk_output_bytes = chunk_output_bytes;
        break;
    }

    if (chunk_batch == 0) {
        status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        goto ngtdm_cleanup;
    }

    for (size_t batch_offset = 0; batch_offset < (size_t)batch_size; batch_offset += chunk_batch) {
        size_t current_batch = (size_t)batch_size - batch_offset;
        if (current_batch > chunk_batch) {
            current_batch = chunk_batch;
        }

        size_t chunk_voxels = 0;
        size_t chunk_discretized_bytes = 0;
        size_t chunk_mask_bytes = 0;
        size_t chunk_vector_entries = 0;
        size_t chunk_vector_bytes = 0;
        size_t chunk_output_rows = 0;
        size_t chunk_output_bytes = 0;

        if (!safe_mul_size_t(current_batch, spec.window_voxels, &chunk_voxels) ||
            !safe_mul_size_t(chunk_voxels, sizeof(int), &chunk_discretized_bytes) ||
            !safe_mul_size_t(chunk_voxels, sizeof(uint8_t), &chunk_mask_bytes) ||
            !safe_mul_size_t(current_batch, (size_t)ng, &chunk_vector_entries) ||
            !safe_mul_size_t(chunk_vector_entries, sizeof(double), &chunk_vector_bytes) ||
            !safe_mul_size_t(current_batch, (size_t)out_feature_stride, &chunk_output_rows) ||
            !safe_mul_size_t(chunk_output_rows, sizeof(double), &chunk_output_bytes)) {
            status = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
            break;
        }

        const int* chunk_discretized = discretized_windows + batch_offset * spec.window_voxels;
        const uint8_t* chunk_mask = mask_windows + batch_offset * spec.window_voxels;

        const std::chrono::steady_clock::time_point h2d_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            d_discretized,
            chunk_discretized,
            chunk_discretized_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaMemcpyAsync(
            d_mask,
            chunk_mask,
            chunk_mask_bytes,
            cudaMemcpyHostToDevice,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_h2d_ms += elapsed_ms(h2d_start, std::chrono::steady_clock::now());
        }

        const std::chrono::steady_clock::time_point device_build_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        if (use_patch_kernel) {
            if (current_batch > (size_t)INT_MAX) {
                status = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
                break;
            }
            const int patch_block_size = 32;
            build_ngtdm_vectors_batched_patch_kernel<<<
                (int)current_batch,
                patch_block_size,
                ngtdm_shared_bytes,
                ctx->stream
            >>>(
                d_discretized,
                d_mask,
                d_n_i,
                d_s_i,
                (int)current_batch,
                spec.window_voxels,
                spec.dims[0],
                spec.dims[1],
                spec.dims[2],
                spec.ndim,
                d_distances,
                num_distances,
                num_dirs,
                ng
            );
        } else {
            err = cudaMemsetAsync(d_n_i, 0, chunk_vector_bytes, ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            err = cudaMemsetAsync(d_s_i, 0, chunk_vector_bytes, ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }

            const int grid_size = compute_grid_size_1d(chunk_voxels, build_block_size, 4096);
            build_ngtdm_vectors_batched_kernel<<<grid_size, build_block_size, 0, ctx->stream>>>(
                d_discretized,
                d_mask,
                d_n_i,
                d_s_i,
                (int)current_batch,
                spec.window_voxels,
                spec.dims[0],
                spec.dims[1],
                spec.dims[2],
                spec.ndim,
                d_distances,
                num_distances,
                num_dirs,
                ng
            );
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }

        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_build_ms += elapsed_ms(
                device_build_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point device_finalize_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        const int finalize_block_size = choose_voxel_finalize_block_size(current_batch);
        const int finalize_grid_size =
            compute_grid_size_1d(current_batch, finalize_block_size, 4096);
        finalize_ngtdm_features_batched_kernel<<<finalize_grid_size, finalize_block_size, 0, ctx->stream>>>(
            d_n_i,
            d_s_i,
            (int)current_batch,
            ng,
            d_include_features,
            d_feature_rows,
            out_feature_stride
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            err = cudaStreamSynchronize(ctx->stream);
            if (err != cudaSuccess) {
                status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
                break;
            }
            total_device_finalize_ms += elapsed_ms(
                device_finalize_start,
                std::chrono::steady_clock::now()
            );
        }

        const std::chrono::steady_clock::time_point d2h_start =
            log_timing ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point();
        err = cudaMemcpyAsync(
            out_features + batch_offset * (size_t)out_feature_stride,
            d_feature_rows,
            chunk_output_bytes,
            cudaMemcpyDeviceToHost,
            ctx->stream
        );
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_IO_FAILED;
            break;
        }
        err = cudaStreamSynchronize(ctx->stream);
        if (err != cudaSuccess) {
            status = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
            break;
        }
        if (log_timing) {
            total_d2h_ms += elapsed_ms(d2h_start, std::chrono::steady_clock::now());
        }
    }

ngtdm_cleanup:
    if (d_distances) cudaFree(d_distances);
    if (d_discretized) cudaFree(d_discretized);
    if (d_mask) cudaFree(d_mask);
    if (d_n_i) cudaFree(d_n_i);
    if (d_s_i) cudaFree(d_s_i);
    if (d_include_features) cudaFree(d_include_features);
    if (d_feature_rows) cudaFree(d_feature_rows);
    if (log_timing) {
        log_voxel_cuda_stage_summary(
            "ngtdm",
            batch_size,
            chunk_batch,
            spec.window_voxels,
            allocated_chunk_output_bytes,
            allocated_chunk_discretized_bytes + allocated_chunk_mask_bytes,
            2u * allocated_chunk_vector_bytes + allocated_chunk_output_bytes + (size_t)num_distances * sizeof(int),
            0,
            total_h2d_ms,
            total_device_build_ms,
            total_device_finalize_ms,
            total_d2h_ms,
            total_host_finalize_ms
        );
    }
    return status;
}

#include "voxel_multi_class_cuda.h"
#include "glcm_cuda.h"
#include "glrlm_cuda.h"
#include "glszm_cuda.h"
#include "gldm_cuda.h"
#include "ngtdm_cuda.h"

extern "C" int flash_radiomics_voxel_multi_class_cuda_batch(
    /* Shared input */
    const int* discretized_windows,
    const uint8_t* mask_windows,
    int batch_size,
    int ndim,
    const int window_dims[3],
    int ng,
    unsigned int class_mask,

    /* Shared distance array (GLCM, GLDM, NGTDM) */
    const int* distances,
    int num_distances,

    /* GLDM-specific */
    double gldm_a,

    /* GLCM output */
    const uint8_t* glcm_include_features,
    double* glcm_out_features,
    int glcm_out_feature_stride,

    /* GLRLM output */
    const uint8_t* glrlm_include_features,
    double* glrlm_out_features,
    int glrlm_out_feature_stride,

    /* GLSZM output */
    const uint8_t* glszm_include_features,
    double* glszm_out_features,
    int glszm_out_feature_stride,

    /* GLDM output */
    const uint8_t* gldm_include_features,
    double* gldm_out_features,
    int gldm_out_feature_stride,

    /* NGTDM output */
    const uint8_t* ngtdm_include_features,
    double* ngtdm_out_features,
    int ngtdm_out_feature_stride
) {
    if (!discretized_windows || !mask_windows || batch_size <= 0 || ng <= 0) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    int status = FLASH_RADIOMICS_SUCCESS;

    /* Dispatch enabled texture classes sequentially. */

    if ((class_mask & FLASH_VOXEL_CLASS_GLCM) &&
        glcm_include_features && glcm_out_features && glcm_out_feature_stride > 0) {
        status = flash_radiomics_glcm_cuda_discretized_voxel_batch(
            discretized_windows, mask_windows, batch_size, ndim, window_dims,
            const_cast<int*>(distances), num_distances,
            glcm_include_features, ng,
            glcm_out_features, glcm_out_feature_stride
        );
        if (status != FLASH_RADIOMICS_SUCCESS) {
            return status;
        }
    }

    if ((class_mask & FLASH_VOXEL_CLASS_GLRLM) &&
        glrlm_include_features && glrlm_out_features && glrlm_out_feature_stride > 0) {
        status = flash_radiomics_glrlm_cuda_discretized_voxel_batch(
            discretized_windows, mask_windows, batch_size, ndim, window_dims,
            glrlm_include_features, ng,
            glrlm_out_features, glrlm_out_feature_stride
        );
        if (status != FLASH_RADIOMICS_SUCCESS) {
            return status;
        }
    }

    if ((class_mask & FLASH_VOXEL_CLASS_GLSZM) &&
        glszm_include_features && glszm_out_features && glszm_out_feature_stride > 0) {
        status = flash_radiomics_glszm_cuda_discretized_voxel_batch(
            discretized_windows, mask_windows, batch_size, ndim, window_dims,
            glszm_include_features, ng,
            glszm_out_features, glszm_out_feature_stride
        );
        if (status != FLASH_RADIOMICS_SUCCESS) {
            return status;
        }
    }

    if ((class_mask & FLASH_VOXEL_CLASS_GLDM) &&
        gldm_include_features && gldm_out_features && gldm_out_feature_stride > 0) {
        status = flash_radiomics_gldm_cuda_discretized_voxel_batch(
            discretized_windows, mask_windows, batch_size, ndim, window_dims,
            distances, num_distances,
            gldm_include_features, ng, gldm_a,
            gldm_out_features, gldm_out_feature_stride
        );
        if (status != FLASH_RADIOMICS_SUCCESS) {
            return status;
        }
    }

    if ((class_mask & FLASH_VOXEL_CLASS_NGTDM) &&
        ngtdm_include_features && ngtdm_out_features && ngtdm_out_feature_stride > 0) {
        status = flash_radiomics_ngtdm_cuda_discretized_voxel_batch(
            discretized_windows, mask_windows, batch_size, ndim, window_dims,
            distances, num_distances,
            ngtdm_include_features, ng,
            ngtdm_out_features, ngtdm_out_feature_stride
        );
        if (status != FLASH_RADIOMICS_SUCCESS) {
            return status;
        }
    }

    return FLASH_RADIOMICS_SUCCESS;
}
