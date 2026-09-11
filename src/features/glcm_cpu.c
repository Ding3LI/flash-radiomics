#include "glcm_cpu.h"
#include "../core/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define EPSILON 2.2e-16

// Direction vectors for 2D (4 directions: 0°, 45°, 90°, 135°)
static const int DIRECTIONS_2D[4][2] = {
    {1, 0},
    {1, 1},
    {0, 1},
    {-1, 1}
};

// Direction vectors for 3D (13 directions)
static const int DIRECTIONS_3D[13][3] = {
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

typedef struct {
    size_t linear_index;
    int x;
    int y;
    int z;
    int bin;
} GLCMRoiVoxel;

typedef struct {
    GLCMRoiVoxel* voxels;
    size_t count;
} GLCMRoi;

typedef struct {
    double* matrix;
    size_t* touched_positions;
    double* row_sums;
    double* px_add_y;
    double* px_sub_y;
    int* active_bins;
    double* q;
    double* p_active;
    double* v;
    double* u;
    double* x;
    double* w;
    double* y;
    int num_bins;
} GLCMWorkspace;

typedef struct {
    double autocorrelation;
    double joint_average;
    double cluster_prominence;
    double cluster_shade;
    double cluster_tendency;
    double contrast;
    double correlation;
    double difference_average;
    double difference_entropy;
    double difference_variance;
    double idm;
    double idmn;
    double id;
    double idn;
    double inverse_variance;
    double joint_energy;
    double joint_entropy;
    double imc1;
    double imc2;
    double mcc;
    double maximum_probability;
    double sum_average;
    double sum_entropy;
    double sum_squares;
} GLCMFeatureValues;

typedef enum {
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
} GLCMFeatureIndex;

static void free_glcm_workspace(GLCMWorkspace* workspace);

/*
 * PyRadiomics-compatible discretization for GLCM:
 * keep original bin indices (do not compact missing bins), so Ng is max bin index.
 */
static int* discretize_image_glcm(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    double bin_width,
    int bin_count,
    int* out_ng
) {
    if (!img || !mask || !out_ng) {
        return NULL;
    }

    size_t total_voxels = flash_radiomics_get_num_elements(img->ndim, img->dims);
    int* discretized = (int*)flash_radiomics_malloc(total_voxels * sizeof(int));
    if (!discretized) {
        return NULL;
    }

    double min_val = 0.0;
    double max_val = 0.0;
    int first = 1;
    for (size_t i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            float val = img->data[i];
            if (first) {
                min_val = max_val = val;
                first = 0;
            } else {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
    }
    if (first) {
        flash_radiomics_free(discretized);
        return NULL;
    }

    if (bin_count <= 0 && bin_width <= 0.0) {
        bin_width = 25.0;
    }

    int max_bin = 1;
    if (bin_count > 0) {
        double denom = max_val - min_val;
        for (size_t i = 0; i < total_voxels; i++) {
            if (mask->data[i] == 0) {
                discretized[i] = 0;
                continue;
            }
            float val = img->data[i];
            int bin = 1;
            if (denom > 0.0) {
                if (val < max_val) {
                    bin = (int)floor((double)bin_count * (val - min_val) / denom) + 1;
                } else {
                    bin = bin_count;
                }
            }
            discretized[i] = bin;
            if (bin > max_bin) max_bin = bin;
        }
    } else {
        double low_bound = floor(min_val / bin_width) * bin_width;
        for (size_t i = 0; i < total_voxels; i++) {
            if (mask->data[i] == 0) {
                discretized[i] = 0;
                continue;
            }
            float val = img->data[i];
            int bin = (int)floor((val - low_bound) / bin_width) + 1;
            if (bin < 1) bin = 1;
            discretized[i] = bin;
            if (bin > max_bin) max_bin = bin;
        }
    }

    *out_ng = max_bin;
    return discretized;
}

static void free_glcm_roi(GLCMRoi* roi) {
    if (!roi) {
        return;
    }
    flash_radiomics_free(roi->voxels);
    roi->voxels = NULL;
    roi->count = 0;
}

static int build_glcm_roi(
    const FlashRadiomicsImage* img,
    const FlashRadiomicsMask* mask,
    const int* discretized,
    int ng,
    GLCMRoi* roi
) {
    memset(roi, 0, sizeof(*roi));

    const int dims_x = img->dims[0];
    const int dims_y = img->dims[1];
    const int dims_z = (img->ndim == 2) ? 1 : img->dims[2];
    const size_t slice_stride = (size_t)dims_x * (size_t)dims_y;

    size_t roi_count = 0;
    if (img->ndim == 2) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                size_t idx = (size_t)y * (size_t)dims_x + (size_t)x;
                if (mask->data[idx] == 0) {
                    continue;
                }
                int bin = discretized[idx] - 1;
                if ((unsigned)bin >= (unsigned)ng) {
                    continue;
                }
                roi_count++;
            }
        }
    } else {
        for (int z = 0; z < dims_z; z++) {
            for (int y = 0; y < dims_y; y++) {
                for (int x = 0; x < dims_x; x++) {
                    size_t idx = (size_t)z * slice_stride + (size_t)y * (size_t)dims_x + (size_t)x;
                    if (mask->data[idx] == 0) {
                        continue;
                    }
                    int bin = discretized[idx] - 1;
                    if ((unsigned)bin >= (unsigned)ng) {
                        continue;
                    }
                    roi_count++;
                }
            }
        }
    }

    if (roi_count == 0) {
        return 1;
    }

    roi->voxels = (GLCMRoiVoxel*)flash_radiomics_malloc(roi_count * sizeof(GLCMRoiVoxel));
    if (!roi->voxels) {
        return 0;
    }

    size_t out_idx = 0;
    if (img->ndim == 2) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                size_t idx = (size_t)y * (size_t)dims_x + (size_t)x;
                if (mask->data[idx] == 0) {
                    continue;
                }
                int bin = discretized[idx] - 1;
                if ((unsigned)bin >= (unsigned)ng) {
                    continue;
                }
                roi->voxels[out_idx].linear_index = idx;
                roi->voxels[out_idx].x = x;
                roi->voxels[out_idx].y = y;
                roi->voxels[out_idx].z = 0;
                roi->voxels[out_idx].bin = bin;
                out_idx++;
            }
        }
    } else {
        for (int z = 0; z < dims_z; z++) {
            for (int y = 0; y < dims_y; y++) {
                for (int x = 0; x < dims_x; x++) {
                    size_t idx = (size_t)z * slice_stride + (size_t)y * (size_t)dims_x + (size_t)x;
                    if (mask->data[idx] == 0) {
                        continue;
                    }
                    int bin = discretized[idx] - 1;
                    if ((unsigned)bin >= (unsigned)ng) {
                        continue;
                    }
                    roi->voxels[out_idx].linear_index = idx;
                    roi->voxels[out_idx].x = x;
                    roi->voxels[out_idx].y = y;
                    roi->voxels[out_idx].z = z;
                    roi->voxels[out_idx].bin = bin;
                    out_idx++;
                }
            }
        }
    }

    roi->count = out_idx;
    return 1;
}

static int init_glcm_workspace(GLCMWorkspace* workspace, int ng, int include_mcc) {
    memset(workspace, 0, sizeof(*workspace));
    workspace->num_bins = ng;

    size_t matrix_elems = (size_t)ng * (size_t)ng;
    workspace->matrix = (double*)flash_radiomics_calloc(matrix_elems, sizeof(double));
    workspace->touched_positions = (size_t*)flash_radiomics_malloc(matrix_elems * sizeof(size_t));
    workspace->row_sums = (double*)flash_radiomics_calloc((size_t)ng, sizeof(double));
    workspace->px_add_y = (double*)flash_radiomics_calloc((size_t)(2 * ng - 1), sizeof(double));
    workspace->px_sub_y = (double*)flash_radiomics_calloc((size_t)ng, sizeof(double));
    workspace->active_bins = (int*)flash_radiomics_malloc((size_t)ng * sizeof(int));

    if (!workspace->matrix || !workspace->touched_positions || !workspace->row_sums ||
        !workspace->px_add_y || !workspace->px_sub_y || !workspace->active_bins) {
        free_glcm_workspace(workspace);
        return 0;
    }

    if (!include_mcc) {
        return 1;
    }

    workspace->q = (double*)flash_radiomics_malloc(matrix_elems * sizeof(double));
    workspace->p_active = (double*)flash_radiomics_malloc(matrix_elems * sizeof(double));
    workspace->v = (double*)flash_radiomics_malloc((size_t)ng * sizeof(double));
    workspace->u = (double*)flash_radiomics_malloc((size_t)ng * sizeof(double));
    workspace->x = (double*)flash_radiomics_malloc((size_t)ng * sizeof(double));
    workspace->w = (double*)flash_radiomics_malloc((size_t)ng * sizeof(double));
    workspace->y = (double*)flash_radiomics_malloc((size_t)ng * sizeof(double));

    if (!workspace->q || !workspace->p_active || !workspace->v || !workspace->u ||
        !workspace->x || !workspace->w || !workspace->y) {
        free_glcm_workspace(workspace);
        return 0;
    }

    return 1;
}

static void free_glcm_workspace(GLCMWorkspace* workspace) {
    if (!workspace) {
        return;
    }

    flash_radiomics_free(workspace->matrix);
    flash_radiomics_free(workspace->touched_positions);
    flash_radiomics_free(workspace->row_sums);
    flash_radiomics_free(workspace->px_add_y);
    flash_radiomics_free(workspace->px_sub_y);
    flash_radiomics_free(workspace->active_bins);
    flash_radiomics_free(workspace->q);
    flash_radiomics_free(workspace->p_active);
    flash_radiomics_free(workspace->v);
    flash_radiomics_free(workspace->u);
    flash_radiomics_free(workspace->x);
    flash_radiomics_free(workspace->w);
    flash_radiomics_free(workspace->y);
    memset(workspace, 0, sizeof(*workspace));
}

static inline void glcm_mark_active_bin(GLCMWorkspace* workspace, int* active_count, int bin) {
    if (workspace->row_sums[bin] == 0.0) {
        workspace->active_bins[(*active_count)++] = bin;
    }
}

static inline void glcm_touch_entry(GLCMWorkspace* workspace, size_t* touched_count, size_t pos) {
    if (workspace->matrix[pos] == 0.0) {
        workspace->touched_positions[(*touched_count)++] = pos;
    }
}

static double populate_glcm_matrix(
    const GLCMRoi* roi,
    const FlashRadiomicsMask* mask,
    const int* discretized,
    int ndim,
    const int dims[3],
    int dx,
    int dy,
    int dz,
    GLCMWorkspace* workspace,
    size_t* touched_count,
    int* active_count
) {
    const int dims_x = dims[0];
    const int dims_y = dims[1];
    const int dims_z = (ndim == 2) ? 1 : dims[2];
    const ptrdiff_t slice_stride = (ptrdiff_t)dims_x * (ptrdiff_t)dims_y;
    const ptrdiff_t neighbor_offset =
        (ptrdiff_t)dx +
        (ptrdiff_t)dy * (ptrdiff_t)dims_x +
        (ptrdiff_t)dz * slice_stride;

    const int min_x = (dx < 0) ? -dx : 0;
    const int max_x = (dx > 0) ? (dims_x - dx - 1) : (dims_x - 1);
    const int min_y = (dy < 0) ? -dy : 0;
    const int max_y = (dy > 0) ? (dims_y - dy - 1) : (dims_y - 1);
    const int min_z = (dz < 0) ? -dz : 0;
    const int max_z = (dz > 0) ? (dims_z - dz - 1) : (dims_z - 1);

    double pair_sum = 0.0;
    for (size_t idx = 0; idx < roi->count; idx++) {
        const GLCMRoiVoxel* voxel = &roi->voxels[idx];
        if (voxel->x < min_x || voxel->x > max_x ||
            voxel->y < min_y || voxel->y > max_y ||
            voxel->z < min_z || voxel->z > max_z) {
            continue;
        }

        ptrdiff_t neighbor_linear = (ptrdiff_t)voxel->linear_index + neighbor_offset;
        if (neighbor_linear < 0) {
            continue;
        }

        size_t neighbor_index = (size_t)neighbor_linear;
        if (mask->data[neighbor_index] == 0) {
            continue;
        }

        int i = voxel->bin;
        int j = discretized[neighbor_index] - 1;
        if ((unsigned)j >= (unsigned)workspace->num_bins) {
            continue;
        }

        glcm_mark_active_bin(workspace, active_count, i);
        workspace->row_sums[i] += 1.0;
        if (j == i) {
            workspace->row_sums[i] += 1.0;
        } else {
            glcm_mark_active_bin(workspace, active_count, j);
            workspace->row_sums[j] += 1.0;
        }

        size_t pos_ij = (size_t)i * (size_t)workspace->num_bins + (size_t)j;
        glcm_touch_entry(workspace, touched_count, pos_ij);
        workspace->matrix[pos_ij] += 1.0;

        size_t pos_ji = (size_t)j * (size_t)workspace->num_bins + (size_t)i;
        if (pos_ji == pos_ij) {
            workspace->matrix[pos_ij] += 1.0;
        } else {
            glcm_touch_entry(workspace, touched_count, pos_ji);
            workspace->matrix[pos_ji] += 1.0;
        }

        pair_sum += 2.0;
    }

    return pair_sum;
}

static void reset_glcm_workspace(GLCMWorkspace* workspace, size_t touched_count, int active_count) {
    for (size_t idx = 0; idx < touched_count; idx++) {
        workspace->matrix[workspace->touched_positions[idx]] = 0.0;
    }
    for (int idx = 0; idx < active_count; idx++) {
        workspace->row_sums[workspace->active_bins[idx]] = 0.0;
    }
}

static double dot_product(const double* a, const double* b, int n) {
    double out = 0.0;
    for (int i = 0; i < n; i++) {
        out += a[i] * b[i];
    }
    return out;
}

static double vector_norm(const double* a, int n) {
    return sqrt(dot_product(a, a, n));
}

static void matvec(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        const double* row = A + (size_t)i * (size_t)n;
        for (int j = 0; j < n; j++) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

static void matvec_transpose(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) {
            acc += A[(size_t)j * (size_t)n + (size_t)i] * x[j];
        }
        y[i] = acc;
    }
}

static double compute_mcc_active(
    GLCMWorkspace* workspace,
    int active_count,
    double inv_sum
) {
    if (!workspace->q || active_count < 2) {
        return 1.0;
    }

    for (int ai = 0; ai < active_count; ai++) {
        int i = workspace->active_bins[ai];
        double* row_out = workspace->p_active + (size_t)ai * (size_t)active_count;
        for (int aj = 0; aj < active_count; aj++) {
            int j = workspace->active_bins[aj];
            row_out[aj] = workspace->matrix[(size_t)i * (size_t)workspace->num_bins + (size_t)j] * inv_sum;
        }
    }

    for (int ai = 0; ai < active_count; ai++) {
        double px_i = workspace->row_sums[workspace->active_bins[ai]];
        const double* row_i = workspace->p_active + (size_t)ai * (size_t)active_count;
        double* q_row = workspace->q + (size_t)ai * (size_t)active_count;

        if (px_i <= 0.0) {
            for (int aj = 0; aj < active_count; aj++) {
                q_row[aj] = 0.0;
            }
            continue;
        }

        for (int aj = 0; aj < active_count; aj++) {
            const double* row_j = workspace->p_active + (size_t)aj * (size_t)active_count;
            double acc = 0.0;
            for (int ak = 0; ak < active_count; ak++) {
                double px_k = workspace->row_sums[workspace->active_bins[ak]];
                double denom = px_i * px_k + EPSILON;
                acc += (row_i[ak] * row_j[ak]) / denom;
            }
            q_row[aj] = acc;
        }
    }

    const int max_iter = 200;
    const double tol = 1e-10;
    const double init = 1.0 / sqrt((double)active_count);

    for (int i = 0; i < active_count; i++) {
        workspace->v[i] = init;
        workspace->u[i] = init;
        workspace->x[i] = init;
    }

    double lambda1 = 0.0;
    double prev = 0.0;
    for (int it = 0; it < max_iter; it++) {
        matvec(workspace->q, workspace->v, workspace->w, active_count);
        double nrm = vector_norm(workspace->w, active_count);
        if (nrm <= EPSILON) {
            break;
        }
        for (int i = 0; i < active_count; i++) {
            workspace->v[i] = workspace->w[i] / nrm;
        }
        matvec(workspace->q, workspace->v, workspace->y, active_count);
        lambda1 = dot_product(workspace->v, workspace->y, active_count);
        if (fabs(lambda1 - prev) < tol) {
            break;
        }
        prev = lambda1;
    }

    prev = 0.0;
    for (int it = 0; it < max_iter; it++) {
        matvec_transpose(workspace->q, workspace->u, workspace->w, active_count);
        double nrm = vector_norm(workspace->w, active_count);
        if (nrm <= EPSILON) {
            break;
        }
        for (int i = 0; i < active_count; i++) {
            workspace->u[i] = workspace->w[i] / nrm;
        }
        matvec_transpose(workspace->q, workspace->u, workspace->y, active_count);
        double lambda_left = dot_product(workspace->u, workspace->y, active_count);
        if (fabs(lambda_left - prev) < tol) {
            break;
        }
        prev = lambda_left;
    }

    double uv = dot_product(workspace->u, workspace->v, active_count);
    if (fabs(uv) <= EPSILON) {
        return 0.0;
    }
    for (int i = 0; i < active_count; i++) {
        workspace->u[i] /= uv;
    }

    double lambda2 = 0.0;
    prev = 0.0;
    for (int it = 0; it < max_iter; it++) {
        matvec(workspace->q, workspace->x, workspace->w, active_count);
        double ux = dot_product(workspace->u, workspace->x, active_count);
        for (int i = 0; i < active_count; i++) {
            workspace->w[i] -= lambda1 * workspace->v[i] * ux;
        }
        double nrm = vector_norm(workspace->w, active_count);
        if (nrm <= EPSILON) {
            break;
        }
        for (int i = 0; i < active_count; i++) {
            workspace->x[i] = workspace->w[i] / nrm;
        }

        matvec(workspace->q, workspace->x, workspace->y, active_count);
        ux = dot_product(workspace->u, workspace->x, active_count);
        for (int i = 0; i < active_count; i++) {
            workspace->y[i] -= lambda1 * workspace->v[i] * ux;
        }
        lambda2 = dot_product(workspace->x, workspace->y, active_count);
        if (fabs(lambda2 - prev) < tol) {
            break;
        }
        prev = lambda2;
    }

    if (lambda2 <= 0.0) {
        return 0.0;
    }
    return sqrt(lambda2);
}

static void compute_glcm_features(
    GLCMWorkspace* workspace,
    size_t touched_count,
    int active_count,
    double pair_sum,
    int include_mcc,
    GLCMFeatureValues* out
) {
    memset(out, 0, sizeof(*out));

    const int ng = workspace->num_bins;
    const double inv_sum = 1.0 / pair_sum;

    memset(workspace->px_add_y, 0, (size_t)(2 * ng - 1) * sizeof(double));
    memset(workspace->px_sub_y, 0, (size_t)ng * sizeof(double));

    for (size_t idx = 0; idx < touched_count; idx++) {
        size_t pos = workspace->touched_positions[idx];
        int i = (int)(pos / (size_t)ng);
        int j = (int)(pos % (size_t)ng);
        double p = workspace->matrix[pos] * inv_sum;
        double gray_i = (double)(i + 1);
        double gray_j = (double)(j + 1);
        double diff = (double)abs(i - j);

        workspace->px_add_y[i + j] += p;
        workspace->px_sub_y[abs(i - j)] += p;
        out->autocorrelation += p * gray_i * gray_j;
        out->contrast += p * diff * diff;
        out->joint_energy += p * p;
        out->joint_entropy -= p * log2(p + EPSILON);
        if (p > out->maximum_probability) {
            out->maximum_probability = p;
        }
    }

    double joint_average = 0.0;
    double hx = 0.0;
    for (int idx = 0; idx < active_count; idx++) {
        int bin = workspace->active_bins[idx];
        double p = workspace->row_sums[bin] * inv_sum;
        workspace->row_sums[bin] = p;
        joint_average += (double)(bin + 1) * p;
        if (p > 0.0) {
            hx -= p * log2(p + EPSILON);
        }
    }
    out->joint_average = joint_average;

    double variance = 0.0;
    for (int idx = 0; idx < active_count; idx++) {
        int bin = workspace->active_bins[idx];
        double centered = (double)(bin + 1) - joint_average;
        variance += workspace->row_sums[bin] * centered * centered;
    }

    double hxy1 = 0.0;
    double correlation_num = 0.0;
    for (size_t idx = 0; idx < touched_count; idx++) {
        size_t pos = workspace->touched_positions[idx];
        int i = (int)(pos / (size_t)ng);
        int j = (int)(pos % (size_t)ng);
        double p = workspace->matrix[pos] * inv_sum;
        double gray_i = (double)(i + 1);
        double gray_j = (double)(j + 1);
        double centered_i = gray_i - joint_average;
        double centered_j = gray_j - joint_average;
        double cluster = gray_i + gray_j - 2.0 * joint_average;
        double cluster_sq = cluster * cluster;
        double pxy = workspace->row_sums[i] * workspace->row_sums[j];

        out->sum_squares += p * centered_i * centered_i;
        correlation_num += p * centered_i * centered_j;
        out->cluster_tendency += p * cluster_sq;
        out->cluster_shade += p * cluster_sq * cluster;
        out->cluster_prominence += p * cluster_sq * cluster_sq;
        hxy1 -= p * log2(pxy + EPSILON);
    }

    if (variance <= 0.0) {
        out->correlation = 1.0;
    } else {
        out->correlation = correlation_num / variance;
    }

    double hxy2 = 0.0;
    for (int row_idx = 0; row_idx < active_count; row_idx++) {
        double px_i = workspace->row_sums[workspace->active_bins[row_idx]];
        for (int col_idx = 0; col_idx < active_count; col_idx++) {
            double pxy = px_i * workspace->row_sums[workspace->active_bins[col_idx]];
            if (pxy > 0.0) {
                hxy2 -= pxy * log2(pxy + EPSILON);
            }
        }
    }

    const double ng_sq = (double)ng * (double)ng;
    for (int k = 0; k < ng; k++) {
        double px_sub_y = workspace->px_sub_y[k];
        double k_d = (double)k;
        double k_sq = k_d * k_d;

        out->difference_average += k_d * px_sub_y;
        out->difference_variance += k_sq * px_sub_y;
        out->idm += px_sub_y / (1.0 + k_sq);
        out->idmn += px_sub_y / (1.0 + k_sq / ng_sq);
        out->id += px_sub_y / (1.0 + k_d);
        out->idn += px_sub_y / (1.0 + k_d / (double)ng);
        if (k > 0) {
            out->inverse_variance += px_sub_y / k_sq;
        }
        if (px_sub_y > 0.0) {
            out->difference_entropy -= px_sub_y * log2(px_sub_y + EPSILON);
        }
    }
    out->difference_variance -= out->difference_average * out->difference_average;

    for (int k = 0; k < 2 * ng - 1; k++) {
        double px_add_y = workspace->px_add_y[k];
        out->sum_average += (double)(k + 2) * px_add_y;
        if (px_add_y > 0.0) {
            out->sum_entropy -= px_add_y * log2(px_add_y + EPSILON);
        }
    }

    double max_h = hx;
    out->imc1 = (max_h == 0.0) ? 0.0 : (out->joint_entropy - hxy1) / max_h;
    out->imc2 = (hxy2 <= out->joint_entropy) ? 0.0 : sqrt(1.0 - exp(-2.0 * (hxy2 - out->joint_entropy)));
    out->mcc = include_mcc ? compute_mcc_active(workspace, active_count, inv_sum) : 0.0;
}

static void accumulate_glcm_features(GLCMFeatureValues* total, const GLCMFeatureValues* current) {
    total->autocorrelation += current->autocorrelation;
    total->joint_average += current->joint_average;
    total->cluster_prominence += current->cluster_prominence;
    total->cluster_shade += current->cluster_shade;
    total->cluster_tendency += current->cluster_tendency;
    total->contrast += current->contrast;
    total->correlation += current->correlation;
    total->difference_average += current->difference_average;
    total->difference_entropy += current->difference_entropy;
    total->difference_variance += current->difference_variance;
    total->idm += current->idm;
    total->idmn += current->idmn;
    total->id += current->id;
    total->idn += current->idn;
    total->inverse_variance += current->inverse_variance;
    total->joint_energy += current->joint_energy;
    total->joint_entropy += current->joint_entropy;
    total->imc1 += current->imc1;
    total->imc2 += current->imc2;
    total->mcc += current->mcc;
    total->maximum_probability += current->maximum_probability;
    total->sum_average += current->sum_average;
    total->sum_entropy += current->sum_entropy;
    total->sum_squares += current->sum_squares;
}

static void average_glcm_features(GLCMFeatureValues* values, int valid_matrices, int include_mcc) {
    const double denom = (double)valid_matrices;
    values->autocorrelation /= denom;
    values->joint_average /= denom;
    values->cluster_prominence /= denom;
    values->cluster_shade /= denom;
    values->cluster_tendency /= denom;
    values->contrast /= denom;
    values->correlation /= denom;
    values->difference_average /= denom;
    values->difference_entropy /= denom;
    values->difference_variance /= denom;
    values->idm /= denom;
    values->idmn /= denom;
    values->id /= denom;
    values->idn /= denom;
    values->inverse_variance /= denom;
    values->joint_energy /= denom;
    values->joint_entropy /= denom;
    values->imc1 /= denom;
    values->imc2 /= denom;
    if (include_mcc) {
        values->mcc /= denom;
    }
    values->maximum_probability /= denom;
    values->sum_average /= denom;
    values->sum_entropy /= denom;
    values->sum_squares /= denom;
}

static void nan_glcm_features(GLCMFeatureValues* values, int include_mcc) {
    values->autocorrelation = NAN;
    values->joint_average = NAN;
    values->cluster_prominence = NAN;
    values->cluster_shade = NAN;
    values->cluster_tendency = NAN;
    values->contrast = NAN;
    values->correlation = NAN;
    values->difference_average = NAN;
    values->difference_entropy = NAN;
    values->difference_variance = NAN;
    values->idm = NAN;
    values->idmn = NAN;
    values->id = NAN;
    values->idn = NAN;
    values->inverse_variance = NAN;
    values->joint_energy = NAN;
    values->joint_entropy = NAN;
    values->imc1 = NAN;
    values->imc2 = NAN;
    values->mcc = include_mcc ? NAN : 0.0;
    values->maximum_probability = NAN;
    values->sum_average = NAN;
    values->sum_entropy = NAN;
    values->sum_squares = NAN;
}

static double populate_glcm_matrix_dense_masked(
    const uint8_t* mask,
    const int* discretized,
    int ndim,
    const int dims[3],
    int dx,
    int dy,
    int dz,
    GLCMWorkspace* workspace,
    size_t* touched_count,
    int* active_count
) {
    const int dims_x = dims[0];
    const int dims_y = dims[1];
    const int dims_z = (ndim == 2) ? 1 : dims[2];
    const ptrdiff_t slice_stride = (ptrdiff_t)dims_x * (ptrdiff_t)dims_y;

    double pair_sum = 0.0;
    for (int z = 0; z < dims_z; z++) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                const int nx = x + dx;
                const int ny = y + dy;
                const int nz = z + dz;
                if (nx < 0 || nx >= dims_x ||
                    ny < 0 || ny >= dims_y ||
                    nz < 0 || nz >= dims_z) {
                    continue;
                }

                size_t linear_index = (size_t)z * (size_t)slice_stride + (size_t)y * (size_t)dims_x + (size_t)x;
                if (mask[linear_index] == 0) {
                    continue;
                }

                int i = discretized[linear_index] - 1;
                if ((unsigned)i >= (unsigned)workspace->num_bins) {
                    continue;
                }

                size_t neighbor_index = (size_t)nz * (size_t)slice_stride + (size_t)ny * (size_t)dims_x + (size_t)nx;
                if (mask[neighbor_index] == 0) {
                    continue;
                }

                int j = discretized[neighbor_index] - 1;
                if ((unsigned)j >= (unsigned)workspace->num_bins) {
                    continue;
                }

                glcm_mark_active_bin(workspace, active_count, i);
                workspace->row_sums[i] += 1.0;
                if (j == i) {
                    workspace->row_sums[i] += 1.0;
                } else {
                    glcm_mark_active_bin(workspace, active_count, j);
                    workspace->row_sums[j] += 1.0;
                }

                size_t pos_ij = (size_t)i * (size_t)workspace->num_bins + (size_t)j;
                glcm_touch_entry(workspace, touched_count, pos_ij);
                workspace->matrix[pos_ij] += 1.0;

                size_t pos_ji = (size_t)j * (size_t)workspace->num_bins + (size_t)i;
                if (pos_ji == pos_ij) {
                    workspace->matrix[pos_ij] += 1.0;
                } else {
                    glcm_touch_entry(workspace, touched_count, pos_ji);
                    workspace->matrix[pos_ji] += 1.0;
                }

                pair_sum += 2.0;
            }
        }
    }

    return pair_sum;
}

static int glcm_has_full_3x3x3_support(
    const uint8_t* mask,
    const int* discretized,
    int ng
) {
    for (int idx = 0; idx < 27; idx++) {
        if (mask[idx] == 0) {
            return 0;
        }
        int bin = discretized[idx];
        if (bin <= 0 || bin > ng) {
            return 0;
        }
    }
    return 1;
}

static double populate_glcm_matrix_full_3x3x3(
    const int* discretized,
    int dx,
    int dy,
    int dz,
    GLCMWorkspace* workspace,
    size_t* touched_count,
    int* active_count
) {
    const int min_x = (dx < 0) ? -dx : 0;
    const int max_x = (dx > 0) ? (2 - dx) : 2;
    const int min_y = (dy < 0) ? -dy : 0;
    const int max_y = (dy > 0) ? (2 - dy) : 2;
    const int min_z = (dz < 0) ? -dz : 0;
    const int max_z = (dz > 0) ? (2 - dz) : 2;
    const ptrdiff_t neighbor_offset = (ptrdiff_t)dx + (ptrdiff_t)dy * 3 + (ptrdiff_t)dz * 9;

    double pair_sum = 0.0;
    for (int z = min_z; z <= max_z; z++) {
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                size_t linear_index = (size_t)z * 9 + (size_t)y * 3 + (size_t)x;
                size_t neighbor_index = (size_t)((ptrdiff_t)linear_index + neighbor_offset);

                int i = discretized[linear_index] - 1;
                int j = discretized[neighbor_index] - 1;

                glcm_mark_active_bin(workspace, active_count, i);
                workspace->row_sums[i] += 1.0;
                if (j == i) {
                    workspace->row_sums[i] += 1.0;
                } else {
                    glcm_mark_active_bin(workspace, active_count, j);
                    workspace->row_sums[j] += 1.0;
                }

                size_t pos_ij = (size_t)i * (size_t)workspace->num_bins + (size_t)j;
                glcm_touch_entry(workspace, touched_count, pos_ij);
                workspace->matrix[pos_ij] += 1.0;

                size_t pos_ji = (size_t)j * (size_t)workspace->num_bins + (size_t)i;
                if (pos_ji == pos_ij) {
                    workspace->matrix[pos_ij] += 1.0;
                } else {
                    glcm_touch_entry(workspace, touched_count, pos_ji);
                    workspace->matrix[pos_ji] += 1.0;
                }

                pair_sum += 2.0;
            }
        }
    }

    return pair_sum;
}

static void jacobi_eigenvalues_symmetric(double* matrix, int n, double* eigenvalues) {
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
            const double* row = matrix + (size_t)i * (size_t)n;
            for (int j = i + 1; j < n; j++) {
                double magnitude = fabs(row[j]);
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

        const size_t pn = (size_t)p * (size_t)n;
        const size_t qn = (size_t)q * (size_t)n;
        const double app = matrix[pn + (size_t)p];
        const double aqq = matrix[qn + (size_t)q];
        const double apq = matrix[pn + (size_t)q];
        if (fabs(apq) <= tol) {
            continue;
        }

        const double tau = (aqq - app) / (2.0 * apq);
        const double t = ((tau >= 0.0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
        const double c = 1.0 / sqrt(1.0 + t * t);
        const double s = t * c;

        for (int k = 0; k < n; k++) {
            if (k == p || k == q) {
                continue;
            }

            const size_t kn = (size_t)k * (size_t)n;
            const double akp = matrix[kn + (size_t)p];
            const double akq = matrix[kn + (size_t)q];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;

            matrix[kn + (size_t)p] = new_kp;
            matrix[pn + (size_t)k] = new_kp;
            matrix[kn + (size_t)q] = new_kq;
            matrix[qn + (size_t)k] = new_kq;
        }

        matrix[pn + (size_t)p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[qn + (size_t)q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[pn + (size_t)q] = 0.0;
        matrix[qn + (size_t)p] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = matrix[(size_t)i * (size_t)n + (size_t)i];
    }
}

static double compute_mcc_exact_active(
    GLCMWorkspace* workspace,
    int active_count,
    double pair_sum
) {
    if (!workspace || active_count < 2 || pair_sum <= 0.0) {
        return 1.0;
    }
    if (!workspace->q || !workspace->p_active || !workspace->v || !workspace->w) {
        return 1.0;
    }

    const double inv_sum = 1.0 / pair_sum;
    for (int ai = 0; ai < active_count; ai++) {
        int bin_i = workspace->active_bins[ai];
        workspace->v[ai] = workspace->row_sums[bin_i] * inv_sum;
        double* p_row = workspace->p_active + (size_t)ai * (size_t)active_count;
        const double* matrix_row = workspace->matrix + (size_t)bin_i * (size_t)workspace->num_bins;
        for (int ak = 0; ak < active_count; ak++) {
            int bin_k = workspace->active_bins[ak];
            p_row[ak] = matrix_row[bin_k] * inv_sum;
        }
    }

    for (int ai = 0; ai < active_count; ai++) {
        const double px_i = workspace->v[ai];
        const double* p_i = workspace->p_active + (size_t)ai * (size_t)active_count;
        for (int aj = ai; aj < active_count; aj++) {
            const double px_j = workspace->v[aj];
            const double* p_j = workspace->p_active + (size_t)aj * (size_t)active_count;
            double acc = 0.0;
            for (int ak = 0; ak < active_count; ak++) {
                acc += (p_i[ak] * p_j[ak]) / (workspace->v[ak] + EPSILON);
            }
            const double denom = sqrt((px_i + EPSILON) * (px_j + EPSILON));
            double value = acc / denom;
            if (!isfinite(value)) {
                value = 0.0;
            }
            workspace->q[(size_t)ai * (size_t)active_count + (size_t)aj] = value;
            workspace->q[(size_t)aj * (size_t)active_count + (size_t)ai] = value;
        }
    }

    jacobi_eigenvalues_symmetric(workspace->q, active_count, workspace->w);
    for (int i = 1; i < active_count; i++) {
        double key = workspace->w[i];
        int j = i - 1;
        while (j >= 0 && workspace->w[j] > key) {
            workspace->w[j + 1] = workspace->w[j];
            j--;
        }
        workspace->w[j + 1] = key;
    }

    double second_eig = workspace->w[active_count - 2];
    if (!isfinite(second_eig)) {
        return NAN;
    }
    if (second_eig < 0.0 && second_eig > -1e-12) {
        second_eig = 0.0;
    }
    if (second_eig <= 0.0) {
        return 0.0;
    }
    return sqrt(second_eig);
}

static int compute_glcm_patch_feature_values(
    const int* discretized,
    const uint8_t* mask,
    int ndim,
    const int dims[3],
    int* distances,
    int num_distances,
    int ng,
    int include_mcc,
    GLCMWorkspace* workspace,
    GLCMFeatureValues* out_values
) {
    if (!discretized || !mask || !dims || !distances || !workspace || !out_values) {
        return 0;
    }
    if (num_distances <= 0 || ng <= 0) {
        nan_glcm_features(out_values, include_mcc);
        return 1;
    }

    GLCMFeatureValues totals;
    memset(&totals, 0, sizeof(totals));

    int valid_matrices = 0;
    const int num_directions = (ndim == 2) ? 4 : 13;
    const int use_fast_full_3x3x3 =
        (ndim == 3 && dims[0] == 3 && dims[1] == 3 && dims[2] == 3 &&
         glcm_has_full_3x3x3_support(mask, discretized, ng));

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
                dx = DIRECTIONS_2D[dir_idx][0] * distance;
                dy = DIRECTIONS_2D[dir_idx][1] * distance;
            } else {
                dx = DIRECTIONS_3D[dir_idx][0] * distance;
                dy = DIRECTIONS_3D[dir_idx][1] * distance;
                dz = DIRECTIONS_3D[dir_idx][2] * distance;
            }

            size_t touched_count = 0;
            int active_count = 0;
            double pair_sum = 0.0;
            if (use_fast_full_3x3x3 && distance == 1) {
                pair_sum = populate_glcm_matrix_full_3x3x3(
                    discretized,
                    dx,
                    dy,
                    dz,
                    workspace,
                    &touched_count,
                    &active_count
                );
            } else {
                pair_sum = populate_glcm_matrix_dense_masked(
                    mask,
                    discretized,
                    ndim,
                    dims,
                    dx,
                    dy,
                    dz,
                    workspace,
                    &touched_count,
                    &active_count
                );
            }

            if (pair_sum > 0.0) {
                GLCMFeatureValues current;
                double mcc = NAN;
                if (include_mcc) {
                    mcc = compute_mcc_exact_active(workspace, active_count, pair_sum);
                }
                compute_glcm_features(
                    workspace,
                    touched_count,
                    active_count,
                    pair_sum,
                    0,
                    &current
                );
                current.mcc = include_mcc ? mcc : NAN;
                accumulate_glcm_features(&totals, &current);
                valid_matrices++;
            }

            reset_glcm_workspace(workspace, touched_count, active_count);
        }
    }

    if (valid_matrices > 0) {
        average_glcm_features(&totals, valid_matrices, include_mcc);
        if (!include_mcc) {
            totals.mcc = NAN;
        }
    } else {
        nan_glcm_features(&totals, include_mcc);
    }

    *out_values = totals;
    return 1;
}

static void write_glcm_feature_row(
    const GLCMFeatureValues* values,
    double* out_row
) {
    out_row[GLCM_FEATURE_AUTOCORRELATION] = values->autocorrelation;
    out_row[GLCM_FEATURE_JOINT_AVERAGE] = values->joint_average;
    out_row[GLCM_FEATURE_CLUSTER_PROMINENCE] = values->cluster_prominence;
    out_row[GLCM_FEATURE_CLUSTER_SHADE] = values->cluster_shade;
    out_row[GLCM_FEATURE_CLUSTER_TENDENCY] = values->cluster_tendency;
    out_row[GLCM_FEATURE_CONTRAST] = values->contrast;
    out_row[GLCM_FEATURE_CORRELATION] = values->correlation;
    out_row[GLCM_FEATURE_DIFFERENCE_AVERAGE] = values->difference_average;
    out_row[GLCM_FEATURE_DIFFERENCE_ENTROPY] = values->difference_entropy;
    out_row[GLCM_FEATURE_DIFFERENCE_VARIANCE] = values->difference_variance;
    out_row[GLCM_FEATURE_JOINT_ENERGY] = values->joint_energy;
    out_row[GLCM_FEATURE_JOINT_ENTROPY] = values->joint_entropy;
    out_row[GLCM_FEATURE_IMC1] = values->imc1;
    out_row[GLCM_FEATURE_IMC2] = values->imc2;
    out_row[GLCM_FEATURE_IDM] = values->idm;
    out_row[GLCM_FEATURE_MCC] = values->mcc;
    out_row[GLCM_FEATURE_IDMN] = values->idmn;
    out_row[GLCM_FEATURE_ID] = values->id;
    out_row[GLCM_FEATURE_IDN] = values->idn;
    out_row[GLCM_FEATURE_INVERSE_VARIANCE] = values->inverse_variance;
    out_row[GLCM_FEATURE_MAXIMUM_PROBABILITY] = values->maximum_probability;
    out_row[GLCM_FEATURE_SUM_AVERAGE] = values->sum_average;
    out_row[GLCM_FEATURE_SUM_ENTROPY] = values->sum_entropy;
    out_row[GLCM_FEATURE_SUM_SQUARES] = values->sum_squares;
}

static int is_glcm_feature_enabled(const uint8_t* include_features, int idx) {
    return (!include_features) || (include_features[idx] != 0);
}

static FlashRadiomicsResult* flash_radiomics_glcm_cpu_internal(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    const int* discretized_input,
    int ng_input,
    int include_mcc
) {
    clock_t start_time = clock();

    FlashRadiomicsResult* result = (FlashRadiomicsResult*)malloc(sizeof(FlashRadiomicsResult));
    if (!result) {
        return NULL;
    }

    result->features = NULL;
    result->count = 0;
    result->compute_time_ms = 0.0;
    result->error_code = FLASH_RADIOMICS_SUCCESS;
    result->error_msg[0] = '\0';

    if (!img || !mask) {
        result->error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Invalid input: image or mask is NULL");
        return result;
    }

    if (img->ndim != mask->ndim) {
        result->error_code = FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Dimension mismatch: image ndim=%d, mask ndim=%d",
                 img->ndim, mask->ndim);
        return result;
    }

    for (int i = 0; i < 3; i++) {
        if (img->dims[i] != mask->dims[i]) {
            result->error_code = FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH;
            snprintf(result->error_msg, sizeof(result->error_msg),
                     "Dimension mismatch at axis %d: image=%d, mask=%d",
                     i, img->dims[i], mask->dims[i]);
            return result;
        }
    }

    if (!distances || num_distances <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Invalid distances parameter");
        return result;
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif

    int ng = ng_input;
    int* discretized_owned = NULL;
    const int* discretized = discretized_input;
    if (!discretized || ng <= 0) {
        discretized_owned = discretize_image_glcm(img, mask, bin_width, bin_count, &ng);
        discretized = discretized_owned;
    }
    if (!discretized || ng <= 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Failed to discretize image for GLCM");
        flash_radiomics_free(discretized_owned);
        return result;
    }

    GLCMRoi roi;
    if (!build_glcm_roi(img, mask, discretized, ng, &roi)) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Failed to allocate ROI voxel list for GLCM");
        flash_radiomics_free(discretized_owned);
        return result;
    }

    if (roi.count == 0) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Mask contains no discretized voxels for GLCM");
        free_glcm_roi(&roi);
        flash_radiomics_free(discretized_owned);
        return result;
    }

    GLCMWorkspace workspace;
    if (!init_glcm_workspace(&workspace, ng, include_mcc)) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Failed to allocate GLCM workspace");
        free_glcm_roi(&roi);
        flash_radiomics_free(discretized_owned);
        return result;
    }

    result->count = include_mcc ? 24 : 23;
    result->features = (FlashRadiomicsFeature*)malloc((size_t)result->count * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg),
                 "Failed to allocate feature array");
        free_glcm_workspace(&workspace);
        free_glcm_roi(&roi);
        flash_radiomics_free(discretized_owned);
        return result;
    }

    GLCMFeatureValues totals;
    memset(&totals, 0, sizeof(totals));

    int num_directions = (img->ndim == 2) ? 4 : 13;
    int valid_matrices = 0;
    for (int distance_idx = 0; distance_idx < num_distances; distance_idx++) {
        int distance = distances[distance_idx];
        if (distance <= 0) {
            continue;
        }

        for (int dir_idx = 0; dir_idx < num_directions; dir_idx++) {
            int dx;
            int dy;
            int dz = 0;
            if (img->ndim == 2) {
                dx = DIRECTIONS_2D[dir_idx][0] * distance;
                dy = DIRECTIONS_2D[dir_idx][1] * distance;
            } else {
                dx = DIRECTIONS_3D[dir_idx][0] * distance;
                dy = DIRECTIONS_3D[dir_idx][1] * distance;
                dz = DIRECTIONS_3D[dir_idx][2] * distance;
            }

            size_t touched_count = 0;
            int active_count = 0;
            double pair_sum = populate_glcm_matrix(
                &roi,
                mask,
                discretized,
                img->ndim,
                img->dims,
                dx,
                dy,
                dz,
                &workspace,
                &touched_count,
                &active_count
            );

            if (pair_sum > 0.0) {
                GLCMFeatureValues current;
                compute_glcm_features(
                    &workspace,
                    touched_count,
                    active_count,
                    pair_sum,
                    include_mcc,
                    &current
                );
                accumulate_glcm_features(&totals, &current);
                valid_matrices++;
            }

            reset_glcm_workspace(&workspace, touched_count, active_count);
        }
    }

    if (valid_matrices > 0) {
        average_glcm_features(&totals, valid_matrices, include_mcc);
    } else {
        nan_glcm_features(&totals, include_mcc);
    }

    int idx = 0;
    snprintf(result->features[idx].name, 64, "glcm_Autocorrelation");
    result->features[idx++].value = totals.autocorrelation;

    snprintf(result->features[idx].name, 64, "glcm_JointAverage");
    result->features[idx++].value = totals.joint_average;

    snprintf(result->features[idx].name, 64, "glcm_ClusterProminence");
    result->features[idx++].value = totals.cluster_prominence;

    snprintf(result->features[idx].name, 64, "glcm_ClusterShade");
    result->features[idx++].value = totals.cluster_shade;

    snprintf(result->features[idx].name, 64, "glcm_ClusterTendency");
    result->features[idx++].value = totals.cluster_tendency;

    snprintf(result->features[idx].name, 64, "glcm_Contrast");
    result->features[idx++].value = totals.contrast;

    snprintf(result->features[idx].name, 64, "glcm_Correlation");
    result->features[idx++].value = totals.correlation;

    snprintf(result->features[idx].name, 64, "glcm_DifferenceAverage");
    result->features[idx++].value = totals.difference_average;

    snprintf(result->features[idx].name, 64, "glcm_DifferenceEntropy");
    result->features[idx++].value = totals.difference_entropy;

    snprintf(result->features[idx].name, 64, "glcm_DifferenceVariance");
    result->features[idx++].value = totals.difference_variance;

    snprintf(result->features[idx].name, 64, "glcm_JointEnergy");
    result->features[idx++].value = totals.joint_energy;

    snprintf(result->features[idx].name, 64, "glcm_JointEntropy");
    result->features[idx++].value = totals.joint_entropy;

    snprintf(result->features[idx].name, 64, "glcm_Imc1");
    result->features[idx++].value = totals.imc1;

    snprintf(result->features[idx].name, 64, "glcm_Imc2");
    result->features[idx++].value = totals.imc2;

    snprintf(result->features[idx].name, 64, "glcm_Idm");
    result->features[idx++].value = totals.idm;

    if (include_mcc) {
        snprintf(result->features[idx].name, 64, "glcm_MCC");
        result->features[idx++].value = totals.mcc;
    }

    snprintf(result->features[idx].name, 64, "glcm_Idmn");
    result->features[idx++].value = totals.idmn;

    snprintf(result->features[idx].name, 64, "glcm_Id");
    result->features[idx++].value = totals.id;

    snprintf(result->features[idx].name, 64, "glcm_Idn");
    result->features[idx++].value = totals.idn;

    snprintf(result->features[idx].name, 64, "glcm_InverseVariance");
    result->features[idx++].value = totals.inverse_variance;

    snprintf(result->features[idx].name, 64, "glcm_MaximumProbability");
    result->features[idx++].value = totals.maximum_probability;

    snprintf(result->features[idx].name, 64, "glcm_SumAverage");
    result->features[idx++].value = totals.sum_average;

    snprintf(result->features[idx].name, 64, "glcm_SumEntropy");
    result->features[idx++].value = totals.sum_entropy;

    snprintf(result->features[idx].name, 64, "glcm_SumSquares");
    result->features[idx++].value = totals.sum_squares;

    free_glcm_workspace(&workspace);
    free_glcm_roi(&roi);
    flash_radiomics_free(discretized_owned);

    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    return result;
}

FlashRadiomicsResult* flash_radiomics_glcm_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
) {
    return flash_radiomics_glcm_cpu_internal(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        bin_width,
        bin_count,
        NULL,
        0,
        1
    );
}

FlashRadiomicsResult* flash_radiomics_glcm_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    const int* discretized,
    int ng
) {
    return flash_radiomics_glcm_cpu_internal(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        0.0,
        0,
        discretized,
        ng,
        1
    );
}

FlashRadiomicsResult* flash_radiomics_glcm_cpu_discretized_selective(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    const int* discretized,
    int ng,
    int include_mcc
) {
    return flash_radiomics_glcm_cpu_internal(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        0.0,
        0,
        discretized,
        ng,
        include_mcc
    );
}

int flash_radiomics_glcm_cpu_discretized_voxel_batch(
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
    if (!discretized_windows || !mask_windows || !window_dims || !distances || !out_features) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (batch_size <= 0 || num_distances <= 0 || ng <= 0) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (!flash_radiomics_validate_dims_with_ndim(ndim, window_dims)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    int selected_feature_indices[FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT];
    int selected_feature_count = 0;
    for (int idx = 0; idx < FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT; idx++) {
        if (is_glcm_feature_enabled(include_features, idx)) {
            selected_feature_indices[selected_feature_count++] = idx;
        }
    }
    if (selected_feature_count <= 0 || out_feature_stride < selected_feature_count) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    const int include_mcc = is_glcm_feature_enabled(
        include_features,
        GLCM_FEATURE_MCC
    ) ? 1 : 0;

    GLCMWorkspace workspace;
    if (!init_glcm_workspace(&workspace, ng, include_mcc)) {
        return FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
    }

    const size_t window_voxels = flash_radiomics_get_num_elements(ndim, window_dims);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        const int* batch_discretized = discretized_windows + (size_t)batch_idx * window_voxels;
        const uint8_t* batch_mask = mask_windows + (size_t)batch_idx * window_voxels;
        double* batch_output = out_features + (size_t)batch_idx * (size_t)out_feature_stride;

        GLCMFeatureValues values;
        if (!compute_glcm_patch_feature_values(
                batch_discretized,
                batch_mask,
                ndim,
                window_dims,
                distances,
                num_distances,
                ng,
                include_mcc,
                &workspace,
                &values)) {
            free_glcm_workspace(&workspace);
            return FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        }

        if (selected_feature_count == FLASH_RADIOMICS_GLCM_VOXEL_FEATURE_COUNT) {
            write_glcm_feature_row(&values, batch_output);
            continue;
        }

        for (int out_idx = 0; out_idx < selected_feature_count; out_idx++) {
            switch (selected_feature_indices[out_idx]) {
                case GLCM_FEATURE_AUTOCORRELATION:
                    batch_output[out_idx] = values.autocorrelation;
                    break;
                case GLCM_FEATURE_JOINT_AVERAGE:
                    batch_output[out_idx] = values.joint_average;
                    break;
                case GLCM_FEATURE_CLUSTER_PROMINENCE:
                    batch_output[out_idx] = values.cluster_prominence;
                    break;
                case GLCM_FEATURE_CLUSTER_SHADE:
                    batch_output[out_idx] = values.cluster_shade;
                    break;
                case GLCM_FEATURE_CLUSTER_TENDENCY:
                    batch_output[out_idx] = values.cluster_tendency;
                    break;
                case GLCM_FEATURE_CONTRAST:
                    batch_output[out_idx] = values.contrast;
                    break;
                case GLCM_FEATURE_CORRELATION:
                    batch_output[out_idx] = values.correlation;
                    break;
                case GLCM_FEATURE_DIFFERENCE_AVERAGE:
                    batch_output[out_idx] = values.difference_average;
                    break;
                case GLCM_FEATURE_DIFFERENCE_ENTROPY:
                    batch_output[out_idx] = values.difference_entropy;
                    break;
                case GLCM_FEATURE_DIFFERENCE_VARIANCE:
                    batch_output[out_idx] = values.difference_variance;
                    break;
                case GLCM_FEATURE_JOINT_ENERGY:
                    batch_output[out_idx] = values.joint_energy;
                    break;
                case GLCM_FEATURE_JOINT_ENTROPY:
                    batch_output[out_idx] = values.joint_entropy;
                    break;
                case GLCM_FEATURE_IMC1:
                    batch_output[out_idx] = values.imc1;
                    break;
                case GLCM_FEATURE_IMC2:
                    batch_output[out_idx] = values.imc2;
                    break;
                case GLCM_FEATURE_IDM:
                    batch_output[out_idx] = values.idm;
                    break;
                case GLCM_FEATURE_MCC:
                    batch_output[out_idx] = values.mcc;
                    break;
                case GLCM_FEATURE_IDMN:
                    batch_output[out_idx] = values.idmn;
                    break;
                case GLCM_FEATURE_ID:
                    batch_output[out_idx] = values.id;
                    break;
                case GLCM_FEATURE_IDN:
                    batch_output[out_idx] = values.idn;
                    break;
                case GLCM_FEATURE_INVERSE_VARIANCE:
                    batch_output[out_idx] = values.inverse_variance;
                    break;
                case GLCM_FEATURE_MAXIMUM_PROBABILITY:
                    batch_output[out_idx] = values.maximum_probability;
                    break;
                case GLCM_FEATURE_SUM_AVERAGE:
                    batch_output[out_idx] = values.sum_average;
                    break;
                case GLCM_FEATURE_SUM_ENTROPY:
                    batch_output[out_idx] = values.sum_entropy;
                    break;
                case GLCM_FEATURE_SUM_SQUARES:
                    batch_output[out_idx] = values.sum_squares;
                    break;
                default:
                    batch_output[out_idx] = NAN;
                    break;
            }
        }
    }

    free_glcm_workspace(&workspace);
    return FLASH_RADIOMICS_SUCCESS;
}
