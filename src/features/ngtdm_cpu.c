#include "ngtdm_cpu.h"
#include "../core/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define EPSILON 2.2e-16

static const int DIRECTIONS_2D[8][2] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};

static const int DIRECTIONS_3D[26][3] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
    {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1},
    {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1},
    {1, 1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, -1, 1},
    {1, -1, 1}, {-1, 1, -1}, {1, -1, -1}, {-1, 1, 1}
};

/* PyRadiomics-compatible discretization for NGTDM:
 * keep original bin indices (do not compact missing bins), so Ng is max bin index.
 */
static int* discretize_image_ngtdm(
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

typedef struct {
    int ndim;
    int dims[3];
    int ng;
    double* n_i;
    double* s_i;
    double* p_i;
    int* active_bins;
} NGTDMVoxelWorkspace;

static void free_ngtdm_voxel_workspace(NGTDMVoxelWorkspace* ws) {
    if (!ws) {
        return;
    }
    free(ws->n_i);
    free(ws->s_i);
    free(ws->p_i);
    free(ws->active_bins);
    ws->n_i = NULL;
    ws->s_i = NULL;
    ws->p_i = NULL;
    ws->active_bins = NULL;
}

static int init_ngtdm_voxel_workspace(
    NGTDMVoxelWorkspace* ws,
    int ndim,
    const int dims[3],
    int ng
) {
    if (!ws || !dims || ng <= 0) {
        return 0;
    }

    memset(ws, 0, sizeof(*ws));
    ws->ndim = ndim;
    ws->dims[0] = dims[0];
    ws->dims[1] = dims[1];
    ws->dims[2] = dims[2];
    ws->ng = ng;

    ws->n_i = (double*)calloc((size_t)ng, sizeof(double));
    ws->s_i = (double*)calloc((size_t)ng, sizeof(double));
    ws->p_i = (double*)calloc((size_t)ng, sizeof(double));
    ws->active_bins = (int*)calloc((size_t)ng, sizeof(int));
    if (!ws->n_i || !ws->s_i || !ws->p_i || !ws->active_bins) {
        free_ngtdm_voxel_workspace(ws);
        return 0;
    }
    return 1;
}

static void reset_ngtdm_voxel_workspace(NGTDMVoxelWorkspace* ws) {
    if (!ws || ws->ng <= 0) {
        return;
    }
    memset(ws->n_i, 0, (size_t)ws->ng * sizeof(double));
    memset(ws->s_i, 0, (size_t)ws->ng * sizeof(double));
}

static void set_ngtdm_features_nan(double* out_features) {
    if (!out_features) {
        return;
    }
    for (int idx = 0; idx < FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT; idx++) {
        out_features[idx] = NAN;
    }
}

static int is_ngtdm_feature_enabled(const uint8_t* include_features, int idx) {
    return (!include_features) || (include_features[idx] != 0);
}

static void compute_ngtdm_features_voxel_workspace(
    NGTDMVoxelWorkspace* ws,
    const int* discretized,
    const uint8_t* mask,
    const int* distances,
    int num_distances,
    const uint8_t* include_features,
    double out_features[FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT]
) {
    if (!ws || !discretized || !mask || !distances || num_distances <= 0 || !out_features) {
        set_ngtdm_features_nan(out_features);
        return;
    }

    set_ngtdm_features_nan(out_features);
    const int need_coarseness = is_ngtdm_feature_enabled(include_features, 0);
    const int need_contrast = is_ngtdm_feature_enabled(include_features, 1);
    const int need_busyness = is_ngtdm_feature_enabled(include_features, 2);
    const int need_complexity = is_ngtdm_feature_enabled(include_features, 3);
    const int need_strength = is_ngtdm_feature_enabled(include_features, 4);
    if (need_busyness) {
        /* Match PyRadiomics: undefined/degenerate busyness falls back to 0. */
        out_features[2] = 0.0;
    }

    if (!need_coarseness && !need_contrast && !need_busyness &&
        !need_complexity && !need_strength) {
        return;
    }

    reset_ngtdm_voxel_workspace(ws);

    const int dims_x = ws->dims[0];
    const int dims_y = ws->dims[1];
    const int dims_z = ws->dims[2];
    const int num_dirs = (ws->ndim == 2) ? 8 : 26;

    for (int z = 0; z < dims_z; z++) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                const size_t idx = flash_radiomics_index(x, y, z, ws->ndim, ws->dims);
                if (mask[idx] == 0) {
                    continue;
                }

                const int gray = discretized[idx];
                if (gray <= 0 || gray > ws->ng) {
                    continue;
                }
                const int g_idx = gray - 1;

                double sum_neighbors = 0.0;
                int count_neighbors = 0;

                for (int d = 0; d < num_distances; d++) {
                    const int dist = distances[d];
                    if (ws->ndim == 2) {
                        for (int dir = 0; dir < num_dirs; dir++) {
                            const int nx = x + DIRECTIONS_2D[dir][0] * dist;
                            const int ny = y + DIRECTIONS_2D[dir][1] * dist;
                            if (nx < 0 || nx >= dims_x || ny < 0 || ny >= dims_y) {
                                continue;
                            }
                            const size_t nidx = (size_t)ny * (size_t)dims_x + (size_t)nx;
                            if (mask[nidx] == 0) {
                                continue;
                            }
                            const int ngray = discretized[nidx];
                            if (ngray <= 0 || ngray > ws->ng) {
                                continue;
                            }
                            sum_neighbors += (double)ngray;
                            count_neighbors++;
                        }
                    } else {
                        for (int dir = 0; dir < num_dirs; dir++) {
                            const int nx = x + DIRECTIONS_3D[dir][0] * dist;
                            const int ny = y + DIRECTIONS_3D[dir][1] * dist;
                            const int nz = z + DIRECTIONS_3D[dir][2] * dist;
                            if (nx < 0 || nx >= dims_x || ny < 0 || ny >= dims_y ||
                                nz < 0 || nz >= dims_z) {
                                continue;
                            }
                            const size_t nidx =
                                (size_t)nz * (size_t)dims_y * (size_t)dims_x +
                                (size_t)ny * (size_t)dims_x + (size_t)nx;
                            if (mask[nidx] == 0) {
                                continue;
                            }
                            const int ngray = discretized[nidx];
                            if (ngray <= 0 || ngray > ws->ng) {
                                continue;
                            }
                            sum_neighbors += (double)ngray;
                            count_neighbors++;
                        }
                    }
                }

                if (count_neighbors <= 0) {
                    continue;
                }

                const double avg = sum_neighbors / (double)count_neighbors;
                const double diff = fabs((double)gray - avg);
                ws->n_i[g_idx] += 1.0;
                ws->s_i[g_idx] += diff;
            }
        }
    }

    double Nvp = 0.0;
    int Ngp = 0;
    for (int i = 0; i < ws->ng; i++) {
        const double count = ws->n_i[i];
        Nvp += count;
        if (count > 0.0) {
            ws->active_bins[Ngp++] = i;
        }
    }

    if (Nvp <= 0.0) {
        return;
    }

    double sum_p_s = 0.0;
    double sum_s_i = 0.0;
    for (int k = 0; k < Ngp; k++) {
        const int i = ws->active_bins[k];
        const double p = ws->n_i[i] / Nvp;
        ws->p_i[i] = p;
        sum_p_s += p * ws->s_i[i];
        sum_s_i += ws->s_i[i];
    }

    if (need_coarseness) {
        out_features[0] = (sum_p_s != 0.0) ? (1.0 / sum_p_s) : 1e6;
    }

    if (need_contrast && Ngp > 1) {
        double sum_ij = 0.0;
        for (int ki = 0; ki < Ngp; ki++) {
            const int i = ws->active_bins[ki];
            const double p_i = ws->p_i[i];
            const double i_val = (double)(i + 1);
            for (int kj = ki + 1; kj < Ngp; kj++) {
                const int j = ws->active_bins[kj];
                const double p_j = ws->p_i[j];
                const double j_val = (double)(j + 1);
                const double gray_diff = i_val - j_val;
                sum_ij += 2.0 * p_i * p_j * gray_diff * gray_diff;
            }
        }
        out_features[1] =
            (sum_ij / ((double)Ngp * (double)(Ngp - 1))) * (sum_s_i / Nvp);
    }

    if (need_busyness) {
        double denom_busy = 0.0;
        for (int ki = 0; ki < Ngp; ki++) {
            const int i = ws->active_bins[ki];
            const double i_pi = (double)(i + 1) * ws->p_i[i];
            for (int kj = ki + 1; kj < Ngp; kj++) {
                const int j = ws->active_bins[kj];
                const double j_pi = (double)(j + 1) * ws->p_i[j];
                denom_busy += 2.0 * fabs(i_pi - j_pi);
            }
        }
        if (isfinite(denom_busy) && denom_busy != 0.0 && isfinite(sum_p_s)) {
            const double busyness = sum_p_s / denom_busy;
            out_features[2] = isfinite(busyness) ? busyness : 0.0;
        }
    }

    if (need_complexity) {
        double complexity = 0.0;
        for (int ki = 0; ki < Ngp; ki++) {
            const int i = ws->active_bins[ki];
            const double p_i = ws->p_i[i];
            const double s_i = ws->s_i[i];
            const double i_val = (double)(i + 1);
            for (int kj = ki + 1; kj < Ngp; kj++) {
                const int j = ws->active_bins[kj];
                const double p_j = ws->p_i[j];
                const double s_j = ws->s_i[j];
                const double j_val = (double)(j + 1);
                const double denom = p_i + p_j;
                complexity += 2.0 * (fabs(i_val - j_val) / denom) *
                    (p_i * s_i + p_j * s_j);
            }
        }
        out_features[3] = complexity / Nvp;
    }

    if (need_strength && sum_s_i != 0.0) {
        double sum_strength = 0.0;
        for (int ki = 0; ki < Ngp; ki++) {
            const int i = ws->active_bins[ki];
            const double p_i = ws->p_i[i];
            const double i_val = (double)(i + 1);
            for (int kj = ki + 1; kj < Ngp; kj++) {
                const int j = ws->active_bins[kj];
                const double p_j = ws->p_i[j];
                const double j_val = (double)(j + 1);
                const double gray_diff = i_val - j_val;
                sum_strength += 2.0 * (p_i + p_j) * gray_diff * gray_diff;
            }
        }
        out_features[4] = sum_strength / sum_s_i;
    }
}

static FlashRadiomicsResult* flash_radiomics_ngtdm_cpu_internal(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    const int* discretized_input,
    int ng_input
) {
    clock_t start_time = clock();

    FlashRadiomicsResult* result = (FlashRadiomicsResult*)malloc(sizeof(FlashRadiomicsResult));
    if (!result) return NULL;
    result->features = NULL;
    result->count = 0;
    result->compute_time_ms = 0.0;
    result->error_code = FLASH_RADIOMICS_SUCCESS;
    result->error_msg[0] = 0;

    if (!img || !mask) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                           "Image or mask is NULL");
        return result;
    }
    if (img->ndim != mask->ndim || !flash_radiomics_dims_match(img->dims, mask->dims)) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH,
                           "Image and mask dimensions do not match");
        return result;
    }
    if (!distances || num_distances <= 0) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                           "Invalid distances parameter");
        return result;
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif

    int ng = ng_input;
    int* discretized_owned = NULL;
    const int* discretized = discretized_input;
    if (!discretized || ng <= 0) {
        discretized_owned = discretize_image_ngtdm(
            img, mask, bin_width, bin_count, &ng
        );
        discretized = discretized_owned;
    }
    if (!discretized || ng <= 0) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
                           "Failed to discretize image for NGTDM");
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        return result;
    }

    int num_dirs = (img->ndim == 2) ? 8 : 26;
    int dims_x = img->dims[0];
    int dims_y = img->dims[1];
    int dims_z = img->dims[2];

    double* n_i = (double*)calloc(ng, sizeof(double));
    double* s_i = (double*)calloc(ng, sizeof(double));
    if (!n_i || !s_i) {
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        free(n_i);
        free(s_i);
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate NGTDM coefficients");
        return result;
    }

    #pragma omp parallel for collapse(2)
    for (int z = 0; z < dims_z; z++) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                size_t idx = flash_radiomics_index(x, y, z, img->ndim, img->dims);
                if (mask->data[idx] == 0) continue;
                int gray = discretized[idx];
                if (gray <= 0) continue;
                int g_idx = gray - 1;

                double sum_neighbors = 0.0;
                int count_neighbors = 0;

                for (int d = 0; d < num_distances; d++) {
                    int dist = distances[d];
                    if (img->ndim == 2) {
                        for (int dir = 0; dir < num_dirs; dir++) {
                            int nx = x + DIRECTIONS_2D[dir][0] * dist;
                            int ny = y + DIRECTIONS_2D[dir][1] * dist;
                            if (nx < 0 || nx >= dims_x || ny < 0 || ny >= dims_y) continue;
                            size_t nidx = (size_t)ny * dims_x + nx;
                            if (mask->data[nidx] == 0) continue;
                            int ngray = discretized[nidx];
                            if (ngray <= 0) continue;
                            sum_neighbors += ngray;
                            count_neighbors++;
                        }
                    } else {
                        for (int dir = 0; dir < num_dirs; dir++) {
                            int nx = x + DIRECTIONS_3D[dir][0] * dist;
                            int ny = y + DIRECTIONS_3D[dir][1] * dist;
                            int nz = z + DIRECTIONS_3D[dir][2] * dist;
                            if (nx < 0 || nx >= dims_x || ny < 0 || ny >= dims_y || nz < 0 || nz >= dims_z) continue;
                            size_t nidx = (size_t)nz * dims_y * dims_x + (size_t)ny * dims_x + nx;
                            if (mask->data[nidx] == 0) continue;
                            int ngray = discretized[nidx];
                            if (ngray <= 0) continue;
                            sum_neighbors += ngray;
                            count_neighbors++;
                        }
                    }
                }

                /* Match PyRadiomics NGTDM definition:
                 * only voxels with at least one valid neighbor contribute to Nvp/n_i/s_i. */
                if (count_neighbors <= 0) continue;

                double avg = sum_neighbors / count_neighbors;
                double diff = fabs((double)gray - avg);
                #pragma omp atomic
                n_i[g_idx] += 1.0;
                #pragma omp atomic
                s_i[g_idx] += diff;
            }
        }
    }

    double Nvp = 0.0;
    int Ngp = 0;
    for (int i = 0; i < ng; i++) {
        Nvp += n_i[i];
        if (n_i[i] > 0) Ngp++;
    }

    if (Nvp <= 0.0) {
        free(n_i);
        free(s_i);
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
                           "Empty NGTDM matrix");
        return result;
    }

    double* p_i = (double*)calloc(ng, sizeof(double));
    if (!p_i) {
        free(n_i);
        free(s_i);
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate NGTDM probabilities");
        return result;
    }
    for (int i = 0; i < ng; i++) {
        p_i[i] = n_i[i] / Nvp;
    }

    double sum_p_s = 0.0;
    for (int i = 0; i < ng; i++) {
        sum_p_s += p_i[i] * s_i[i];
    }

    double sum_s_i = 0.0;
    for (int i = 0; i < ng; i++) {
        sum_s_i += s_i[i];
    }

    // Coarseness
    double coarseness = (sum_p_s != 0.0) ? (1.0 / sum_p_s) : 1e6;

    // Contrast
    double contrast = 0.0;
    if (Ngp > 1) {
        double sum_ij = 0.0;
        for (int i = 0; i < ng; i++) {
            if (p_i[i] == 0) continue;
            double i_val = (double)(i + 1);
            for (int j = 0; j < ng; j++) {
                if (p_i[j] == 0) continue;
                double j_val = (double)(j + 1);
                double diff = i_val - j_val;
                sum_ij += p_i[i] * p_i[j] * diff * diff;
            }
        }
        contrast = (sum_ij / (Ngp * (Ngp - 1))) * (sum_s_i / Nvp);
    }

    // Busyness
    double busyness = 0.0;
    double denom_busy = 0.0;
    for (int i = 0; i < ng; i++) {
        if (p_i[i] == 0) continue;
        double i_pi = (i + 1) * p_i[i];
        for (int j = 0; j < ng; j++) {
            if (p_i[j] == 0) continue;
            double j_pi = (j + 1) * p_i[j];
            denom_busy += fabs(i_pi - j_pi);
        }
    }
    if (denom_busy != 0.0) {
        busyness = sum_p_s / denom_busy;
    }

    // Complexity
    double complexity = 0.0;
    for (int i = 0; i < ng; i++) {
        if (p_i[i] == 0) continue;
        double i_val = (double)(i + 1);
        for (int j = 0; j < ng; j++) {
            if (p_i[j] == 0) continue;
            double j_val = (double)(j + 1);
            double denom = p_i[i] + p_i[j];
            if (denom == 0.0) continue;
            complexity += (fabs(i_val - j_val) / denom) * (p_i[i] * s_i[i] + p_i[j] * s_i[j]);
        }
    }
    complexity = complexity / Nvp;

    // Strength
    double strength = 0.0;
    if (sum_s_i != 0.0) {
        double sum_strength = 0.0;
        for (int i = 0; i < ng; i++) {
            if (p_i[i] == 0) continue;
            double i_val = (double)(i + 1);
            for (int j = 0; j < ng; j++) {
                if (p_i[j] == 0) continue;
                double j_val = (double)(j + 1);
                double diff = i_val - j_val;
                sum_strength += (p_i[i] + p_i[j]) * diff * diff;
            }
        }
        strength = sum_strength / sum_s_i;
    }

    result->count = 5;
    result->features = (FlashRadiomicsFeature*)malloc(result->count * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        free(n_i);
        free(s_i);
        free(p_i);
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate feature array");
        return result;
    }

    int idx = 0;
    snprintf(result->features[idx].name, 64, "ngtdm_Coarseness");
    result->features[idx++].value = coarseness;

    snprintf(result->features[idx].name, 64, "ngtdm_Contrast");
    result->features[idx++].value = contrast;

    snprintf(result->features[idx].name, 64, "ngtdm_Busyness");
    result->features[idx++].value = busyness;

    snprintf(result->features[idx].name, 64, "ngtdm_Complexity");
    result->features[idx++].value = complexity;

    snprintf(result->features[idx].name, 64, "ngtdm_Strength");
    result->features[idx++].value = strength;

    free(n_i);
    free(s_i);
    free(p_i);
    if (discretized_owned) {
        flash_radiomics_free(discretized_owned);
    }

    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    return result;
}

FlashRadiomicsResult* flash_radiomics_ngtdm_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count
) {
    return flash_radiomics_ngtdm_cpu_internal(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        bin_width,
        bin_count,
        NULL,
        0
    );
}

FlashRadiomicsResult* flash_radiomics_ngtdm_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    const int* discretized,
    int ng
) {
    return flash_radiomics_ngtdm_cpu_internal(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        0.0,
        0,
        discretized,
        ng
    );
}

int flash_radiomics_ngtdm_cpu_discretized_voxel_batch(
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
    if (!discretized_windows || !mask_windows || !window_dims || !distances || !out_features) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (batch_size <= 0 || ng <= 0 || num_distances <= 0) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (out_feature_stride < FLASH_RADIOMICS_NGTDM_VOXEL_FEATURE_COUNT) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (!flash_radiomics_validate_dims_with_ndim(ndim, window_dims)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    NGTDMVoxelWorkspace workspace;
    if (!init_ngtdm_voxel_workspace(&workspace, ndim, window_dims, ng)) {
        free_ngtdm_voxel_workspace(&workspace);
        return FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
    }

    const size_t window_voxels = flash_radiomics_get_num_elements(ndim, window_dims);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        const int* batch_discretized =
            discretized_windows + ((size_t)batch_idx * window_voxels);
        const uint8_t* batch_mask =
            mask_windows + ((size_t)batch_idx * window_voxels);
        double* batch_output =
            out_features + ((size_t)batch_idx * (size_t)out_feature_stride);

        compute_ngtdm_features_voxel_workspace(
            &workspace,
            batch_discretized,
            batch_mask,
            distances,
            num_distances,
            include_features,
            batch_output
        );
    }

    free_ngtdm_voxel_workspace(&workspace);
    return FLASH_RADIOMICS_SUCCESS;
}
