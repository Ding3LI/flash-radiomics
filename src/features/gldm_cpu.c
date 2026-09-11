#include "gldm_cpu.h"
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

// 8 directions for 2D (full neighborhood)
static const int DIRECTIONS_2D[8][2] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {-1, -1}, {1, -1}, {-1, 1}
};

// 26 directions for 3D (full neighborhood)
static const int DIRECTIONS_3D[26][3] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
    {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1},
    {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1},
    {1, 1, 1}, {-1, -1, -1}, {1, 1, -1}, {-1, -1, 1},
    {1, -1, 1}, {-1, 1, -1}, {1, -1, -1}, {-1, 1, 1}
};

static FlashRadiomicsResult* flash_radiomics_gldm_cpu_internal(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    double gldm_a,
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
        discretized_owned = flash_radiomics_discretize_image(
            img, mask, bin_width, bin_count, &ng
        );
        discretized = discretized_owned;
    }
    if (!discretized || ng <= 0) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
                           "Failed to discretize image for GLDM");
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        return result;
    }

    int num_dirs = (img->ndim == 2) ? 8 : 26;
    int max_dep = num_dirs * num_distances;
    int dep_len = max_dep + 1;

    double* P = (double*)calloc((size_t)ng * dep_len, sizeof(double));
    if (!P) {
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate GLDM matrix");
        return result;
    }

    int dims_x = img->dims[0];
    int dims_y = img->dims[1];
    int dims_z = img->dims[2];

    #pragma omp parallel for collapse(2)
    for (int z = 0; z < dims_z; z++) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                size_t idx = flash_radiomics_index(x, y, z, img->ndim, img->dims);
                if (mask->data[idx] == 0) continue;

                int gray = discretized[idx];
                if (gray <= 0) continue;
                int g_idx = gray - 1;

                int dep = 0;
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
                            int diff = gray - ngray;
                            if (diff < 0) diff = -diff;
                            if ((double)diff <= gldm_a) dep++;
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
                            int diff = gray - ngray;
                            if (diff < 0) diff = -diff;
                            if ((double)diff <= gldm_a) dep++;
                        }
                    }
                }

                if (dep < 0) dep = 0;
                if (dep > max_dep) dep = max_dep;

                size_t pidx = (size_t)g_idx * dep_len + dep;
                #pragma omp atomic
                P[pidx] += 1.0;
            }
        }
    }

    // Compute coefficients
    double* pd = (double*)calloc(dep_len, sizeof(double));
    double* pg = (double*)calloc(ng, sizeof(double));
    if (!pd || !pg) {
        free(P);
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate GLDM coefficients");
        free(pd);
        free(pg);
        return result;
    }

    double Nz = 0.0;
    for (int i = 0; i < ng; i++) {
        for (int j = 0; j < dep_len; j++) {
            double v = P[i * dep_len + j];
            pg[i] += v;
            pd[j] += v;
            Nz += v;
        }
    }

    if (Nz <= 0.0) {
        free(P);
        free(pd);
        free(pg);
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
                           "Empty GLDM matrix");
        return result;
    }

    // Precompute means
    double mean_g = 0.0;
    double mean_d = 0.0;
    for (int i = 0; i < ng; i++) {
        mean_g += (i + 1) * pg[i];
    }
    for (int j = 0; j < dep_len; j++) {
        mean_d += (j + 1) * pd[j];
    }
    mean_g /= Nz;
    mean_d /= Nz;

    double sde = 0.0, lde = 0.0;
    double gln = 0.0, dn = 0.0, dnn = 0.0;
    double glv = 0.0, dv = 0.0, de = 0.0;
    double lgle = 0.0, hgle = 0.0;
    double sdlgle = 0.0, sdhgle = 0.0, ldlgle = 0.0, ldhgle = 0.0;

    for (int i = 0; i < ng; i++) {
        double i_val = (double)(i + 1);
        for (int j = 0; j < dep_len; j++) {
            double j_val = (double)(j + 1);
            double v = P[i * dep_len + j];
            if (v <= 0.0) continue;
            double p = v / Nz;
            sde += p / (j_val * j_val);
            lde += p * j_val * j_val;
            glv += p * (i_val - mean_g) * (i_val - mean_g);
            dv += p * (j_val - mean_d) * (j_val - mean_d);
            de -= p * log2(p + EPSILON);
            lgle += p / (i_val * i_val);
            hgle += p * i_val * i_val;
            sdlgle += p / (i_val * i_val * j_val * j_val);
            sdhgle += p * i_val * i_val / (j_val * j_val);
            ldlgle += p * j_val * j_val / (i_val * i_val);
            ldhgle += p * i_val * i_val * j_val * j_val;
        }
    }

    for (int i = 0; i < ng; i++) {
        gln += (pg[i] * pg[i]);
    }
    for (int j = 0; j < dep_len; j++) {
        dn += (pd[j] * pd[j]);
    }
    gln /= Nz;
    dn /= Nz;
    dnn = dn / Nz;

    // Populate result
    result->count = 14;
    result->features = (FlashRadiomicsFeature*)malloc(result->count * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        free(P);
        free(pd);
        free(pg);
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate feature array");
        return result;
    }

    int idx = 0;
    snprintf(result->features[idx].name, 64, "gldm_SmallDependenceEmphasis");
    result->features[idx++].value = sde;

    snprintf(result->features[idx].name, 64, "gldm_LargeDependenceEmphasis");
    result->features[idx++].value = lde;

    snprintf(result->features[idx].name, 64, "gldm_GrayLevelNonUniformity");
    result->features[idx++].value = gln;

    snprintf(result->features[idx].name, 64, "gldm_DependenceNonUniformity");
    result->features[idx++].value = dn;

    snprintf(result->features[idx].name, 64, "gldm_DependenceNonUniformityNormalized");
    result->features[idx++].value = dnn;

    snprintf(result->features[idx].name, 64, "gldm_GrayLevelVariance");
    result->features[idx++].value = glv;

    snprintf(result->features[idx].name, 64, "gldm_DependenceVariance");
    result->features[idx++].value = dv;

    snprintf(result->features[idx].name, 64, "gldm_DependenceEntropy");
    result->features[idx++].value = de;

    snprintf(result->features[idx].name, 64, "gldm_LowGrayLevelEmphasis");
    result->features[idx++].value = lgle;

    snprintf(result->features[idx].name, 64, "gldm_HighGrayLevelEmphasis");
    result->features[idx++].value = hgle;

    snprintf(result->features[idx].name, 64, "gldm_SmallDependenceLowGrayLevelEmphasis");
    result->features[idx++].value = sdlgle;

    snprintf(result->features[idx].name, 64, "gldm_SmallDependenceHighGrayLevelEmphasis");
    result->features[idx++].value = sdhgle;

    snprintf(result->features[idx].name, 64, "gldm_LargeDependenceLowGrayLevelEmphasis");
    result->features[idx++].value = ldlgle;

    snprintf(result->features[idx].name, 64, "gldm_LargeDependenceHighGrayLevelEmphasis");
    result->features[idx++].value = ldhgle;

    free(P);
    free(pd);
    free(pg);
    if (discretized_owned) {
        flash_radiomics_free(discretized_owned);
    }

    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    return result;
}

FlashRadiomicsResult* flash_radiomics_gldm_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    double bin_width,
    int bin_count,
    double gldm_a
) {
    return flash_radiomics_gldm_cpu_internal(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        bin_width,
        bin_count,
        gldm_a,
        NULL,
        0
    );
}

FlashRadiomicsResult* flash_radiomics_gldm_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    int* distances,
    int num_distances,
    const int* discretized,
    int ng,
    double gldm_a
) {
    return flash_radiomics_gldm_cpu_internal(
        img,
        mask,
        num_threads,
        distances,
        num_distances,
        0.0,
        0,
        gldm_a,
        discretized,
        ng
    );
}

typedef struct {
    int ndim;
    int dims[3];
    int num_bins;
    int num_distances;
    int num_dirs;
    int max_dep;
    int dep_len;
    size_t matrix_size;
    double* P;
    double* pd;
    double* pg;
} GLDMVoxelWorkspace;

static int init_gldm_voxel_workspace(
    GLDMVoxelWorkspace* ws,
    int ndim,
    const int dims[3],
    int num_bins,
    int num_distances
) {
    memset(ws, 0, sizeof(*ws));
    ws->ndim = ndim;
    ws->num_bins = num_bins;
    ws->num_distances = num_distances;
    ws->num_dirs = (ndim == 2) ? 8 : 26;

    for (int axis = 0; axis < 3; axis++) {
        ws->dims[axis] = (axis < ndim) ? dims[axis] : 1;
    }

    ws->max_dep = ws->num_dirs * ws->num_distances;
    ws->dep_len = ws->max_dep + 1;
    ws->matrix_size = (size_t)ws->num_bins * (size_t)ws->dep_len;

    ws->P = (double*)calloc(ws->matrix_size, sizeof(double));
    ws->pd = (double*)calloc((size_t)ws->dep_len, sizeof(double));
    ws->pg = (double*)calloc((size_t)ws->num_bins, sizeof(double));
    if (!ws->P || !ws->pd || !ws->pg) {
        return 0;
    }
    return 1;
}

static void free_gldm_voxel_workspace(GLDMVoxelWorkspace* ws) {
    if (!ws) {
        return;
    }
    free(ws->pg);
    free(ws->pd);
    free(ws->P);
    memset(ws, 0, sizeof(*ws));
}

static int is_gldm_feature_enabled(const uint8_t* include_features, int idx) {
    return (!include_features) || (include_features[idx] != 0);
}

static void compute_gldm_matrix_voxel_workspace(
    GLDMVoxelWorkspace* ws,
    const int* discretized,
    const uint8_t* mask,
    const int* distances,
    double gldm_a
) {
    memset(ws->P, 0, ws->matrix_size * sizeof(double));

    const int dims_x = ws->dims[0];
    const int dims_y = ws->dims[1];
    const int dims_z = ws->dims[2];

    if (ws->ndim == 2) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                const size_t idx = (size_t)y * (size_t)dims_x + (size_t)x;
                if (mask[idx] == 0) {
                    continue;
                }

                const int gray = discretized[idx];
                if (gray <= 0 || gray > ws->num_bins) {
                    continue;
                }

                int dep = 0;
                for (int dist_idx = 0; dist_idx < ws->num_distances; dist_idx++) {
                    const int dist = distances[dist_idx];
                    if (dist <= 0) {
                        continue;
                    }
                    for (int dir = 0; dir < ws->num_dirs; dir++) {
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
                        if (ngray <= 0 || ngray > ws->num_bins) {
                            continue;
                        }

                        int diff = gray - ngray;
                        if (diff < 0) {
                            diff = -diff;
                        }
                        if ((double)diff <= gldm_a) {
                            dep++;
                        }
                    }
                }

                if (dep < 0) {
                    dep = 0;
                }
                if (dep > ws->max_dep) {
                    dep = ws->max_dep;
                }
                ws->P[(size_t)(gray - 1) * (size_t)ws->dep_len + (size_t)dep] += 1.0;
            }
        }
        return;
    }

    for (int z = 0; z < dims_z; z++) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                const size_t idx = (size_t)z * (size_t)dims_y * (size_t)dims_x
                    + (size_t)y * (size_t)dims_x + (size_t)x;
                if (mask[idx] == 0) {
                    continue;
                }

                const int gray = discretized[idx];
                if (gray <= 0 || gray > ws->num_bins) {
                    continue;
                }

                int dep = 0;
                for (int dist_idx = 0; dist_idx < ws->num_distances; dist_idx++) {
                    const int dist = distances[dist_idx];
                    if (dist <= 0) {
                        continue;
                    }
                    for (int dir = 0; dir < ws->num_dirs; dir++) {
                        const int nx = x + DIRECTIONS_3D[dir][0] * dist;
                        const int ny = y + DIRECTIONS_3D[dir][1] * dist;
                        const int nz = z + DIRECTIONS_3D[dir][2] * dist;
                        if (nx < 0 || nx >= dims_x ||
                            ny < 0 || ny >= dims_y ||
                            nz < 0 || nz >= dims_z) {
                            continue;
                        }
                        const size_t nidx = (size_t)nz * (size_t)dims_y * (size_t)dims_x
                            + (size_t)ny * (size_t)dims_x + (size_t)nx;
                        if (mask[nidx] == 0) {
                            continue;
                        }
                        const int ngray = discretized[nidx];
                        if (ngray <= 0 || ngray > ws->num_bins) {
                            continue;
                        }

                        int diff = gray - ngray;
                        if (diff < 0) {
                            diff = -diff;
                        }
                        if ((double)diff <= gldm_a) {
                            dep++;
                        }
                    }
                }

                if (dep < 0) {
                    dep = 0;
                }
                if (dep > ws->max_dep) {
                    dep = ws->max_dep;
                }
                ws->P[(size_t)(gray - 1) * (size_t)ws->dep_len + (size_t)dep] += 1.0;
            }
        }
    }
}

static void compute_gldm_features_voxel_workspace(
    GLDMVoxelWorkspace* ws,
    const uint8_t* include_features,
    double out_features[FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT]
) {
    for (int idx = 0; idx < FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT; idx++) {
        out_features[idx] = NAN;
    }

    const int need_sde = is_gldm_feature_enabled(include_features, 0);
    const int need_lde = is_gldm_feature_enabled(include_features, 1);
    const int need_gln = is_gldm_feature_enabled(include_features, 2);
    const int need_dn = is_gldm_feature_enabled(include_features, 3);
    const int need_dnn = is_gldm_feature_enabled(include_features, 4);
    const int need_glv = is_gldm_feature_enabled(include_features, 5);
    const int need_dv = is_gldm_feature_enabled(include_features, 6);
    const int need_de = is_gldm_feature_enabled(include_features, 7);
    const int need_lgle = is_gldm_feature_enabled(include_features, 8);
    const int need_hgle = is_gldm_feature_enabled(include_features, 9);
    const int need_sdlgle = is_gldm_feature_enabled(include_features, 10);
    const int need_sdhgle = is_gldm_feature_enabled(include_features, 11);
    const int need_ldlgle = is_gldm_feature_enabled(include_features, 12);
    const int need_ldhgle = is_gldm_feature_enabled(include_features, 13);

    memset(ws->pg, 0, (size_t)ws->num_bins * sizeof(double));
    memset(ws->pd, 0, (size_t)ws->dep_len * sizeof(double));

    double Nz = 0.0;
    for (int i = 0; i < ws->num_bins; i++) {
        const size_t row_offset = (size_t)i * (size_t)ws->dep_len;
        for (int j = 0; j < ws->dep_len; j++) {
            const double count = ws->P[row_offset + (size_t)j];
            ws->pg[i] += count;
            ws->pd[j] += count;
            Nz += count;
        }
    }
    if (Nz <= 0.0) {
        return;
    }

    double mean_g = 0.0;
    if (need_glv) {
        for (int i = 0; i < ws->num_bins; i++) {
            mean_g += (double)(i + 1) * ws->pg[i];
        }
        mean_g /= Nz;
    }

    double mean_d = 0.0;
    if (need_dv) {
        for (int j = 0; j < ws->dep_len; j++) {
            mean_d += (double)(j + 1) * ws->pd[j];
        }
        mean_d /= Nz;
    }

    double sde = 0.0;
    double lde = 0.0;
    double glv = 0.0;
    double dv = 0.0;
    double de = 0.0;
    double lgle = 0.0;
    double hgle = 0.0;
    double sdlgle = 0.0;
    double sdhgle = 0.0;
    double ldlgle = 0.0;
    double ldhgle = 0.0;

    if (need_sde || need_lde || need_glv || need_dv || need_de ||
        need_lgle || need_hgle || need_sdlgle || need_sdhgle ||
        need_ldlgle || need_ldhgle) {
        for (int i = 0; i < ws->num_bins; i++) {
            const double i_val = (double)(i + 1);
            const double i_sq = i_val * i_val;
            const size_t row_offset = (size_t)i * (size_t)ws->dep_len;
            for (int j = 0; j < ws->dep_len; j++) {
                const double count = ws->P[row_offset + (size_t)j];
                if (count <= 0.0) {
                    continue;
                }
                const double j_val = (double)(j + 1);
                const double j_sq = j_val * j_val;
                const double p = count / Nz;

                if (need_sde) sde += p / j_sq;
                if (need_lde) lde += p * j_sq;
                if (need_glv) {
                    const double diff = i_val - mean_g;
                    glv += p * diff * diff;
                }
                if (need_dv) {
                    const double diff = j_val - mean_d;
                    dv += p * diff * diff;
                }
                if (need_de) de -= p * log2(p + EPSILON);
                if (need_lgle) lgle += p / i_sq;
                if (need_hgle) hgle += p * i_sq;
                if (need_sdlgle) sdlgle += p / (i_sq * j_sq);
                if (need_sdhgle) sdhgle += p * i_sq / j_sq;
                if (need_ldlgle) ldlgle += p * j_sq / i_sq;
                if (need_ldhgle) ldhgle += p * i_sq * j_sq;
            }
        }
    }

    double gln_sum = 0.0;
    if (need_gln) {
        for (int i = 0; i < ws->num_bins; i++) {
            gln_sum += ws->pg[i] * ws->pg[i];
        }
    }

    double dn_sum = 0.0;
    if (need_dn || need_dnn) {
        for (int j = 0; j < ws->dep_len; j++) {
            dn_sum += ws->pd[j] * ws->pd[j];
        }
    }

    if (need_sde) out_features[0] = sde;
    if (need_lde) out_features[1] = lde;
    if (need_gln) out_features[2] = gln_sum / Nz;
    if (need_dn) out_features[3] = dn_sum / Nz;
    if (need_dnn) out_features[4] = dn_sum / (Nz * Nz);
    if (need_glv) out_features[5] = glv;
    if (need_dv) out_features[6] = dv;
    if (need_de) out_features[7] = de;
    if (need_lgle) out_features[8] = lgle;
    if (need_hgle) out_features[9] = hgle;
    if (need_sdlgle) out_features[10] = sdlgle;
    if (need_sdhgle) out_features[11] = sdhgle;
    if (need_ldlgle) out_features[12] = ldlgle;
    if (need_ldhgle) out_features[13] = ldhgle;
}

int flash_radiomics_gldm_cpu_discretized_voxel_batch(
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
    if (!discretized_windows || !mask_windows || !window_dims || !distances || !out_features) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (batch_size <= 0 || ng <= 0 || num_distances <= 0) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (out_feature_stride < FLASH_RADIOMICS_GLDM_VOXEL_FEATURE_COUNT) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (!flash_radiomics_validate_dims_with_ndim(ndim, window_dims)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    GLDMVoxelWorkspace workspace;
    if (!init_gldm_voxel_workspace(&workspace, ndim, window_dims, ng, num_distances)) {
        free_gldm_voxel_workspace(&workspace);
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

        compute_gldm_matrix_voxel_workspace(
            &workspace,
            batch_discretized,
            batch_mask,
            distances,
            gldm_a
        );
        compute_gldm_features_voxel_workspace(&workspace, include_features, batch_output);
    }

    free_gldm_voxel_workspace(&workspace);
    return FLASH_RADIOMICS_SUCCESS;
}
