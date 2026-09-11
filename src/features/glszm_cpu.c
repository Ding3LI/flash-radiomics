#include "glszm_cpu.h"
#include "../core/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define EPSILON 2.2e-16

// Helper structure for GLSZM computation
typedef struct {
    double** matrix;        // GLSZM matrix [i][j] where i=gray level, j=zone size
    int num_bins;          // Number of gray levels (Ng)
    int max_zone_size;     // Maximum zone size (Ns)
    int num_entries;       // Number of non-zero (gray level, zone size) entries
    int* entry_gray_idx;   // 0-indexed gray level per non-zero entry
    int* entry_zone_size;  // 1-indexed zone size per non-zero entry
    double* entry_count;   // Count per non-zero entry
    double* pg;            // Marginal gray-level sums
    double* ps;            // Marginal zone-size sums
    double total_zones;    // Sum of all zone counts
    double min_val;        // Minimum value in ROI
    double max_val;        // Maximum value in ROI
    double bin_width;      // Width of each bin
} GLSZMMatrix;

// Union-Find data structure for connected component labeling
typedef struct {
    int* parent;
    int* rank;
    int size;
} UnionFind;

typedef struct {
    int gray_idx;
    int zone_size;
} GLSZMPair;

typedef struct {
    double small_area_emphasis;
    double large_area_emphasis;
    double low_gray_level_zone_emphasis;
    double high_gray_level_zone_emphasis;
    double small_area_low_gray_level_emphasis;
    double small_area_high_gray_level_emphasis;
    double large_area_low_gray_level_emphasis;
    double large_area_high_gray_level_emphasis;
    double gray_level_non_uniformity;
    double gray_level_non_uniformity_normalized;
    double size_zone_non_uniformity;
    double size_zone_non_uniformity_normalized;
    double zone_percentage;
    double gray_level_variance;
    double size_zone_variance;
    double zone_entropy;
} GLSZMFeatureValues;

// Initialize Union-Find structure
static UnionFind* uf_create(int size) {
    UnionFind* uf = (UnionFind*)malloc(sizeof(UnionFind));
    if (!uf) return NULL;
    
    uf->parent = (int*)malloc(size * sizeof(int));
    uf->rank = (int*)malloc(size * sizeof(int));
    if (!uf->parent || !uf->rank) {
        free(uf->parent);
        free(uf->rank);
        free(uf);
        return NULL;
    }
    
    uf->size = size;
    for (int i = 0; i < size; i++) {
        uf->parent[i] = i;
        uf->rank[i] = 0;
    }
    
    return uf;
}

// Find with path compression
static int uf_find(UnionFind* uf, int x) {
    if (uf->parent[x] != x) {
        uf->parent[x] = uf_find(uf, uf->parent[x]);
    }
    return uf->parent[x];
}

// Union by rank
static void uf_union(UnionFind* uf, int x, int y) {
    int root_x = uf_find(uf, x);
    int root_y = uf_find(uf, y);
    
    if (root_x == root_y) return;
    
    if (uf->rank[root_x] < uf->rank[root_y]) {
        uf->parent[root_x] = root_y;
    } else if (uf->rank[root_x] > uf->rank[root_y]) {
        uf->parent[root_y] = root_x;
    } else {
        uf->parent[root_y] = root_x;
        uf->rank[root_x]++;
    }
}

// Free Union-Find structure
static void uf_free(UnionFind* uf) {
    if (uf) {
        free(uf->parent);
        free(uf->rank);
        free(uf);
    }
}

static int compare_glszm_pairs(const void* lhs, const void* rhs) {
    const GLSZMPair* a = (const GLSZMPair*)lhs;
    const GLSZMPair* b = (const GLSZMPair*)rhs;
    if (a->gray_idx != b->gray_idx) {
        return (a->gray_idx < b->gray_idx) ? -1 : 1;
    }
    if (a->zone_size != b->zone_size) {
        return (a->zone_size < b->zone_size) ? -1 : 1;
    }
    return 0;
}

// Task 8.1: Connected component labeling with union-find for 2D (8-connectivity)
static int* label_zones_2d(int* discretized, uint8_t* mask, int dims_x, int dims_y,
                           int* num_zones_out) {
    size_t total_pixels = dims_x * dims_y;
    int* labels = (int*)malloc(total_pixels * sizeof(int));
    if (!labels) return NULL;
    
    // Initialize labels
    for (size_t i = 0; i < total_pixels; i++) {
        labels[i] = -1;
    }
    
    // Create union-find structure
    UnionFind* uf = uf_create(total_pixels);
    if (!uf) {
        free(labels);
        return NULL;
    }
    
    // 8-connectivity forward neighbors for raster scan:
    // right, down-left, down, down-right
    int dx[4] = {1, -1, 0, 1};
    int dy[4] = {0, 1, 1, 1};
    
    // First pass: connect neighbors with same gray level
    for (int y = 0; y < dims_y; y++) {
        for (int x = 0; x < dims_x; x++) {
            size_t idx = y * dims_x + x;
            
            if (mask[idx] == 0) continue;
            
            int gray_level = discretized[idx];
            
            // Check forward neighbors
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                
                if (nx >= 0 && nx < dims_x && ny >= 0 && ny < dims_y) {
                    size_t nidx = ny * dims_x + nx;
                    
                    if (mask[nidx] != 0 && discretized[nidx] == gray_level) {
                        uf_union(uf, idx, nidx);
                    }
                }
            }
        }
    }
    
    // Second pass: assign unique labels to each zone
    int next_label = 0;
    int* root_to_label = (int*)malloc(total_pixels * sizeof(int));
    if (!root_to_label) {
        uf_free(uf);
        free(labels);
        return NULL;
    }
    
    for (size_t i = 0; i < total_pixels; i++) {
        root_to_label[i] = -1;
    }
    
    for (size_t i = 0; i < total_pixels; i++) {
        if (mask[i] != 0) {
            int root = uf_find(uf, i);
            if (root_to_label[root] == -1) {
                root_to_label[root] = next_label++;
            }
            labels[i] = root_to_label[root];
        }
    }
    
    *num_zones_out = next_label;
    
    free(root_to_label);
    uf_free(uf);
    
    return labels;
}

// Task 8.1: Connected component labeling with union-find for 3D (26-connectivity)
static int* label_zones_3d(int* discretized, uint8_t* mask, int dims_x, int dims_y, int dims_z,
                           int* num_zones_out) {
    size_t total_voxels = (size_t)dims_x * dims_y * dims_z;
    int* labels = (int*)malloc(total_voxels * sizeof(int));
    if (!labels) return NULL;
    
    // Initialize labels
    for (size_t i = 0; i < total_voxels; i++) {
        labels[i] = -1;
    }
    
    // Create union-find structure
    UnionFind* uf = uf_create(total_voxels);
    if (!uf) {
        free(labels);
        return NULL;
    }
    
    // 26-connectivity forward neighbors for raster scan (x, then y, then z):
    // all offsets with dz > 0, plus dz == 0 with dy > 0, plus (dx=1,dy=0,dz=0).
    int dx[13] = {
        1,
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };
    int dy[13] = {
        0,
        1, 1, 1,
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1
    };
    int dz[13] = {
        0,
        0, 0, 0,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    
    // First pass: connect neighbors with same gray level
    for (int z = 0; z < dims_z; z++) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                size_t idx = z * dims_y * dims_x + y * dims_x + x;
                
                if (mask[idx] == 0) continue;
                
                int gray_level = discretized[idx];
                
                // Check 13 forward neighbors
                for (int d = 0; d < 13; d++) {
                    int nx = x + dx[d];
                    int ny = y + dy[d];
                    int nz = z + dz[d];
                    
                    if (nx >= 0 && nx < dims_x && ny >= 0 && ny < dims_y && 
                        nz >= 0 && nz < dims_z) {
                        size_t nidx = nz * dims_y * dims_x + ny * dims_x + nx;
                        
                        if (mask[nidx] != 0 && discretized[nidx] == gray_level) {
                            uf_union(uf, idx, nidx);
                        }
                    }
                }
            }
        }
    }
    
    // Second pass: assign unique labels to each zone
    int next_label = 0;
    int* root_to_label = (int*)malloc(total_voxels * sizeof(int));
    if (!root_to_label) {
        uf_free(uf);
        free(labels);
        return NULL;
    }
    
    for (size_t i = 0; i < total_voxels; i++) {
        root_to_label[i] = -1;
    }
    
    for (size_t i = 0; i < total_voxels; i++) {
        if (mask[i] != 0) {
            int root = uf_find(uf, i);
            if (root_to_label[root] == -1) {
                root_to_label[root] = next_label++;
            }
            labels[i] = root_to_label[root];
        }
    }
    
    *num_zones_out = next_label;
    
    free(root_to_label);
    uf_free(uf);
    
    return labels;
}

// Task 8.1: Build GLSZM matrix from labeled zones
static GLSZMMatrix* build_glszm_matrix(int* discretized, int* labels, uint8_t* mask,
                                       int ndim, const int dims[3], int num_bins,
                                       int num_zones, double min_val, double max_val) {
    GLSZMMatrix* glszm = (GLSZMMatrix*)calloc(1, sizeof(GLSZMMatrix));
    if (!glszm) return NULL;

    glszm->num_bins = num_bins;
    glszm->min_val = min_val;
    glszm->max_val = max_val;
    glszm->bin_width = (max_val - min_val) / num_bins;
    glszm->matrix = NULL;

    int* zone_sizes = (int*)calloc((size_t)num_zones, sizeof(int));
    int* zone_gray_levels = (int*)malloc((size_t)num_zones * sizeof(int));
    if (!zone_sizes || !zone_gray_levels) {
        free(zone_sizes);
        free(zone_gray_levels);
        free(glszm);
        return NULL;
    }

    for (int i = 0; i < num_zones; i++) {
        zone_gray_levels[i] = -1;
    }

    size_t total_voxels = flash_radiomics_get_num_elements(ndim, dims);
    for (size_t i = 0; i < total_voxels; i++) {
        if (mask[i] != 0 && labels[i] >= 0) {
            int label = labels[i];
            zone_sizes[label]++;
            if (zone_gray_levels[label] == -1) {
                zone_gray_levels[label] = discretized[i] - 1;
            }
        }
    }

    int max_zone_size = 0;
    for (int i = 0; i < num_zones; i++) {
        if (zone_sizes[i] > max_zone_size) {
            max_zone_size = zone_sizes[i];
        }
    }
    glszm->max_zone_size = max_zone_size;

    glszm->pg = (double*)calloc((size_t)num_bins, sizeof(double));
    if (!glszm->pg) {
        free(zone_sizes);
        free(zone_gray_levels);
        free(glszm);
        return NULL;
    }

    if (max_zone_size > 0) {
        glszm->ps = (double*)calloc((size_t)max_zone_size, sizeof(double));
        if (!glszm->ps) {
            free(zone_sizes);
            free(zone_gray_levels);
            free(glszm->pg);
            free(glszm);
            return NULL;
        }
    }

    int valid_zones = 0;
    for (int i = 0; i < num_zones; i++) {
        if (zone_sizes[i] > 0 && zone_gray_levels[i] >= 0 && zone_gray_levels[i] < num_bins) {
            valid_zones++;
        }
    }

    if (valid_zones > 0) {
        GLSZMPair* pairs = (GLSZMPair*)malloc((size_t)valid_zones * sizeof(GLSZMPair));
        if (!pairs) {
            free(zone_sizes);
            free(zone_gray_levels);
            free(glszm->ps);
            free(glszm->pg);
            free(glszm);
            return NULL;
        }

        int pair_idx = 0;
        for (int i = 0; i < num_zones; i++) {
            if (zone_sizes[i] > 0 && zone_gray_levels[i] >= 0 && zone_gray_levels[i] < num_bins) {
                pairs[pair_idx].gray_idx = zone_gray_levels[i];
                pairs[pair_idx].zone_size = zone_sizes[i];
                pair_idx++;
            }
        }

        qsort(pairs, (size_t)valid_zones, sizeof(GLSZMPair), compare_glszm_pairs);

        glszm->entry_gray_idx = (int*)malloc((size_t)valid_zones * sizeof(int));
        glszm->entry_zone_size = (int*)malloc((size_t)valid_zones * sizeof(int));
        glszm->entry_count = (double*)malloc((size_t)valid_zones * sizeof(double));
        if (!glszm->entry_gray_idx || !glszm->entry_zone_size || !glszm->entry_count) {
            free(pairs);
            free(glszm->entry_gray_idx);
            free(glszm->entry_zone_size);
            free(glszm->entry_count);
            free(zone_sizes);
            free(zone_gray_levels);
            free(glszm->ps);
            free(glszm->pg);
            free(glszm);
            return NULL;
        }

        int entry_idx = 0;
        int current_gray = pairs[0].gray_idx;
        int current_size = pairs[0].zone_size;
        double current_count = 1.0;
        for (int i = 1; i < valid_zones; i++) {
            if (pairs[i].gray_idx == current_gray && pairs[i].zone_size == current_size) {
                current_count += 1.0;
            } else {
                glszm->entry_gray_idx[entry_idx] = current_gray;
                glszm->entry_zone_size[entry_idx] = current_size;
                glszm->entry_count[entry_idx] = current_count;
                glszm->pg[current_gray] += current_count;
                glszm->ps[current_size - 1] += current_count;
                glszm->total_zones += current_count;
                entry_idx++;

                current_gray = pairs[i].gray_idx;
                current_size = pairs[i].zone_size;
                current_count = 1.0;
            }
        }
        glszm->entry_gray_idx[entry_idx] = current_gray;
        glszm->entry_zone_size[entry_idx] = current_size;
        glszm->entry_count[entry_idx] = current_count;
        glszm->pg[current_gray] += current_count;
        glszm->ps[current_size - 1] += current_count;
        glszm->total_zones += current_count;
        entry_idx++;
        glszm->num_entries = entry_idx;

        free(pairs);
    }

    free(zone_sizes);
    free(zone_gray_levels);
    return glszm;
}

// Free GLSZM matrix
static void free_glszm_matrix(GLSZMMatrix* glszm) {
    if (glszm) {
        if (glszm->matrix) {
            for (int i = 0; i < glszm->num_bins; i++) {
                free(glszm->matrix[i]);
            }
            free(glszm->matrix);
        }
        free(glszm->entry_gray_idx);
        free(glszm->entry_zone_size);
        free(glszm->entry_count);
        free(glszm->pg);
        free(glszm->ps);
        free(glszm);
    }
}

static GLSZMFeatureValues compute_glszm_features_sparse(const GLSZMMatrix* glszm, int num_voxels_in_roi) {
    GLSZMFeatureValues values;
    memset(&values, 0, sizeof(values));

    const double total_zones = glszm->total_zones;
    values.zone_percentage = (num_voxels_in_roi > 0) ? (total_zones / (double)num_voxels_in_roi) : 0.0;
    if (total_zones <= EPSILON || glszm->num_entries <= 0) {
        return values;
    }

    const double inv_total_zones = 1.0 / total_zones;
    double sae = 0.0;
    double lae = 0.0;
    double lglze = 0.0;
    double hglze = 0.0;
    double salgle = 0.0;
    double sahgle = 0.0;
    double lalgle = 0.0;
    double lahgle = 0.0;
    double gray_mean = 0.0;
    double size_mean = 0.0;
    double entropy = 0.0;

    for (int e = 0; e < glszm->num_entries; e++) {
        const double count = glszm->entry_count[e];
        const double gray = (double)(glszm->entry_gray_idx[e] + 1);
        const double size = (double)glszm->entry_zone_size[e];
        const double gray_sq = gray * gray;
        const double size_sq = size * size;
        const double prob = count * inv_total_zones;

        sae += count / size_sq;
        lae += count * size_sq;
        lglze += count / gray_sq;
        hglze += count * gray_sq;
        salgle += count / (gray_sq * size_sq);
        sahgle += count * gray_sq / size_sq;
        lalgle += count * size_sq / gray_sq;
        lahgle += count * gray_sq * size_sq;
        gray_mean += count * gray;
        size_mean += count * size;
        entropy -= prob * log2(prob);
    }

    gray_mean *= inv_total_zones;
    size_mean *= inv_total_zones;

    double gray_variance = 0.0;
    double size_variance = 0.0;
    for (int e = 0; e < glszm->num_entries; e++) {
        const double count = glszm->entry_count[e];
        const double gray = (double)(glszm->entry_gray_idx[e] + 1);
        const double size = (double)glszm->entry_zone_size[e];
        const double gray_diff = gray - gray_mean;
        const double size_diff = size - size_mean;
        gray_variance += count * gray_diff * gray_diff;
        size_variance += count * size_diff * size_diff;
    }

    double gln_sq_sum = 0.0;
    for (int i = 0; i < glszm->num_bins; i++) {
        const double row = glszm->pg[i];
        gln_sq_sum += row * row;
    }

    double szn_sq_sum = 0.0;
    for (int j = 0; j < glszm->max_zone_size; j++) {
        const double col = glszm->ps[j];
        szn_sq_sum += col * col;
    }

    values.small_area_emphasis = sae * inv_total_zones;
    values.large_area_emphasis = lae * inv_total_zones;
    values.low_gray_level_zone_emphasis = lglze * inv_total_zones;
    values.high_gray_level_zone_emphasis = hglze * inv_total_zones;
    values.small_area_low_gray_level_emphasis = salgle * inv_total_zones;
    values.small_area_high_gray_level_emphasis = sahgle * inv_total_zones;
    values.large_area_low_gray_level_emphasis = lalgle * inv_total_zones;
    values.large_area_high_gray_level_emphasis = lahgle * inv_total_zones;
    values.gray_level_non_uniformity = gln_sq_sum * inv_total_zones;
    values.gray_level_non_uniformity_normalized = gln_sq_sum * inv_total_zones * inv_total_zones;
    values.size_zone_non_uniformity = szn_sq_sum * inv_total_zones;
    values.size_zone_non_uniformity_normalized = szn_sq_sum * inv_total_zones * inv_total_zones;
    values.gray_level_variance = gray_variance * inv_total_zones;
    values.size_zone_variance = size_variance * inv_total_zones;
    values.zone_entropy = entropy;

    return values;
}

// Task 8.4: Main API function for GLSZM feature extraction
static FlashRadiomicsResult* flash_radiomics_glszm_cpu_internal(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    double bin_width,
    int bin_count,
    const int* discretized_input,
    int ng_input
) {
    clock_t start_time = clock();
    
    // Allocate result structure
    FlashRadiomicsResult* result = (FlashRadiomicsResult*)malloc(sizeof(FlashRadiomicsResult));
    if (!result) return NULL;
    
    result->features = NULL;
    result->count = 0;
    result->compute_time_ms = 0.0;
    result->error_code = FLASH_RADIOMICS_SUCCESS;
    result->error_msg[0] = '\0';
    
    // Validate inputs
    if (!img || !mask) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
                           "Image or mask is NULL");
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
    
    // Set number of threads
    int actual_threads = num_threads;
    if (actual_threads <= 0) {
#ifdef _OPENMP
        actual_threads = omp_get_max_threads();
#else
        actual_threads = 1;
#endif
    }
    
#ifdef _OPENMP
    omp_set_num_threads(actual_threads);
#endif
    
    // Discretize image
    double min_val = 0.0, max_val = 0.0;
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
                           "Failed to discretize image for GLSZM");
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        return result;
    }
    
    // Label connected components (zones)
    int num_zones = 0;
    int* labels = NULL;
    
    if (img->ndim == 2) {
        labels = label_zones_2d((int*)discretized, mask->data, img->dims[0], img->dims[1], &num_zones);
    } else {
        labels = label_zones_3d((int*)discretized, mask->data, img->dims[0], img->dims[1], 
                               img->dims[2], &num_zones);
    }
    
    if (!labels) {
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate memory for zone labels");
        return result;
    }
    
    // Build GLSZM matrix
    GLSZMMatrix* glszm = build_glszm_matrix((int*)discretized, labels, mask->data,
                                            img->ndim, img->dims, ng,
                                            num_zones, min_val, max_val);
    if (discretized_owned) {
        flash_radiomics_free(discretized_owned);
    }
    free(labels);
    
    if (!glszm) {
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate memory for GLSZM matrix");
        return result;
    }
    
    // Count voxels in ROI for zone percentage
    int num_voxels_in_roi = 0;
    size_t total_voxels = flash_radiomics_get_num_elements(img->ndim, img->dims);
    for (size_t i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            num_voxels_in_roi++;
        }
    }
    
    // Compute features
    const int num_features = 16;
    result->features = (FlashRadiomicsFeature*)malloc(num_features * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        free_glszm_matrix(glszm);
        flash_radiomics_set_error(&result->error_code, result->error_msg,
                           FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
                           "Failed to allocate memory for features");
        return result;
    }
    
    result->count = num_features;
    int idx = 0;
    
    GLSZMFeatureValues values = compute_glszm_features_sparse(glszm, num_voxels_in_roi);

    // Task 8.2: Emphasis features
    strcpy(result->features[idx].name, "glszm_SmallAreaEmphasis");
    result->features[idx++].value = values.small_area_emphasis;
    
    strcpy(result->features[idx].name, "glszm_LargeAreaEmphasis");
    result->features[idx++].value = values.large_area_emphasis;
    
    strcpy(result->features[idx].name, "glszm_LowGrayLevelZoneEmphasis");
    result->features[idx++].value = values.low_gray_level_zone_emphasis;
    
    strcpy(result->features[idx].name, "glszm_HighGrayLevelZoneEmphasis");
    result->features[idx++].value = values.high_gray_level_zone_emphasis;
    
    strcpy(result->features[idx].name, "glszm_SmallAreaLowGrayLevelEmphasis");
    result->features[idx++].value = values.small_area_low_gray_level_emphasis;
    
    strcpy(result->features[idx].name, "glszm_SmallAreaHighGrayLevelEmphasis");
    result->features[idx++].value = values.small_area_high_gray_level_emphasis;
    
    strcpy(result->features[idx].name, "glszm_LargeAreaLowGrayLevelEmphasis");
    result->features[idx++].value = values.large_area_low_gray_level_emphasis;
    
    strcpy(result->features[idx].name, "glszm_LargeAreaHighGrayLevelEmphasis");
    result->features[idx++].value = values.large_area_high_gray_level_emphasis;
    
    // Task 8.3: Uniformity and variance features
    strcpy(result->features[idx].name, "glszm_GrayLevelNonUniformity");
    result->features[idx++].value = values.gray_level_non_uniformity;
    
    strcpy(result->features[idx].name, "glszm_GrayLevelNonUniformityNormalized");
    result->features[idx++].value = values.gray_level_non_uniformity_normalized;
    
    strcpy(result->features[idx].name, "glszm_SizeZoneNonUniformity");
    result->features[idx++].value = values.size_zone_non_uniformity;
    
    strcpy(result->features[idx].name, "glszm_SizeZoneNonUniformityNormalized");
    result->features[idx++].value = values.size_zone_non_uniformity_normalized;
    
    strcpy(result->features[idx].name, "glszm_ZonePercentage");
    result->features[idx++].value = values.zone_percentage;
    
    strcpy(result->features[idx].name, "glszm_GrayLevelVariance");
    result->features[idx++].value = values.gray_level_variance;
    
    strcpy(result->features[idx].name, "glszm_SizeZoneVariance");
    result->features[idx++].value = values.size_zone_variance;
    
    strcpy(result->features[idx].name, "glszm_ZoneEntropy");
    result->features[idx++].value = values.zone_entropy;
    
    // Clean up
    free_glszm_matrix(glszm);
    
    // Record computation time
    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    return result;
}

FlashRadiomicsResult* flash_radiomics_glszm_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    double bin_width,
    int bin_count
) {
    return flash_radiomics_glszm_cpu_internal(
        img, mask, num_threads, bin_width, bin_count, NULL, 0
    );
}

FlashRadiomicsResult* flash_radiomics_glszm_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    const int* discretized,
    int ng
) {
    return flash_radiomics_glszm_cpu_internal(
        img, mask, num_threads, 0.0, 0, discretized, ng
    );
}

typedef struct {
    int ndim;
    int dims[3];
    int num_bins;
    size_t total_voxels;
    int max_zone_size;
    int max_forward_neighbors;
    int* labels;
    int* uf_parent;
    int* uf_rank;
    int* root_to_label;
    int* zone_sizes;
    int* zone_gray_levels;
    int* neighbor_indices;
    uint8_t* neighbor_counts;
    double* pg;
    double* ps;
    double* zone_counts;
    double* gray_sq;
    double* gray_inv_sq;
    double* size_sq;
    double* size_inv_sq;
} GLSZMVoxelWorkspace;

static int init_glszm_workspace_neighbors(GLSZMVoxelWorkspace* ws) {
    if (!ws) {
        return 0;
    }

    ws->max_forward_neighbors = (ws->ndim == 2) ? 4 : 13;
    ws->neighbor_indices = (int*)malloc(
        ws->total_voxels * (size_t)ws->max_forward_neighbors * sizeof(int)
    );
    ws->neighbor_counts = (uint8_t*)malloc(ws->total_voxels * sizeof(uint8_t));
    if (!ws->neighbor_indices || !ws->neighbor_counts) {
        return 0;
    }

    if (ws->ndim == 2) {
        const int dx[4] = {1, -1, 0, 1};
        const int dy[4] = {0, 1, 1, 1};
        const int dims_x = ws->dims[0];
        const int dims_y = ws->dims[1];

        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                size_t idx = (size_t)y * (size_t)dims_x + (size_t)x;
                size_t base = idx * (size_t)ws->max_forward_neighbors;
                uint8_t count = 0;
                for (int n = 0; n < 4; n++) {
                    int nx = x + dx[n];
                    int ny = y + dy[n];
                    if (nx >= 0 && nx < dims_x && ny >= 0 && ny < dims_y) {
                        ws->neighbor_indices[base + (size_t)count] =
                            ny * dims_x + nx;
                        count++;
                    }
                }
                ws->neighbor_counts[idx] = count;
            }
        }
    } else {
        const int dx[13] = {
            1,
            -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1,
            -1, 0, 1
        };
        const int dy[13] = {
            0,
            1, 1, 1,
            -1, -1, -1,
            0, 0, 0,
            1, 1, 1
        };
        const int dz[13] = {
            0,
            0, 0, 0,
            1, 1, 1,
            1, 1, 1,
            1, 1, 1
        };
        const int dims_x = ws->dims[0];
        const int dims_y = ws->dims[1];
        const int dims_z = ws->dims[2];
        const size_t plane = (size_t)dims_x * (size_t)dims_y;

        for (size_t idx = 0; idx < ws->total_voxels; idx++) {
            int z = (int)(idx / plane);
            size_t rem = idx - ((size_t)z * plane);
            int y = (int)(rem / (size_t)dims_x);
            int x = (int)(rem - (size_t)y * (size_t)dims_x);
            size_t base = idx * (size_t)ws->max_forward_neighbors;
            uint8_t count = 0;
            for (int n = 0; n < 13; n++) {
                int nx = x + dx[n];
                int ny = y + dy[n];
                int nz = z + dz[n];
                if (nx >= 0 && nx < dims_x &&
                    ny >= 0 && ny < dims_y &&
                    nz >= 0 && nz < dims_z) {
                    ws->neighbor_indices[base + (size_t)count] =
                        (nz * dims_y + ny) * dims_x + nx;
                    count++;
                }
            }
            ws->neighbor_counts[idx] = count;
        }
    }

    return 1;
}

static int init_glszm_voxel_workspace(
    GLSZMVoxelWorkspace* ws,
    int ndim,
    const int dims[3],
    int num_bins
) {
    if (!ws || !dims || num_bins <= 0) {
        return 0;
    }
    memset(ws, 0, sizeof(*ws));
    ws->ndim = ndim;
    ws->num_bins = num_bins;
    for (int axis = 0; axis < 3; axis++) {
        ws->dims[axis] = (axis < ndim) ? dims[axis] : 1;
    }
    ws->total_voxels = flash_radiomics_get_num_elements(ndim, ws->dims);
    ws->max_zone_size = (int)ws->total_voxels;

    ws->labels = (int*)malloc(ws->total_voxels * sizeof(int));
    ws->uf_parent = (int*)malloc(ws->total_voxels * sizeof(int));
    ws->uf_rank = (int*)malloc(ws->total_voxels * sizeof(int));
    ws->root_to_label = (int*)malloc(ws->total_voxels * sizeof(int));
    ws->zone_sizes = (int*)malloc(ws->total_voxels * sizeof(int));
    ws->zone_gray_levels = (int*)malloc(ws->total_voxels * sizeof(int));
    ws->pg = (double*)malloc((size_t)ws->num_bins * sizeof(double));
    ws->ps = (double*)malloc((size_t)(ws->max_zone_size + 1) * sizeof(double));
    ws->gray_sq = (double*)malloc((size_t)ws->num_bins * sizeof(double));
    ws->gray_inv_sq = (double*)malloc((size_t)ws->num_bins * sizeof(double));
    ws->size_sq = (double*)malloc((size_t)(ws->max_zone_size + 1) * sizeof(double));
    ws->size_inv_sq = (double*)malloc((size_t)(ws->max_zone_size + 1) * sizeof(double));

    size_t zone_count_len = (size_t)ws->num_bins * (size_t)(ws->max_zone_size + 1);
    ws->zone_counts = (double*)malloc(zone_count_len * sizeof(double));

    if (!ws->labels || !ws->uf_parent || !ws->uf_rank || !ws->root_to_label ||
        !ws->zone_sizes || !ws->zone_gray_levels || !ws->pg || !ws->ps ||
        !ws->zone_counts || !ws->gray_sq || !ws->gray_inv_sq ||
        !ws->size_sq || !ws->size_inv_sq) {
        return 0;
    }

    for (int i = 0; i < ws->num_bins; i++) {
        double gray = (double)(i + 1);
        double gray_sq = gray * gray;
        ws->gray_sq[i] = gray_sq;
        ws->gray_inv_sq[i] = 1.0 / gray_sq;
    }

    ws->size_sq[0] = 0.0;
    ws->size_inv_sq[0] = 0.0;
    for (int s = 1; s <= ws->max_zone_size; s++) {
        double size = (double)s;
        double size_sq = size * size;
        ws->size_sq[s] = size_sq;
        ws->size_inv_sq[s] = 1.0 / size_sq;
    }

    if (!init_glszm_workspace_neighbors(ws)) {
        return 0;
    }

    return 1;
}

static void free_glszm_voxel_workspace(GLSZMVoxelWorkspace* ws) {
    if (!ws) return;
    free(ws->size_inv_sq);
    free(ws->size_sq);
    free(ws->gray_inv_sq);
    free(ws->gray_sq);
    free(ws->zone_counts);
    free(ws->ps);
    free(ws->pg);
    free(ws->neighbor_counts);
    free(ws->neighbor_indices);
    free(ws->zone_gray_levels);
    free(ws->zone_sizes);
    free(ws->root_to_label);
    free(ws->uf_rank);
    free(ws->uf_parent);
    free(ws->labels);
    memset(ws, 0, sizeof(*ws));
}

static int uf_find_ws(GLSZMVoxelWorkspace* ws, int x) {
    int root = x;
    while (ws->uf_parent[root] != root) {
        root = ws->uf_parent[root];
    }
    while (ws->uf_parent[x] != x) {
        int next = ws->uf_parent[x];
        ws->uf_parent[x] = root;
        x = next;
    }
    return root;
}

static void uf_union_ws(GLSZMVoxelWorkspace* ws, int x, int y) {
    int root_x = uf_find_ws(ws, x);
    int root_y = uf_find_ws(ws, y);
    if (root_x == root_y) {
        return;
    }
    if (ws->uf_rank[root_x] < ws->uf_rank[root_y]) {
        ws->uf_parent[root_x] = root_y;
    } else if (ws->uf_rank[root_x] > ws->uf_rank[root_y]) {
        ws->uf_parent[root_y] = root_x;
    } else {
        ws->uf_parent[root_y] = root_x;
        ws->uf_rank[root_x]++;
    }
}

static int label_zones_workspace(
    GLSZMVoxelWorkspace* ws,
    const int* discretized,
    const uint8_t* mask,
    int* num_zones_out,
    int* num_voxels_in_roi_out
) {
    if (!ws || !discretized || !mask || !num_zones_out || !num_voxels_in_roi_out) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    for (size_t i = 0; i < ws->total_voxels; i++) {
        ws->labels[i] = -1;
        ws->uf_parent[i] = (int)i;
        ws->uf_rank[i] = 0;
    }

    for (size_t idx = 0; idx < ws->total_voxels; idx++) {
        if (mask[idx] == 0) {
            continue;
        }
        int gray_level = discretized[idx];
        size_t base = idx * (size_t)ws->max_forward_neighbors;
        uint8_t count = ws->neighbor_counts[idx];
        for (uint8_t n = 0; n < count; n++) {
            int nidx = ws->neighbor_indices[base + (size_t)n];
            if (mask[(size_t)nidx] != 0 && discretized[(size_t)nidx] == gray_level) {
                uf_union_ws(ws, (int)idx, nidx);
            }
        }
    }

    for (size_t i = 0; i < ws->total_voxels; i++) {
        ws->root_to_label[i] = -1;
    }
    memset(ws->zone_sizes, 0, ws->total_voxels * sizeof(int));
    memset(ws->zone_gray_levels, 0xFF, ws->total_voxels * sizeof(int));

    int next_label = 0;
    int num_voxels_in_roi = 0;
    for (size_t i = 0; i < ws->total_voxels; i++) {
        if (mask[i] != 0) {
            int root = uf_find_ws(ws, (int)i);
            int zone = ws->root_to_label[root];
            if (zone == -1) {
                zone = next_label++;
                ws->root_to_label[root] = zone;
            }
            ws->labels[i] = zone;
            ws->zone_sizes[zone]++;
            if (ws->zone_gray_levels[zone] == -1) {
                ws->zone_gray_levels[zone] = discretized[i] - 1;
            }
            num_voxels_in_roi++;
        }
    }

    *num_zones_out = next_label;
    *num_voxels_in_roi_out = num_voxels_in_roi;
    return FLASH_RADIOMICS_SUCCESS;
}

static int is_glszm_feature_enabled(const uint8_t* include_features, int idx) {
    return (!include_features) || (include_features[idx] != 0);
}

static int compute_glszm_features_workspace(
    GLSZMVoxelWorkspace* ws,
    const int* discretized,
    const uint8_t* mask,
    const uint8_t* include_features,
    GLSZMFeatureValues* out_values
) {
    if (!ws || !discretized || !mask || !out_values) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    memset(out_values, 0, sizeof(*out_values));

    const int need_sae = is_glszm_feature_enabled(include_features, 0);
    const int need_lae = is_glszm_feature_enabled(include_features, 1);
    const int need_lglze = is_glszm_feature_enabled(include_features, 2);
    const int need_hglze = is_glszm_feature_enabled(include_features, 3);
    const int need_salgle = is_glszm_feature_enabled(include_features, 4);
    const int need_sahgle = is_glszm_feature_enabled(include_features, 5);
    const int need_lalgle = is_glszm_feature_enabled(include_features, 6);
    const int need_lahgle = is_glszm_feature_enabled(include_features, 7);
    const int need_gln = is_glszm_feature_enabled(include_features, 8);
    const int need_glnn = is_glszm_feature_enabled(include_features, 9);
    const int need_szn = is_glszm_feature_enabled(include_features, 10);
    const int need_sznn = is_glszm_feature_enabled(include_features, 11);
    const int need_zp = is_glszm_feature_enabled(include_features, 12);
    const int need_glv = is_glszm_feature_enabled(include_features, 13);
    const int need_zv = is_glszm_feature_enabled(include_features, 14);
    const int need_ze = is_glszm_feature_enabled(include_features, 15);

    const int need_pg = need_gln || need_glnn;
    const int need_ps = need_szn || need_sznn;
    const int need_counts = need_ze;
    const int need_gray_moments = need_glv;
    const int need_size_moments = need_zv;

    int num_zones = 0;
    int num_voxels_in_roi = 0;
    int status = label_zones_workspace(
        ws,
        discretized,
        mask,
        &num_zones,
        &num_voxels_in_roi
    );
    if (status != FLASH_RADIOMICS_SUCCESS) {
        return status;
    }

    if (need_pg) {
        memset(ws->pg, 0, (size_t)ws->num_bins * sizeof(double));
    }
    if (need_ps) {
        memset(ws->ps, 0, (size_t)(ws->max_zone_size + 1) * sizeof(double));
    }
    if (need_counts) {
        memset(
            ws->zone_counts,
            0,
            (size_t)ws->num_bins * (size_t)(ws->max_zone_size + 1) * sizeof(double)
        );
    }

    double total_zones = 0.0;
    int max_zone_size_present = 0;
    double sae = 0.0;
    double lae = 0.0;
    double lglze = 0.0;
    double hglze = 0.0;
    double salgle = 0.0;
    double sahgle = 0.0;
    double lalgle = 0.0;
    double lahgle = 0.0;
    double sum_gray = 0.0;
    double sum_size = 0.0;
    double sum_gray_sq = 0.0;
    double sum_size_sq = 0.0;

    for (int zone = 0; zone < num_zones; zone++) {
        int zone_size = ws->zone_sizes[zone];
        int gray_idx = ws->zone_gray_levels[zone];
        if (zone_size <= 0 || gray_idx < 0 || gray_idx >= ws->num_bins) {
            continue;
        }
        if (zone_size > ws->max_zone_size) {
            continue;
        }

        double gray = (double)(gray_idx + 1);
        double size = (double)zone_size;
        double gray_sq = ws->gray_sq[gray_idx];
        double gray_inv_sq = ws->gray_inv_sq[gray_idx];
        double size_sq = ws->size_sq[zone_size];
        double size_inv_sq = ws->size_inv_sq[zone_size];

        if (need_counts) {
            size_t count_idx =
                (size_t)gray_idx * (size_t)(ws->max_zone_size + 1) + (size_t)zone_size;
            ws->zone_counts[count_idx] += 1.0;
        }
        if (need_pg) {
            ws->pg[gray_idx] += 1.0;
        }
        if (need_ps) {
            ws->ps[zone_size] += 1.0;
        }
        total_zones += 1.0;
        if (zone_size > max_zone_size_present) {
            max_zone_size_present = zone_size;
        }

        if (need_sae) sae += size_inv_sq;
        if (need_lae) lae += size_sq;
        if (need_lglze) lglze += gray_inv_sq;
        if (need_hglze) hglze += gray_sq;
        if (need_salgle) salgle += gray_inv_sq * size_inv_sq;
        if (need_sahgle) sahgle += gray_sq * size_inv_sq;
        if (need_lalgle) lalgle += size_sq * gray_inv_sq;
        if (need_lahgle) lahgle += gray_sq * size_sq;
        if (need_gray_moments) {
            sum_gray += gray;
            sum_gray_sq += gray_sq;
        }
        if (need_size_moments) {
            sum_size += size;
            sum_size_sq += size_sq;
        }
    }

    if (need_zp) {
        out_values->zone_percentage = (num_voxels_in_roi > 0)
            ? (total_zones / (double)num_voxels_in_roi)
            : 0.0;
    }
    if (total_zones <= EPSILON) {
        return FLASH_RADIOMICS_SUCCESS;
    }

    double inv_total_zones = 1.0 / total_zones;

    if (need_sae) out_values->small_area_emphasis = sae * inv_total_zones;
    if (need_lae) out_values->large_area_emphasis = lae * inv_total_zones;
    if (need_lglze) out_values->low_gray_level_zone_emphasis = lglze * inv_total_zones;
    if (need_hglze) out_values->high_gray_level_zone_emphasis = hglze * inv_total_zones;
    if (need_salgle) {
        out_values->small_area_low_gray_level_emphasis = salgle * inv_total_zones;
    }
    if (need_sahgle) {
        out_values->small_area_high_gray_level_emphasis = sahgle * inv_total_zones;
    }
    if (need_lalgle) {
        out_values->large_area_low_gray_level_emphasis = lalgle * inv_total_zones;
    }
    if (need_lahgle) {
        out_values->large_area_high_gray_level_emphasis = lahgle * inv_total_zones;
    }

    if (need_gln || need_glnn) {
        double gln_sq_sum = 0.0;
        for (int i = 0; i < ws->num_bins; i++) {
            gln_sq_sum += ws->pg[i] * ws->pg[i];
        }
        if (need_gln) out_values->gray_level_non_uniformity = gln_sq_sum * inv_total_zones;
        if (need_glnn) {
            out_values->gray_level_non_uniformity_normalized =
                gln_sq_sum * inv_total_zones * inv_total_zones;
        }
    }

    if (need_szn || need_sznn) {
        double szn_sq_sum = 0.0;
        for (int s = 1; s <= max_zone_size_present; s++) {
            szn_sq_sum += ws->ps[s] * ws->ps[s];
        }
        if (need_szn) out_values->size_zone_non_uniformity = szn_sq_sum * inv_total_zones;
        if (need_sznn) {
            out_values->size_zone_non_uniformity_normalized =
                szn_sq_sum * inv_total_zones * inv_total_zones;
        }
    }

    if (need_gray_moments) {
        double gray_mean = sum_gray * inv_total_zones;
        double gray_variance = sum_gray_sq * inv_total_zones - gray_mean * gray_mean;
        if (gray_variance < 0.0 && gray_variance > -1e-12) {
            gray_variance = 0.0;
        }
        out_values->gray_level_variance = gray_variance;
    }

    if (need_size_moments) {
        double size_mean = sum_size * inv_total_zones;
        double size_variance = sum_size_sq * inv_total_zones - size_mean * size_mean;
        if (size_variance < 0.0 && size_variance > -1e-12) {
            size_variance = 0.0;
        }
        out_values->size_zone_variance = size_variance;
    }

    if (need_ze) {
        double entropy = 0.0;
        for (int i = 0; i < ws->num_bins; i++) {
            size_t row_offset = (size_t)i * (size_t)(ws->max_zone_size + 1);
            for (int s = 1; s <= max_zone_size_present; s++) {
                double count = ws->zone_counts[row_offset + (size_t)s];
                if (count > 0.0) {
                    double prob = count * inv_total_zones;
                    entropy -= prob * log2(prob);
                }
            }
        }
        out_values->zone_entropy = entropy;
    }

    return FLASH_RADIOMICS_SUCCESS;
}

int flash_radiomics_glszm_cpu_discretized_voxel_batch(
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
    if (!discretized_windows || !mask_windows || !window_dims || !out_features) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (batch_size <= 0 || ng <= 0) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (!flash_radiomics_validate_dims_with_ndim(ndim, window_dims)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    int selected_feature_indices[FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT];
    int selected_feature_count = 0;
    for (int idx = 0; idx < FLASH_RADIOMICS_GLSZM_VOXEL_FEATURE_COUNT; idx++) {
        if (is_glszm_feature_enabled(include_features, idx)) {
            selected_feature_indices[selected_feature_count++] = idx;
        }
    }
    if (selected_feature_count <= 0 || out_feature_stride < selected_feature_count) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    GLSZMVoxelWorkspace workspace;
    if (!init_glszm_voxel_workspace(&workspace, ndim, window_dims, ng)) {
        free_glszm_voxel_workspace(&workspace);
        return FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
    }

    size_t window_voxels = flash_radiomics_get_num_elements(ndim, window_dims);
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        const int* batch_discretized =
            discretized_windows + ((size_t)batch_idx * window_voxels);
        const uint8_t* batch_mask =
            mask_windows + ((size_t)batch_idx * window_voxels);
        double* batch_output =
            out_features + ((size_t)batch_idx * (size_t)out_feature_stride);

        GLSZMFeatureValues values;
        memset(&values, 0, sizeof(values));
        int status = compute_glszm_features_workspace(
            &workspace,
            batch_discretized,
            batch_mask,
            include_features,
            &values
        );
        if (status != FLASH_RADIOMICS_SUCCESS) {
            free_glszm_voxel_workspace(&workspace);
            return status;
        }

        for (int out_idx = 0; out_idx < selected_feature_count; out_idx++) {
            switch (selected_feature_indices[out_idx]) {
                case 0: batch_output[out_idx] = values.small_area_emphasis; break;
                case 1: batch_output[out_idx] = values.large_area_emphasis; break;
                case 2: batch_output[out_idx] = values.low_gray_level_zone_emphasis; break;
                case 3: batch_output[out_idx] = values.high_gray_level_zone_emphasis; break;
                case 4: batch_output[out_idx] = values.small_area_low_gray_level_emphasis; break;
                case 5: batch_output[out_idx] = values.small_area_high_gray_level_emphasis; break;
                case 6: batch_output[out_idx] = values.large_area_low_gray_level_emphasis; break;
                case 7: batch_output[out_idx] = values.large_area_high_gray_level_emphasis; break;
                case 8: batch_output[out_idx] = values.gray_level_non_uniformity; break;
                case 9: batch_output[out_idx] = values.gray_level_non_uniformity_normalized; break;
                case 10: batch_output[out_idx] = values.size_zone_non_uniformity; break;
                case 11: batch_output[out_idx] = values.size_zone_non_uniformity_normalized; break;
                case 12: batch_output[out_idx] = values.zone_percentage; break;
                case 13: batch_output[out_idx] = values.gray_level_variance; break;
                case 14: batch_output[out_idx] = values.size_zone_variance; break;
                case 15: batch_output[out_idx] = values.zone_entropy; break;
                default: batch_output[out_idx] = NAN; break;
            }
        }
    }

    free_glszm_voxel_workspace(&workspace);
    return FLASH_RADIOMICS_SUCCESS;
}
