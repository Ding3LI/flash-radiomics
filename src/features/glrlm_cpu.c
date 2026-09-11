#include "glrlm_cpu.h"
#include "../core/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <limits.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define EPSILON 2.2e-16

// Direction vectors for 2D (4 directions: 0°, 45°, 90°, 135°)
static const int DIRECTIONS_2D[4][2] = {
    {1, 0},   // 0° (horizontal)
    {1, 1},   // 45° (diagonal)
    {0, 1},   // 90° (vertical)
    {-1, 1}   // 135° (anti-diagonal)
};

// Direction vectors for 3D (13 directions)
static const int DIRECTIONS_3D[13][3] = {
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

// Helper structure for GLRLM computation
typedef struct {
    double*** matrices;      // GLRLM matrices [direction][i][j] where i=gray level, j=run length
    double* matrix_data;     // Contiguous matrix storage [direction][i][j]
    int num_bins;           // Number of gray levels (Ng)
    int max_run_length;     // Maximum run length (Nr)
    int num_directions;     // Number of directions (4 for 2D, 13 for 3D)
    double min_val;         // Minimum value in ROI
    double max_val;         // Maximum value in ROI
    double bin_width;       // Width of each bin
} GLRLMMatrices;

typedef struct {
    int min_x;
    int max_x;
    int min_y;
    int max_y;
    int min_z;
    int max_z;
    int has_roi;
} ROIBounds;

// Task 7.1: Discretize image values into bins
static int* discretize_image(FlashRadiomicsImage* img, FlashRadiomicsMask* mask, 
                             int num_bins, double* min_val, double* max_val) {
    int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
    int* discretized = (int*)malloc(total_voxels * sizeof(int));
    if (!discretized) return NULL;
    
    // Find min/max in ROI
    *min_val = DBL_MAX;
    *max_val = -DBL_MAX;
    
    for (int i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            float val = img->data[i];
            if (val < *min_val) *min_val = val;
            if (val > *max_val) *max_val = val;
        }
    }
    
    double bin_width = (*max_val - *min_val) / num_bins;
    if (bin_width == 0) bin_width = 1.0;
    
    // Discretize values
    for (int i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            float val = img->data[i];
            int bin_idx = (int)((val - *min_val) / bin_width);
            if (bin_idx >= num_bins) bin_idx = num_bins - 1;
            if (bin_idx < 0) bin_idx = 0;
            discretized[i] = bin_idx;
        } else {
            discretized[i] = -1;  // Mark as outside ROI
        }
    }
    
    return discretized;
}

static ROIBounds compute_roi_bounds(const uint8_t* mask, int dims_x, int dims_y, int dims_z) {
    ROIBounds roi;
    roi.min_x = dims_x;
    roi.min_y = dims_y;
    roi.min_z = dims_z;
    roi.max_x = -1;
    roi.max_y = -1;
    roi.max_z = -1;
    roi.has_roi = 0;

    const int slice_stride = dims_x * dims_y;
    for (int z = 0; z < dims_z; z++) {
        for (int y = 0; y < dims_y; y++) {
            for (int x = 0; x < dims_x; x++) {
                const int idx = z * slice_stride + y * dims_x + x;
                if (mask[idx] == 0) {
                    continue;
                }
                roi.has_roi = 1;
                if (x < roi.min_x) roi.min_x = x;
                if (x > roi.max_x) roi.max_x = x;
                if (y < roi.min_y) roi.min_y = y;
                if (y > roi.max_y) roi.max_y = y;
                if (z < roi.min_z) roi.min_z = z;
                if (z > roi.max_z) roi.max_z = z;
            }
        }
    }

    if (!roi.has_roi) {
        roi.min_x = roi.min_y = roi.min_z = 0;
        roi.max_x = roi.max_y = roi.max_z = -1;
    }
    return roi;
}

static int clip_line_2d_to_roi(
    int start_x,
    int start_y,
    int dx,
    int dy,
    int dims_x,
    int dims_y,
    const ROIBounds* roi,
    int* t_start,
    int* t_end
) {
    int len_x = (dx == 0) ? INT_MAX : ((dx > 0) ? (dims_x - start_x) : (start_x + 1));
    int len_y = (dy == 0) ? INT_MAX : ((dy > 0) ? (dims_y - start_y) : (start_y + 1));
    int line_len = (len_x < len_y) ? len_x : len_y;
    if (line_len <= 0) {
        return 0;
    }

    int lo = 0;
    int hi = line_len - 1;

    if (dx == 0) {
        if (start_x < roi->min_x || start_x > roi->max_x) {
            return 0;
        }
    } else if (dx > 0) {
        int a = roi->min_x - start_x;
        int b = roi->max_x - start_x;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    } else {
        int a = start_x - roi->max_x;
        int b = start_x - roi->min_x;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    }

    if (dy == 0) {
        if (start_y < roi->min_y || start_y > roi->max_y) {
            return 0;
        }
    } else if (dy > 0) {
        int a = roi->min_y - start_y;
        int b = roi->max_y - start_y;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    } else {
        int a = start_y - roi->max_y;
        int b = start_y - roi->min_y;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    }

    if (lo > hi) {
        return 0;
    }

    *t_start = lo;
    *t_end = hi;
    return 1;
}

static int clip_line_3d_to_roi(
    int start_x,
    int start_y,
    int start_z,
    int dx,
    int dy,
    int dz,
    int dims_x,
    int dims_y,
    int dims_z,
    const ROIBounds* roi,
    int* t_start,
    int* t_end
) {
    int len_x = (dx == 0) ? INT_MAX : ((dx > 0) ? (dims_x - start_x) : (start_x + 1));
    int len_y = (dy == 0) ? INT_MAX : ((dy > 0) ? (dims_y - start_y) : (start_y + 1));
    int len_z = (dz == 0) ? INT_MAX : ((dz > 0) ? (dims_z - start_z) : (start_z + 1));
    int line_len = len_x;
    if (len_y < line_len) line_len = len_y;
    if (len_z < line_len) line_len = len_z;
    if (line_len <= 0) {
        return 0;
    }

    int lo = 0;
    int hi = line_len - 1;

    if (dx == 0) {
        if (start_x < roi->min_x || start_x > roi->max_x) {
            return 0;
        }
    } else if (dx > 0) {
        int a = roi->min_x - start_x;
        int b = roi->max_x - start_x;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    } else {
        int a = start_x - roi->max_x;
        int b = start_x - roi->min_x;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    }

    if (dy == 0) {
        if (start_y < roi->min_y || start_y > roi->max_y) {
            return 0;
        }
    } else if (dy > 0) {
        int a = roi->min_y - start_y;
        int b = roi->max_y - start_y;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    } else {
        int a = start_y - roi->max_y;
        int b = start_y - roi->min_y;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    }

    if (dz == 0) {
        if (start_z < roi->min_z || start_z > roi->max_z) {
            return 0;
        }
    } else if (dz > 0) {
        int a = roi->min_z - start_z;
        int b = roi->max_z - start_z;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    } else {
        int a = start_z - roi->max_z;
        int b = start_z - roi->min_z;
        if (a > lo) lo = a;
        if (b < hi) hi = b;
    }

    if (lo > hi) {
        return 0;
    }

    *t_start = lo;
    *t_end = hi;
    return 1;
}

// Task 7.1: Compute run-length matrix for a single direction
static inline void add_run_count(double* matrix, int gray_level, int run_length, int max_run_length) {
    if (gray_level >= 0 && run_length > 0 && run_length <= max_run_length) {
        matrix[gray_level * max_run_length + (run_length - 1)]++;
    }
}

static int process_line_2d(const int* discretized, const uint8_t* mask, int dims_x, int dims_y,
                           const ROIBounds* roi, int start_x, int start_y, int dx, int dy, int max_run_length,
                           double* matrix) {
    int t_start = 0;
    int t_end = 0;
    if (!clip_line_2d_to_roi(
            start_x, start_y, dx, dy, dims_x, dims_y, roi, &t_start, &t_end
        )) {
        return 0;
    }

    int x = start_x + t_start * dx;
    int y = start_y + t_start * dy;
    int current_raw_level = -1;
    int run_length = 0;
    int elements = 0;

    for (int t = t_start; t <= t_end; t++) {
        const int idx = y * dims_x + x;
        const int raw_level = (mask[idx] != 0) ? discretized[idx] : -1;

        if (raw_level > 0) {
            elements++;
            if (raw_level == current_raw_level) {
                run_length++;
            } else {
                if (current_raw_level > 0) {
                    add_run_count(matrix, current_raw_level - 1, run_length, max_run_length);
                }
                current_raw_level = raw_level;
                run_length = 1;
            }
        } else if (current_raw_level > 0) {
            add_run_count(matrix, current_raw_level - 1, run_length, max_run_length);
            current_raw_level = -1;
            run_length = 0;
        }

        x += dx;
        y += dy;
    }

    if (current_raw_level > 0) {
        add_run_count(matrix, current_raw_level - 1, run_length, max_run_length);
    }

    return elements;
}

static void compute_runs_2d(int* discretized, uint8_t* mask, int dims_x, int dims_y,
                            const ROIBounds* roi,
                            int dx, int dy, int num_bins, int max_run_length,
                            double* matrix) {
    int multi_element = 0;
    if (!roi->has_roi) {
        return;
    }

    if (dx != 0) {
        const int start_x = (dx > 0) ? 0 : (dims_x - 1);
        for (int y = 0; y < dims_y; y++) {
            if (process_line_2d(discretized, mask, dims_x, dims_y, roi,
                                start_x, y, dx, dy, max_run_length, matrix) > 1) {
                multi_element = 1;
            }
        }
    }

    if (dy != 0) {
        const int start_y = (dy > 0) ? 0 : (dims_y - 1);
        int x_begin = 0;
        int x_end = dims_x;
        if (dx != 0) {
            const int boundary_x = (dx > 0) ? 0 : (dims_x - 1);
            if (boundary_x == 0) {
                x_begin = 1;
            } else {
                x_end = dims_x - 1;
            }
        }
        for (int x = x_begin; x < x_end; x++) {
            if (process_line_2d(discretized, mask, dims_x, dims_y, roi,
                                x, start_y, dx, dy, max_run_length, matrix) > 1) {
                multi_element = 1;
            }
        }
    }

    if (!multi_element) {
        memset(matrix, 0, (size_t)num_bins * (size_t)max_run_length * sizeof(double));
    }
}

static int process_line_3d(const int* discretized, const uint8_t* mask, int dims_x, int dims_y, int dims_z,
                           const ROIBounds* roi, int start_x, int start_y, int start_z, int dx, int dy, int dz,
                           int max_run_length, double* matrix) {
    int t_start = 0;
    int t_end = 0;
    if (!clip_line_3d_to_roi(
            start_x, start_y, start_z, dx, dy, dz, dims_x, dims_y, dims_z, roi, &t_start, &t_end
        )) {
        return 0;
    }

    const int slice_stride = dims_y * dims_x;
    int x = start_x + t_start * dx;
    int y = start_y + t_start * dy;
    int z = start_z + t_start * dz;
    int current_raw_level = -1;
    int run_length = 0;
    int elements = 0;

    for (int t = t_start; t <= t_end; t++) {
        const int idx = z * slice_stride + y * dims_x + x;
        const int raw_level = (mask[idx] != 0) ? discretized[idx] : -1;

        if (raw_level > 0) {
            elements++;
            if (raw_level == current_raw_level) {
                run_length++;
            } else {
                if (current_raw_level > 0) {
                    add_run_count(matrix, current_raw_level - 1, run_length, max_run_length);
                }
                current_raw_level = raw_level;
                run_length = 1;
            }
        } else if (current_raw_level > 0) {
            add_run_count(matrix, current_raw_level - 1, run_length, max_run_length);
            current_raw_level = -1;
            run_length = 0;
        }

        x += dx;
        y += dy;
        z += dz;
    }

    if (current_raw_level > 0) {
        add_run_count(matrix, current_raw_level - 1, run_length, max_run_length);
    }

    return elements;
}

static void compute_runs_3d(int* discretized, uint8_t* mask, int dims_x, int dims_y, int dims_z,
                            const ROIBounds* roi,
                            int dx, int dy, int dz, int num_bins, int max_run_length,
                            double* matrix) {
    int multi_element = 0;
    if (!roi->has_roi) {
        return;
    }

    if (dx != 0) {
        const int start_x = (dx > 0) ? 0 : (dims_x - 1);
        for (int z = 0; z < dims_z; z++) {
            for (int y = 0; y < dims_y; y++) {
                if (process_line_3d(discretized, mask, dims_x, dims_y, dims_z, roi,
                                    start_x, y, z, dx, dy, dz, max_run_length, matrix) > 1) {
                    multi_element = 1;
                }
            }
        }
    }

    if (dy != 0) {
        const int start_y = (dy > 0) ? 0 : (dims_y - 1);
        int x_begin = 0;
        int x_end = dims_x;
        if (dx != 0) {
            const int boundary_x = (dx > 0) ? 0 : (dims_x - 1);
            if (boundary_x == 0) {
                x_begin = 1;
            } else {
                x_end = dims_x - 1;
            }
        }
        for (int z = 0; z < dims_z; z++) {
            for (int x = x_begin; x < x_end; x++) {
                if (process_line_3d(discretized, mask, dims_x, dims_y, dims_z, roi,
                                    x, start_y, z, dx, dy, dz, max_run_length, matrix) > 1) {
                    multi_element = 1;
                }
            }
        }
    }

    if (dz != 0) {
        const int start_z = (dz > 0) ? 0 : (dims_z - 1);
        int x_begin = 0;
        int x_end = dims_x;
        int y_begin = 0;
        int y_end = dims_y;

        if (dx != 0) {
            const int boundary_x = (dx > 0) ? 0 : (dims_x - 1);
            if (boundary_x == 0) {
                x_begin = 1;
            } else {
                x_end = dims_x - 1;
            }
        }
        if (dy != 0) {
            const int boundary_y = (dy > 0) ? 0 : (dims_y - 1);
            if (boundary_y == 0) {
                y_begin = 1;
            } else {
                y_end = dims_y - 1;
            }
        }

        for (int y = y_begin; y < y_end; y++) {
            for (int x = x_begin; x < x_end; x++) {
                if (process_line_3d(discretized, mask, dims_x, dims_y, dims_z, roi,
                                    x, y, start_z, dx, dy, dz, max_run_length, matrix) > 1) {
                    multi_element = 1;
                }
            }
        }
    }

    if (!multi_element) {
        memset(matrix, 0, (size_t)num_bins * (size_t)max_run_length * sizeof(double));
    }
}

// Task 7.1: Compute GLRLM matrices with parallel processing
static GLRLMMatrices* compute_glrlm_matrices(FlashRadiomicsImage* img, FlashRadiomicsMask* mask,
                                             int num_bins, int* discretized) {
    GLRLMMatrices* glrlm = (GLRLMMatrices*)calloc(1, sizeof(GLRLMMatrices));
    if (!glrlm) return NULL;
    
    glrlm->num_bins = num_bins;
    glrlm->num_directions = (img->ndim == 2) ? 4 : 13;
    
    // Maximum run length is the maximum dimension
    glrlm->max_run_length = img->dims[0];
    if (img->dims[1] > glrlm->max_run_length) glrlm->max_run_length = img->dims[1];
    if (img->dims[2] > glrlm->max_run_length) glrlm->max_run_length = img->dims[2];
    
    glrlm->min_val = 0.0;
    glrlm->max_val = 0.0;

    const size_t dir_stride = (size_t)num_bins * (size_t)glrlm->max_run_length;
    const size_t total_matrix_size = (size_t)glrlm->num_directions * dir_stride;

    glrlm->matrix_data = (double*)calloc(total_matrix_size, sizeof(double));
    glrlm->matrices = (double***)malloc((size_t)glrlm->num_directions * sizeof(double**));
    if (!glrlm->matrix_data || !glrlm->matrices) {
        free(glrlm->matrix_data);
        free(glrlm->matrices);
        free(glrlm);
        return NULL;
    }
    for (int dir = 0; dir < glrlm->num_directions; dir++) {
        glrlm->matrices[dir] = (double**)malloc((size_t)num_bins * sizeof(double*));
        if (!glrlm->matrices[dir]) {
            for (int prev = 0; prev < dir; prev++) {
                free(glrlm->matrices[prev]);
            }
            free(glrlm->matrices);
            free(glrlm->matrix_data);
            free(glrlm);
            return NULL;
        }
        double* dir_base = glrlm->matrix_data + ((size_t)dir * dir_stride);
        for (int i = 0; i < num_bins; i++) {
            glrlm->matrices[dir][i] = dir_base + ((size_t)i * (size_t)glrlm->max_run_length);
        }
    }
    
    int dims_x = img->dims[0];
    int dims_y = img->dims[1];
    int dims_z = img->dims[2];
    ROIBounds roi = compute_roi_bounds(mask->data, dims_x, dims_y, dims_z);

    if (!roi.has_roi) {
        return glrlm;
    }
    
    // Compute GLRLM for each direction
    #pragma omp parallel for
    for (int dir_idx = 0; dir_idx < glrlm->num_directions; dir_idx++) {
        double* flat_matrix = glrlm->matrix_data + ((size_t)dir_idx * dir_stride);
        
        if (img->ndim == 2) {
            // 2D processing
            int dx = DIRECTIONS_2D[dir_idx][0];
            int dy = DIRECTIONS_2D[dir_idx][1];
            
            compute_runs_2d(discretized, mask->data, dims_x, dims_y, &roi,
                          dx, dy, num_bins, glrlm->max_run_length, flat_matrix);
        } else {
            // 3D processing
            int dx = DIRECTIONS_3D[dir_idx][0];
            int dy = DIRECTIONS_3D[dir_idx][1];
            int dz = DIRECTIONS_3D[dir_idx][2];
            
            compute_runs_3d(discretized, mask->data, dims_x, dims_y, dims_z, &roi,
                          dx, dy, dz, num_bins, glrlm->max_run_length, flat_matrix);
        }
    }
    
    return glrlm;
}

// Free GLRLM matrices
static void free_glrlm_matrices(GLRLMMatrices* glrlm) {
    if (!glrlm) return;
    
    for (int dir = 0; dir < glrlm->num_directions; dir++) {
        if (glrlm->matrices) {
            free(glrlm->matrices[dir]);
        }
    }
    free(glrlm->matrices);
    free(glrlm->matrix_data);
    free(glrlm);
}


// Helper: Compute coefficients for GLRLM features
typedef struct {
    double* pr;              // Marginal run length probabilities [Nr]
    double* pg;              // Marginal gray level probabilities [Ng]
    double Nr;               // Total number of runs
    int Ng;                  // Number of gray levels
    int Nr_len;              // Number of run lengths
} GLRLMCoefficients;

static GLRLMCoefficients* compute_glrlm_coefficients(double** P, int Ng, int Nr_len) {
    GLRLMCoefficients* coef = (GLRLMCoefficients*)malloc(sizeof(GLRLMCoefficients));
    if (!coef) return NULL;
    
    coef->pr = (double*)calloc(Nr_len, sizeof(double));
    coef->pg = (double*)calloc(Ng, sizeof(double));
    coef->Ng = Ng;
    coef->Nr_len = Nr_len;
    
    // Compute marginal probabilities and total runs
    coef->Nr = 0.0;
    for (int i = 0; i < Ng; i++) {
        for (int j = 0; j < Nr_len; j++) {
            double p = P[i][j];
            coef->pg[i] += p;
            coef->pr[j] += p;
            coef->Nr += p;
        }
    }
    
    return coef;
}

static void free_glrlm_coefficients(GLRLMCoefficients* coef) {
    if (!coef) return;
    free(coef->pr);
    free(coef->pg);
    free(coef);
}

// Task 7.2: Compute GLRLM emphasis features
static double compute_short_run_emphasis(GLRLMCoefficients* coef) {
    double sre = 0.0;
    for (int j = 0; j < coef->Nr_len; j++) {
        double j_val = (double)(j + 1);  // Run length starts from 1
        sre += coef->pr[j] / (j_val * j_val);
    }
    return (coef->Nr > 0) ? (sre / coef->Nr) : 0.0;
}

static double compute_long_run_emphasis(GLRLMCoefficients* coef) {
    double lre = 0.0;
    for (int j = 0; j < coef->Nr_len; j++) {
        double j_val = (double)(j + 1);  // Run length starts from 1
        lre += coef->pr[j] * j_val * j_val;
    }
    return (coef->Nr > 0) ? (lre / coef->Nr) : 0.0;
}

static double compute_low_gray_level_run_emphasis(GLRLMCoefficients* coef, double* ivector) {
    double lglre = 0.0;
    for (int i = 0; i < coef->Ng; i++) {
        double i_val = ivector[i];
        if (i_val > 0) {
            lglre += coef->pg[i] / (i_val * i_val);
        }
    }
    return (coef->Nr > 0) ? (lglre / coef->Nr) : 0.0;
}

static double compute_high_gray_level_run_emphasis(GLRLMCoefficients* coef, double* ivector) {
    double hglre = 0.0;
    for (int i = 0; i < coef->Ng; i++) {
        double i_val = ivector[i];
        hglre += coef->pg[i] * i_val * i_val;
    }
    return (coef->Nr > 0) ? (hglre / coef->Nr) : 0.0;
}

static double compute_short_run_low_gray_level_emphasis(double** P, int Ng, int Nr_len, 
                                                        double* ivector, double Nr) {
    double srlgle = 0.0;
    for (int i = 0; i < Ng; i++) {
        double i_val = ivector[i];
        if (i_val > 0) {
            for (int j = 0; j < Nr_len; j++) {
                double j_val = (double)(j + 1);
                srlgle += P[i][j] / (i_val * i_val * j_val * j_val);
            }
        }
    }
    return (Nr > 0) ? (srlgle / Nr) : 0.0;
}

static double compute_short_run_high_gray_level_emphasis(double** P, int Ng, int Nr_len,
                                                         double* ivector, double Nr) {
    double srhgle = 0.0;
    for (int i = 0; i < Ng; i++) {
        double i_val = ivector[i];
        for (int j = 0; j < Nr_len; j++) {
            double j_val = (double)(j + 1);
            srhgle += P[i][j] * i_val * i_val / (j_val * j_val);
        }
    }
    return (Nr > 0) ? (srhgle / Nr) : 0.0;
}

static double compute_long_run_low_gray_level_emphasis(double** P, int Ng, int Nr_len,
                                                       double* ivector, double Nr) {
    double lrlgle = 0.0;
    for (int i = 0; i < Ng; i++) {
        double i_val = ivector[i];
        if (i_val > 0) {
            for (int j = 0; j < Nr_len; j++) {
                double j_val = (double)(j + 1);
                lrlgle += P[i][j] * j_val * j_val / (i_val * i_val);
            }
        }
    }
    return (Nr > 0) ? (lrlgle / Nr) : 0.0;
}

static double compute_long_run_high_gray_level_emphasis(double** P, int Ng, int Nr_len,
                                                        double* ivector, double Nr) {
    double lrhgle = 0.0;
    for (int i = 0; i < Ng; i++) {
        double i_val = ivector[i];
        for (int j = 0; j < Nr_len; j++) {
            double j_val = (double)(j + 1);
            lrhgle += P[i][j] * i_val * i_val * j_val * j_val;
        }
    }
    return (Nr > 0) ? (lrhgle / Nr) : 0.0;
}

// Task 7.3: Compute GLRLM uniformity and variance features
static double compute_gray_level_non_uniformity(GLRLMCoefficients* coef) {
    double gln = 0.0;
    for (int i = 0; i < coef->Ng; i++) {
        gln += coef->pg[i] * coef->pg[i];
    }
    return (coef->Nr > 0) ? (gln / coef->Nr) : 0.0;
}

static double compute_gray_level_non_uniformity_normalized(GLRLMCoefficients* coef) {
    double glnn = 0.0;
    for (int i = 0; i < coef->Ng; i++) {
        glnn += coef->pg[i] * coef->pg[i];
    }
    return (coef->Nr > 0) ? (glnn / (coef->Nr * coef->Nr)) : 0.0;
}

static double compute_run_length_non_uniformity(GLRLMCoefficients* coef) {
    double rln = 0.0;
    for (int j = 0; j < coef->Nr_len; j++) {
        rln += coef->pr[j] * coef->pr[j];
    }
    return (coef->Nr > 0) ? (rln / coef->Nr) : 0.0;
}

static double compute_run_length_non_uniformity_normalized(GLRLMCoefficients* coef) {
    double rlnn = 0.0;
    for (int j = 0; j < coef->Nr_len; j++) {
        rlnn += coef->pr[j] * coef->pr[j];
    }
    return (coef->Nr > 0) ? (rlnn / (coef->Nr * coef->Nr)) : 0.0;
}

static double compute_run_percentage(GLRLMCoefficients* coef) {
    // Np = sum of (run_length * count)
    double Np = 0.0;
    for (int j = 0; j < coef->Nr_len; j++) {
        double j_val = (double)(j + 1);
        Np += coef->pr[j] * j_val;
    }
    return (Np > 0) ? (coef->Nr / Np) : 0.0;
}

static double compute_gray_level_variance(double** P, int Ng, int Nr_len, 
                                         double* ivector, double Nr) {
    if (Nr == 0) return 0.0;
    
    // Compute mean
    double mean = 0.0;
    for (int i = 0; i < Ng; i++) {
        for (int j = 0; j < Nr_len; j++) {
            mean += (P[i][j] / Nr) * ivector[i];
        }
    }
    
    // Compute variance
    double variance = 0.0;
    for (int i = 0; i < Ng; i++) {
        for (int j = 0; j < Nr_len; j++) {
            double diff = ivector[i] - mean;
            variance += (P[i][j] / Nr) * diff * diff;
        }
    }
    
    return variance;
}

static double compute_run_variance(double** P, int Ng, int Nr_len, double Nr) {
    if (Nr == 0) return 0.0;
    
    // Compute mean
    double mean = 0.0;
    for (int i = 0; i < Ng; i++) {
        for (int j = 0; j < Nr_len; j++) {
            double j_val = (double)(j + 1);
            mean += (P[i][j] / Nr) * j_val;
        }
    }
    
    // Compute variance
    double variance = 0.0;
    for (int i = 0; i < Ng; i++) {
        for (int j = 0; j < Nr_len; j++) {
            double j_val = (double)(j + 1);
            double diff = j_val - mean;
            variance += (P[i][j] / Nr) * diff * diff;
        }
    }
    
    return variance;
}

static double compute_run_entropy(double** P, int Ng, int Nr_len, double Nr) {
    if (Nr == 0) return 0.0;
    
    double entropy = 0.0;
    for (int i = 0; i < Ng; i++) {
        for (int j = 0; j < Nr_len; j++) {
            if (P[i][j] > 0) {
                double p = P[i][j] / Nr;
                entropy -= p * log2(p + EPSILON);
            }
        }
    }
    
    return entropy;
}

// Task 7.4: Main API function for GLRLM features
static FlashRadiomicsResult* flash_radiomics_glrlm_cpu_internal(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    double bin_width,
    int bin_count,
    const int* discretized_input,
    int ng_input
) {
    // Start timing
    clock_t start_time = clock();
    
    // Allocate result structure
    FlashRadiomicsResult* result = (FlashRadiomicsResult*)malloc(sizeof(FlashRadiomicsResult));
    if (!result) return NULL;
    
    // Initialize result
    result->features = NULL;
    result->count = 0;
    result->compute_time_ms = 0.0;
    result->error_code = FLASH_RADIOMICS_SUCCESS;
    result->error_msg[0] = '\0';
    
    // Validate inputs
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
    
    // Set number of threads for OpenMP
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
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to discretize image for GLRLM");
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        return result;
    }

    // Compute GLRLM matrices
    GLRLMMatrices* glrlm = compute_glrlm_matrices(img, mask, ng, (int*)discretized);
    if (!glrlm) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to compute GLRLM matrices");
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        return result;
    }
    
    // Number of features: 16 features
    result->count = 16;
    result->features = (FlashRadiomicsFeature*)malloc(result->count * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        free_glrlm_matrices(glrlm);
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to allocate feature array");
        return result;
    }
    
    // Create gray level vector (1-indexed in pyradiomics)
    double* ivector = (double*)malloc(glrlm->num_bins * sizeof(double));
    for (int i = 0; i < glrlm->num_bins; i++) {
        ivector[i] = (double)(i + 1);
    }
    
    // Compute features by averaging over all directions
    double sre = 0.0;
    double lre = 0.0;
    double gln = 0.0;
    double glnn = 0.0;
    double rln = 0.0;
    double rlnn = 0.0;
    double rp = 0.0;
    double glv = 0.0;
    double rv = 0.0;
    double re = 0.0;
    double lglre = 0.0;
    double hglre = 0.0;
    double srlgle = 0.0;
    double srhgle = 0.0;
    double lrlgle = 0.0;
    double lrhgle = 0.0;
    
    int valid_directions = 0;
    for (int dir = 0; dir < glrlm->num_directions; dir++) {
        double** P = glrlm->matrices[dir];
        
        // Compute coefficients for this direction
        GLRLMCoefficients* coef = compute_glrlm_coefficients(P, glrlm->num_bins, glrlm->max_run_length);
        if (coef->Nr <= 0.0) {
            free_glrlm_coefficients(coef);
            continue;
        }
        valid_directions++;
        
        // Accumulate features
        sre += compute_short_run_emphasis(coef);
        lre += compute_long_run_emphasis(coef);
        gln += compute_gray_level_non_uniformity(coef);
        glnn += compute_gray_level_non_uniformity_normalized(coef);
        rln += compute_run_length_non_uniformity(coef);
        rlnn += compute_run_length_non_uniformity_normalized(coef);
        rp += compute_run_percentage(coef);
        glv += compute_gray_level_variance(P, glrlm->num_bins, glrlm->max_run_length, ivector, coef->Nr);
        rv += compute_run_variance(P, glrlm->num_bins, glrlm->max_run_length, coef->Nr);
        re += compute_run_entropy(P, glrlm->num_bins, glrlm->max_run_length, coef->Nr);
        lglre += compute_low_gray_level_run_emphasis(coef, ivector);
        hglre += compute_high_gray_level_run_emphasis(coef, ivector);
        srlgle += compute_short_run_low_gray_level_emphasis(P, glrlm->num_bins, glrlm->max_run_length, ivector, coef->Nr);
        srhgle += compute_short_run_high_gray_level_emphasis(P, glrlm->num_bins, glrlm->max_run_length, ivector, coef->Nr);
        lrlgle += compute_long_run_low_gray_level_emphasis(P, glrlm->num_bins, glrlm->max_run_length, ivector, coef->Nr);
        lrhgle += compute_long_run_high_gray_level_emphasis(P, glrlm->num_bins, glrlm->max_run_length, ivector, coef->Nr);
        
        free_glrlm_coefficients(coef);
    }
    
    if (valid_directions > 0) {
        // Average over valid directions only. In voxel kernels many
        // directions can be empty and should not dilute the feature.
        sre /= valid_directions;
        lre /= valid_directions;
        gln /= valid_directions;
        glnn /= valid_directions;
        rln /= valid_directions;
        rlnn /= valid_directions;
        rp /= valid_directions;
        glv /= valid_directions;
        rv /= valid_directions;
        re /= valid_directions;
        lglre /= valid_directions;
        hglre /= valid_directions;
        srlgle /= valid_directions;
        srhgle /= valid_directions;
        lrlgle /= valid_directions;
        lrhgle /= valid_directions;
    } else {
        sre = NAN;
        lre = NAN;
        gln = NAN;
        glnn = NAN;
        rln = NAN;
        rlnn = NAN;
        rp = NAN;
        glv = NAN;
        rv = NAN;
        re = NAN;
        lglre = NAN;
        hglre = NAN;
        srlgle = NAN;
        srhgle = NAN;
        lrlgle = NAN;
        lrhgle = NAN;
    }
    
    // Populate features
    int idx = 0;
    
    snprintf(result->features[idx].name, 64, "glrlm_ShortRunEmphasis");
    result->features[idx++].value = sre;
    
    snprintf(result->features[idx].name, 64, "glrlm_LongRunEmphasis");
    result->features[idx++].value = lre;
    
    snprintf(result->features[idx].name, 64, "glrlm_GrayLevelNonUniformity");
    result->features[idx++].value = gln;
    
    snprintf(result->features[idx].name, 64, "glrlm_GrayLevelNonUniformityNormalized");
    result->features[idx++].value = glnn;
    
    snprintf(result->features[idx].name, 64, "glrlm_RunLengthNonUniformity");
    result->features[idx++].value = rln;
    
    snprintf(result->features[idx].name, 64, "glrlm_RunLengthNonUniformityNormalized");
    result->features[idx++].value = rlnn;
    
    snprintf(result->features[idx].name, 64, "glrlm_RunPercentage");
    result->features[idx++].value = rp;
    
    snprintf(result->features[idx].name, 64, "glrlm_GrayLevelVariance");
    result->features[idx++].value = glv;
    
    snprintf(result->features[idx].name, 64, "glrlm_RunVariance");
    result->features[idx++].value = rv;
    
    snprintf(result->features[idx].name, 64, "glrlm_RunEntropy");
    result->features[idx++].value = re;
    
    snprintf(result->features[idx].name, 64, "glrlm_LowGrayLevelRunEmphasis");
    result->features[idx++].value = lglre;
    
    snprintf(result->features[idx].name, 64, "glrlm_HighGrayLevelRunEmphasis");
    result->features[idx++].value = hglre;
    
    snprintf(result->features[idx].name, 64, "glrlm_ShortRunLowGrayLevelEmphasis");
    result->features[idx++].value = srlgle;
    
    snprintf(result->features[idx].name, 64, "glrlm_ShortRunHighGrayLevelEmphasis");
    result->features[idx++].value = srhgle;
    
    snprintf(result->features[idx].name, 64, "glrlm_LongRunLowGrayLevelEmphasis");
    result->features[idx++].value = lrlgle;
    
    snprintf(result->features[idx].name, 64, "glrlm_LongRunHighGrayLevelEmphasis");
    result->features[idx++].value = lrhgle;
    
    // Clean up
    free(ivector);
    free_glrlm_matrices(glrlm);
    if (discretized_owned) {
        flash_radiomics_free(discretized_owned);
    }
    
    // Calculate execution time
    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    return result;
}

FlashRadiomicsResult* flash_radiomics_glrlm_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    double bin_width,
    int bin_count
) {
    return flash_radiomics_glrlm_cpu_internal(
        img, mask, num_threads, bin_width, bin_count, NULL, 0
    );
}

FlashRadiomicsResult* flash_radiomics_glrlm_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    const int* discretized,
    int ng
) {
    return flash_radiomics_glrlm_cpu_internal(
        img, mask, num_threads, 0.0, 0, discretized, ng
    );
}

typedef struct {
    int ndim;
    int dims[3];
    int num_bins;
    int num_directions;
    int max_run_length;
    size_t dir_stride;
    size_t matrix_size;
    double* matrix_data;
    double*** matrices;
    double** row_ptrs;
    double* ivector;
    double* pr;
    double* pg;
} GLRLMVoxelWorkspace;

static int init_glrlm_voxel_workspace(
    GLRLMVoxelWorkspace* ws,
    int ndim,
    const int dims[3],
    int num_bins
) {
    memset(ws, 0, sizeof(*ws));
    ws->ndim = ndim;
    ws->num_bins = num_bins;
    for (int axis = 0; axis < 3; axis++) {
        ws->dims[axis] = (axis < ndim) ? dims[axis] : 1;
    }
    ws->num_directions = (ndim == 2) ? 4 : 13;

    ws->max_run_length = ws->dims[0];
    if (ws->dims[1] > ws->max_run_length) ws->max_run_length = ws->dims[1];
    if (ws->dims[2] > ws->max_run_length) ws->max_run_length = ws->dims[2];

    ws->dir_stride = (size_t)ws->num_bins * (size_t)ws->max_run_length;
    ws->matrix_size = (size_t)ws->num_directions * ws->dir_stride;

    ws->matrix_data = (double*)calloc(ws->matrix_size, sizeof(double));
    ws->matrices = (double***)malloc((size_t)ws->num_directions * sizeof(double**));
    ws->row_ptrs = (double**)malloc(
        (size_t)ws->num_directions * (size_t)ws->num_bins * sizeof(double*)
    );
    ws->ivector = (double*)malloc((size_t)ws->num_bins * sizeof(double));
    ws->pr = (double*)malloc((size_t)ws->max_run_length * sizeof(double));
    ws->pg = (double*)malloc((size_t)ws->num_bins * sizeof(double));
    if (!ws->matrix_data || !ws->matrices || !ws->row_ptrs ||
        !ws->ivector || !ws->pr || !ws->pg) {
        return 0;
    }

    for (int dir = 0; dir < ws->num_directions; dir++) {
        ws->matrices[dir] = ws->row_ptrs + ((size_t)dir * (size_t)ws->num_bins);
        double* dir_base = ws->matrix_data + ((size_t)dir * ws->dir_stride);
        for (int i = 0; i < ws->num_bins; i++) {
            ws->matrices[dir][i] = dir_base + ((size_t)i * (size_t)ws->max_run_length);
        }
    }
    for (int i = 0; i < ws->num_bins; i++) {
        ws->ivector[i] = (double)(i + 1);
    }

    return 1;
}

static void free_glrlm_voxel_workspace(GLRLMVoxelWorkspace* ws) {
    if (!ws) return;
    free(ws->pg);
    free(ws->pr);
    free(ws->ivector);
    free(ws->row_ptrs);
    free(ws->matrices);
    free(ws->matrix_data);
    memset(ws, 0, sizeof(*ws));
}

static void compute_glrlm_matrices_voxel_workspace(
    GLRLMVoxelWorkspace* ws,
    const int* discretized,
    const uint8_t* mask
) {
    memset(ws->matrix_data, 0, ws->matrix_size * sizeof(double));

    ROIBounds roi = compute_roi_bounds(mask, ws->dims[0], ws->dims[1], ws->dims[2]);
    if (!roi.has_roi) {
        return;
    }

    for (int dir_idx = 0; dir_idx < ws->num_directions; dir_idx++) {
        double* flat_matrix = ws->matrix_data + ((size_t)dir_idx * ws->dir_stride);
        if (ws->ndim == 2) {
            int dx = DIRECTIONS_2D[dir_idx][0];
            int dy = DIRECTIONS_2D[dir_idx][1];
            compute_runs_2d(
                (int*)discretized,
                (uint8_t*)mask,
                ws->dims[0],
                ws->dims[1],
                &roi,
                dx,
                dy,
                ws->num_bins,
                ws->max_run_length,
                flat_matrix
            );
        } else {
            int dx = DIRECTIONS_3D[dir_idx][0];
            int dy = DIRECTIONS_3D[dir_idx][1];
            int dz = DIRECTIONS_3D[dir_idx][2];
            compute_runs_3d(
                (int*)discretized,
                (uint8_t*)mask,
                ws->dims[0],
                ws->dims[1],
                ws->dims[2],
                &roi,
                dx,
                dy,
                dz,
                ws->num_bins,
                ws->max_run_length,
                flat_matrix
            );
        }
    }
}

static int is_glrlm_feature_enabled(const uint8_t* include_features, int idx) {
    return (!include_features) || (include_features[idx] != 0);
}

static void compute_glrlm_features_voxel_workspace(
    GLRLMVoxelWorkspace* ws,
    const uint8_t* include_features,
    double out_features[FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT]
) {
    for (int i = 0; i < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT; i++) {
        out_features[i] = NAN;
    }

    const int need_sre = is_glrlm_feature_enabled(include_features, 0);
    const int need_lre = is_glrlm_feature_enabled(include_features, 1);
    const int need_gln = is_glrlm_feature_enabled(include_features, 2);
    const int need_glnn = is_glrlm_feature_enabled(include_features, 3);
    const int need_rln = is_glrlm_feature_enabled(include_features, 4);
    const int need_rlnn = is_glrlm_feature_enabled(include_features, 5);
    const int need_rp = is_glrlm_feature_enabled(include_features, 6);
    const int need_glv = is_glrlm_feature_enabled(include_features, 7);
    const int need_rv = is_glrlm_feature_enabled(include_features, 8);
    const int need_re = is_glrlm_feature_enabled(include_features, 9);
    const int need_lglre = is_glrlm_feature_enabled(include_features, 10);
    const int need_hglre = is_glrlm_feature_enabled(include_features, 11);
    const int need_srlgle = is_glrlm_feature_enabled(include_features, 12);
    const int need_srhgle = is_glrlm_feature_enabled(include_features, 13);
    const int need_lrlgle = is_glrlm_feature_enabled(include_features, 14);
    const int need_lrhgle = is_glrlm_feature_enabled(include_features, 15);

    double sums[FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT];
    for (int i = 0; i < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT; i++) {
        sums[i] = 0.0;
    }

    int valid_directions = 0;
    for (int dir = 0; dir < ws->num_directions; dir++) {
        memset(ws->pg, 0, (size_t)ws->num_bins * sizeof(double));
        memset(ws->pr, 0, (size_t)ws->max_run_length * sizeof(double));

        double Nr = 0.0;
        double** P = ws->matrices[dir];
        for (int i = 0; i < ws->num_bins; i++) {
            for (int j = 0; j < ws->max_run_length; j++) {
                double p = P[i][j];
                ws->pg[i] += p;
                ws->pr[j] += p;
                Nr += p;
            }
        }
        if (Nr <= 0.0) {
            continue;
        }
        valid_directions++;

        if (need_sre || need_lre || need_rln || need_rlnn || need_rp || need_rv) {
            double sre_sum = 0.0;
            double lre_sum = 0.0;
            double rln_sum = 0.0;
            double Np = 0.0;
            double mean_j = 0.0;
            double rv_sum = 0.0;

            for (int j = 0; j < ws->max_run_length; j++) {
                double j_val = (double)(j + 1);
                double pr = ws->pr[j];
                double j_sq = j_val * j_val;
                if (need_sre) sre_sum += pr / j_sq;
                if (need_lre) lre_sum += pr * j_sq;
                if (need_rln || need_rlnn) rln_sum += pr * pr;
                if (need_rp) Np += pr * j_val;
                if (need_rv) mean_j += pr * j_val;
            }
            if (need_sre) sums[0] += sre_sum / Nr;
            if (need_lre) sums[1] += lre_sum / Nr;
            if (need_rln) sums[4] += rln_sum / Nr;
            if (need_rlnn) sums[5] += rln_sum / (Nr * Nr);
            if (need_rp) sums[6] += (Np > 0.0) ? (Nr / Np) : 0.0;
            if (need_rv) {
                mean_j /= Nr;
                for (int j = 0; j < ws->max_run_length; j++) {
                    double j_val = (double)(j + 1);
                    double diff = j_val - mean_j;
                    rv_sum += ws->pr[j] * diff * diff;
                }
                sums[8] += rv_sum / Nr;
            }
        }

        if (need_gln || need_glnn || need_lglre || need_hglre || need_glv) {
            double gln_sum = 0.0;
            double lglre_sum = 0.0;
            double hglre_sum = 0.0;
            double mean_i = 0.0;
            double glv_sum = 0.0;

            for (int i = 0; i < ws->num_bins; i++) {
                double i_val = ws->ivector[i];
                double pg = ws->pg[i];
                double i_sq = i_val * i_val;
                if (need_gln || need_glnn) gln_sum += pg * pg;
                if (need_lglre) lglre_sum += pg / i_sq;
                if (need_hglre) hglre_sum += pg * i_sq;
                if (need_glv) mean_i += pg * i_val;
            }
            if (need_gln) sums[2] += gln_sum / Nr;
            if (need_glnn) sums[3] += gln_sum / (Nr * Nr);
            if (need_lglre) sums[10] += lglre_sum / Nr;
            if (need_hglre) sums[11] += hglre_sum / Nr;
            if (need_glv) {
                mean_i /= Nr;
                for (int i = 0; i < ws->num_bins; i++) {
                    double diff = ws->ivector[i] - mean_i;
                    glv_sum += ws->pg[i] * diff * diff;
                }
                sums[7] += glv_sum / Nr;
            }
        }

        if (need_re || need_srlgle || need_srhgle || need_lrlgle || need_lrhgle) {
            double re_sum = 0.0;
            double srlgle_sum = 0.0;
            double srhgle_sum = 0.0;
            double lrlgle_sum = 0.0;
            double lrhgle_sum = 0.0;

            for (int i = 0; i < ws->num_bins; i++) {
                double i_val = ws->ivector[i];
                double i_sq = i_val * i_val;
                for (int j = 0; j < ws->max_run_length; j++) {
                    double p_raw = P[i][j];
                    if (p_raw <= 0.0) {
                        continue;
                    }
                    double j_val = (double)(j + 1);
                    double j_sq = j_val * j_val;
                    if (need_re) {
                        double p = p_raw / Nr;
                        re_sum -= p * log2(p + EPSILON);
                    }
                    if (need_srlgle) {
                        srlgle_sum += p_raw / (i_sq * j_sq);
                    }
                    if (need_srhgle) {
                        srhgle_sum += p_raw * i_sq / j_sq;
                    }
                    if (need_lrlgle) {
                        lrlgle_sum += p_raw * j_sq / i_sq;
                    }
                    if (need_lrhgle) {
                        lrhgle_sum += p_raw * i_sq * j_sq;
                    }
                }
            }
            if (need_re) sums[9] += re_sum;
            if (need_srlgle) sums[12] += srlgle_sum / Nr;
            if (need_srhgle) sums[13] += srhgle_sum / Nr;
            if (need_lrlgle) sums[14] += lrlgle_sum / Nr;
            if (need_lrhgle) sums[15] += lrhgle_sum / Nr;
        }
    }

    if (valid_directions <= 0) {
        return;
    }
    for (int idx = 0; idx < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT; idx++) {
        if (is_glrlm_feature_enabled(include_features, idx)) {
            out_features[idx] = sums[idx] / valid_directions;
        }
    }
}

int flash_radiomics_glrlm_cpu_discretized_voxel_batch(
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
    if (out_feature_stride < FLASH_RADIOMICS_GLRLM_VOXEL_FEATURE_COUNT) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    if (!flash_radiomics_validate_dims_with_ndim(ndim, window_dims)) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }

    GLRLMVoxelWorkspace workspace;
    if (!init_glrlm_voxel_workspace(&workspace, ndim, window_dims, ng)) {
        free_glrlm_voxel_workspace(&workspace);
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

        compute_glrlm_matrices_voxel_workspace(&workspace, batch_discretized, batch_mask);
        compute_glrlm_features_voxel_workspace(
            &workspace,
            include_features,
            batch_output
        );
    }

    free_glrlm_voxel_workspace(&workspace);
    return FLASH_RADIOMICS_SUCCESS;
}
