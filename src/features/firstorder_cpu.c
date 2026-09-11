#include "firstorder_cpu.h"
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

// Helper structure for histogram computation
typedef struct {
    double* bins;           // Histogram bins (probabilities)
    int num_bins;           // Number of bins
    int num_voxels;         // Number of voxels in ROI
} Histogram;

// Helper structure for statistical calculations
typedef struct {
    double mean;
    double variance;
    double std_dev;
    double skewness;
    double kurtosis;
    double min;
    double max;
    double range;
    double median;
    double mode;
    double p10;             // 10th percentile
    double p25;             // 25th percentile
    double p75;             // 75th percentile
    double p90;             // 90th percentile
    double iqr;             // Interquartile range
    double energy;
    double total_energy;
    double entropy;
    double rms;             // Root mean squared
    double uniformity;
    double mad;             // Mean absolute deviation
    double rmad;            // Robust mean absolute deviation
    double medad;           // Median absolute deviation
    double cov;             // Coefficient of variation
    double qcod;            // Quartile coefficient of dispersion
} FirstOrderStats;

// Comparison function for qsort
static int compare_doubles(const void* a, const void* b) {
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);
}

static double linear_percentile_from_sorted(const double* values, int count, double percentile) {
    if (!values || count <= 0) {
        return NAN;
    }
    if (count == 1) {
        return values[0];
    }

    double bounded_percentile = percentile;
    if (bounded_percentile < 0.0) {
        bounded_percentile = 0.0;
    } else if (bounded_percentile > 100.0) {
        bounded_percentile = 100.0;
    }

    double rank = ((double)count - 1.0) * (bounded_percentile / 100.0);
    int lower_index = (int)floor(rank);
    int upper_index = (int)ceil(rank);
    if (lower_index == upper_index) {
        return values[lower_index];
    }

    double interpolation = rank - (double)lower_index;
    double lower = values[lower_index];
    double upper = values[upper_index];
    return lower + (upper - lower) * interpolation;
}

// Task 5.1: Compute histogram with parallel processing
static Histogram* compute_histogram(const int* discretized, FlashRadiomicsMask* mask, int num_bins, size_t total_voxels) {
    Histogram* hist = (Histogram*)malloc(sizeof(Histogram));
    if (!hist) return NULL;
    
    hist->num_bins = num_bins;
    hist->bins = (double*)calloc(num_bins, sizeof(double));
    if (!hist->bins) {
        free(hist);
        return NULL;
    }

    // Count voxels in ROI
    int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (size_t i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            count++;
        }
    }
    hist->num_voxels = count;
    
    if (hist->num_voxels == 0) {
        free(hist->bins);
        free(hist);
        return NULL;
    }

    // Build histogram with parallel processing
    #pragma omp parallel
    {
        double* local_bins = (double*)calloc(num_bins, sizeof(double));
        
        #pragma omp for nowait
        for (size_t i = 0; i < total_voxels; i++) {
            int bin = discretized[i];
            if (bin > 0) {
                int bin_idx = bin - 1;
                if (bin_idx >= 0 && bin_idx < num_bins) {
                    local_bins[bin_idx]++;
                }
            }
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < num_bins; i++) {
                hist->bins[i] += local_bins[i];
            }
        }
        
        free(local_bins);
    }
    
    // Normalize histogram to get probabilities (p_i)
    for (int i = 0; i < num_bins; i++) {
        hist->bins[i] /= hist->num_voxels;
    }
    
    return hist;
}

// Task 5.2: Calculate statistical moments
static void compute_statistical_moments(FlashRadiomicsImage* img, FlashRadiomicsMask* mask, 
                                        FirstOrderStats* stats, double voxel_array_shift) {
    int total_voxels = img->dims[0] * img->dims[1] * img->dims[2];
    
    // Collect ROI values for percentile calculations
    double* roi_values = (double*)malloc(total_voxels * sizeof(double));
    int roi_count = 0;
    
    // First pass: mean, min, max
    double sum = 0.0;
    stats->min = DBL_MAX;
    stats->max = -DBL_MAX;
    
    #pragma omp parallel
    {
        double local_sum = 0.0;
        double local_min = DBL_MAX;
        double local_max = -DBL_MAX;
        
        #pragma omp for nowait
        for (int i = 0; i < total_voxels; i++) {
            if (mask->data[i] != 0) {
                float val = img->data[i];
                local_sum += val;
                if (val < local_min) local_min = val;
                if (val > local_max) local_max = val;
            }
        }
        
        #pragma omp critical
        {
            sum += local_sum;
            if (local_min < stats->min) stats->min = local_min;
            if (local_max > stats->max) stats->max = local_max;
        }
    }
    
    // Collect values for median and percentiles
    for (int i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            roi_values[roi_count++] = img->data[i];
        }
    }
    
    stats->mean = sum / roi_count;
    stats->range = stats->max - stats->min;
    
    // Second pass: variance, skewness, kurtosis
    double sum_sq_diff = 0.0;
    double sum_cube_diff = 0.0;
    double sum_quad_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sum_squares = 0.0;
    double sum_shifted_squares = 0.0;
    
    #pragma omp parallel for reduction(+:sum_sq_diff,sum_cube_diff,sum_quad_diff,sum_abs_diff,sum_squares,sum_shifted_squares)
    for (int i = 0; i < total_voxels; i++) {
        if (mask->data[i] != 0) {
            double val = img->data[i];
            double diff = val - stats->mean;
            double sq_diff = diff * diff;
            sum_sq_diff += sq_diff;
            sum_cube_diff += sq_diff * diff;
            sum_quad_diff += sq_diff * sq_diff;
            sum_abs_diff += fabs(diff);
            sum_squares += val * val;
            double shifted = val + voxel_array_shift;
            sum_shifted_squares += shifted * shifted;
        }
    }
    
    stats->variance = sum_sq_diff / roi_count;
    stats->std_dev = sqrt(stats->variance);
    
    if (stats->variance > 0) {
        stats->skewness = (sum_cube_diff / roi_count) / pow(stats->std_dev, 3);
        // IBSI defines kurtosis as excess kurtosis (Pearson kurtosis minus 3).
        stats->kurtosis = (sum_quad_diff / roi_count) / (stats->variance * stats->variance) - 3.0;
    } else {
        stats->skewness = 0.0;
        stats->kurtosis = 0.0;
    }
    
    // Task 5.5: Mean absolute deviation
    stats->mad = sum_abs_diff / roi_count;
    
    // Task 5.3: Root mean squared (with voxelArrayShift)
    stats->rms = sqrt(sum_shifted_squares / roi_count);
    
    // Task 5.4: Median and percentiles (requires sorting)
    qsort(roi_values, roi_count, sizeof(double), compare_doubles);
    
    stats->median = linear_percentile_from_sorted(roi_values, roi_count, 50.0);
    stats->p10 = linear_percentile_from_sorted(roi_values, roi_count, 10.0);
    stats->p25 = linear_percentile_from_sorted(roi_values, roi_count, 25.0);
    stats->p75 = linear_percentile_from_sorted(roi_values, roi_count, 75.0);
    stats->p90 = linear_percentile_from_sorted(roi_values, roi_count, 90.0);
    stats->iqr = stats->p75 - stats->p25;

    double sum_abs_diff_median = 0.0;
    double robust_sum = 0.0;
    int robust_count = 0;
    #pragma omp parallel for reduction(+:sum_abs_diff_median)
    for (int i = 0; i < roi_count; i++) {
        sum_abs_diff_median += fabs(roi_values[i] - stats->median);
    }
    stats->medad = sum_abs_diff_median / roi_count;

    for (int i = 0; i < roi_count; i++) {
        if (roi_values[i] >= stats->p10 && roi_values[i] <= stats->p90) {
            robust_sum += roi_values[i];
            robust_count++;
        }
    }
    if (robust_count > 0) {
        double robust_mean = robust_sum / (double)robust_count;
        double robust_abs_sum = 0.0;
        for (int i = 0; i < roi_count; i++) {
            if (roi_values[i] >= stats->p10 && roi_values[i] <= stats->p90) {
                robust_abs_sum += fabs(roi_values[i] - robust_mean);
            }
        }
        stats->rmad = robust_abs_sum / (double)robust_count;
    } else {
        stats->rmad = 0.0;
    }

    if (stats->std_dev == 0.0) {
        stats->cov = 0.0;
    } else {
        stats->cov = stats->std_dev / stats->mean;
    }

    double qcod_denom = stats->p75 + stats->p25;
    if (qcod_denom == 0.0) {
        stats->qcod = 1.0;
    } else {
        stats->qcod = stats->iqr / qcod_denom;
    }
    
    // Mode (most frequent value - approximate using histogram)
    // For continuous data, we use the bin center with highest frequency
    stats->mode = stats->mean; // Default to mean if no clear mode

    // Energy uses voxelArrayShift (sum of squared shifted intensities)
    stats->energy = sum_shifted_squares;
    
    free(roi_values);
}

// Task 5.3: Calculate energy and entropy features
static void compute_entropy_uniformity(Histogram* hist, FirstOrderStats* stats) {
    stats->entropy = 0.0;
    stats->uniformity = 0.0;
    
    for (int i = 0; i < hist->num_bins; i++) {
        double p = hist->bins[i];
        if (p > 0) {
            stats->entropy -= p * log2(p);
            stats->uniformity += p * p;
        }
    }
}

// Task 5.6: Main API function for first-order features
static FlashRadiomicsResult* flash_radiomics_firstorder_cpu_internal(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    double bin_width,
    int bin_count,
    double voxel_array_shift,
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
    
    size_t total_voxels = flash_radiomics_get_num_elements(img->ndim, img->dims);
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
                "Failed to discretize image for histogram");
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        return result;
    }

    // Compute histogram on discretized values
    Histogram* hist = compute_histogram(discretized, mask, ng, total_voxels);
    if (!hist) {
        result->error_code = FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to compute histogram");
        if (discretized_owned) {
            flash_radiomics_free(discretized_owned);
        }
        return result;
    }
    
    // Compute statistical moments (Task 5.2)
    FirstOrderStats stats;
    memset(&stats, 0, sizeof(FirstOrderStats));
    compute_statistical_moments(img, mask, &stats, voxel_array_shift);
    
    // Compute entropy and uniformity (Task 5.3)
    compute_entropy_uniformity(hist, &stats);

    // Total energy scales by voxel volume
    double voxel_volume = img->spacing[0] * img->spacing[1] * (img->ndim == 3 ? img->spacing[2] : 1.0);
    stats.total_energy = stats.energy * voxel_volume;
    
    // Allocate feature array
    result->count = 22;
    result->features = (FlashRadiomicsFeature*)malloc(result->count * sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        free(hist->bins);
        free(hist);
        result->error_code = FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION;
        snprintf(result->error_msg, sizeof(result->error_msg),
                "Failed to allocate feature array");
        return result;
    }
    
    // Populate features
    int idx = 0;
    
    snprintf(result->features[idx].name, 64, "firstorder_Mean");
    result->features[idx++].value = stats.mean;
    
    snprintf(result->features[idx].name, 64, "firstorder_Variance");
    result->features[idx++].value = stats.variance;
    
    snprintf(result->features[idx].name, 64, "firstorder_StandardDeviation");
    result->features[idx++].value = stats.std_dev;
    
    snprintf(result->features[idx].name, 64, "firstorder_Skewness");
    result->features[idx++].value = stats.skewness;
    
    snprintf(result->features[idx].name, 64, "firstorder_Kurtosis");
    result->features[idx++].value = stats.kurtosis;
    
    snprintf(result->features[idx].name, 64, "firstorder_Minimum");
    result->features[idx++].value = stats.min;
    
    snprintf(result->features[idx].name, 64, "firstorder_Maximum");
    result->features[idx++].value = stats.max;
    
    snprintf(result->features[idx].name, 64, "firstorder_Range");
    result->features[idx++].value = stats.range;
    
    snprintf(result->features[idx].name, 64, "firstorder_Median");
    result->features[idx++].value = stats.median;
    
    snprintf(result->features[idx].name, 64, "firstorder_10Percentile");
    result->features[idx++].value = stats.p10;
    
    snprintf(result->features[idx].name, 64, "firstorder_90Percentile");
    result->features[idx++].value = stats.p90;
    
    snprintf(result->features[idx].name, 64, "firstorder_InterquartileRange");
    result->features[idx++].value = stats.iqr;
    
    snprintf(result->features[idx].name, 64, "firstorder_Energy");
    result->features[idx++].value = stats.energy;
    
    snprintf(result->features[idx].name, 64, "firstorder_TotalEnergy");
    result->features[idx++].value = stats.total_energy;
    
    snprintf(result->features[idx].name, 64, "firstorder_Entropy");
    result->features[idx++].value = stats.entropy;
    
    snprintf(result->features[idx].name, 64, "firstorder_RootMeanSquared");
    result->features[idx++].value = stats.rms;
    
    snprintf(result->features[idx].name, 64, "firstorder_Uniformity");
    result->features[idx++].value = stats.uniformity;
    
    snprintf(result->features[idx].name, 64, "firstorder_MeanAbsoluteDeviation");
    result->features[idx++].value = stats.mad;

    snprintf(result->features[idx].name, 64, "firstorder_RobustMeanAbsoluteDeviation");
    result->features[idx++].value = stats.rmad;

    snprintf(result->features[idx].name, 64, "firstorder_MedianAbsoluteDeviation");
    result->features[idx++].value = stats.medad;

    snprintf(result->features[idx].name, 64, "firstorder_CoefficientOfVariation");
    result->features[idx++].value = stats.cov;

    snprintf(result->features[idx].name, 64, "firstorder_QuartileCoefficientOfDispersion");
    result->features[idx++].value = stats.qcod;
    
    // Clean up
    if (discretized_owned) {
        flash_radiomics_free(discretized_owned);
    }
    free(hist->bins);
    free(hist);
    
    // Calculate execution time
    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    return result;
}

FlashRadiomicsResult* flash_radiomics_firstorder_cpu(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    double bin_width,
    int bin_count,
    double voxel_array_shift
) {
    return flash_radiomics_firstorder_cpu_internal(
        img,
        mask,
        num_threads,
        bin_width,
        bin_count,
        voxel_array_shift,
        NULL,
        0
    );
}

FlashRadiomicsResult* flash_radiomics_firstorder_cpu_discretized(
    FlashRadiomicsImage* img,
    FlashRadiomicsMask* mask,
    int num_threads,
    const int* discretized,
    int ng,
    double voxel_array_shift
) {
    return flash_radiomics_firstorder_cpu_internal(
        img,
        mask,
        num_threads,
        0.0,
        0,
        voxel_array_shift,
        discretized,
        ng
    );
}
