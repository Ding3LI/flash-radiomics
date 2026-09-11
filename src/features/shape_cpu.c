#include "shape_cpu.h"
#include "cshape.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef FLASH_RADIOMICS_PI
#define FLASH_RADIOMICS_PI 3.14159265358979323846
#endif

#define SHAPE_3D_FEATURE_COUNT 17
#define SHAPE_2D_FEATURE_COUNT 10

static FlashRadiomicsResult* allocate_result(int feature_count) {
    FlashRadiomicsResult* result =
        (FlashRadiomicsResult*)malloc(sizeof(FlashRadiomicsResult));
    if (!result) {
        return NULL;
    }

    result->features =
        (FlashRadiomicsFeature*)calloc((size_t)feature_count, sizeof(FlashRadiomicsFeature));
    if (!result->features) {
        free(result);
        return NULL;
    }

    result->count = feature_count;
    result->compute_time_ms = 0.0;
    result->error_code = FLASH_RADIOMICS_SUCCESS;
    result->error_msg[0] = '\0';
    return result;
}

static void set_result_error(FlashRadiomicsResult* result, int error_code, const char* msg) {
    if (!result) {
        return;
    }
    if (result->features) {
        free(result->features);
        result->features = NULL;
    }
    result->count = 0;
    result->compute_time_ms = 0.0;
    result->error_code = error_code;
    snprintf(
        result->error_msg,
        sizeof(result->error_msg),
        "%s",
        msg ? msg : "Unknown shape computation error"
    );
}

static int count_mask_elements(const FlashRadiomicsMask* mask) {
    int x = mask->dims[0];
    int y = mask->dims[1];
    int z = (mask->ndim == 3) ? mask->dims[2] : 1;
    int total = x * y * z;
    int count = 0;
    for (int i = 0; i < total; i++) {
        if (mask->data[i]) {
            count++;
        }
    }
    return count;
}

static uint8_t* pad_mask_3d(const FlashRadiomicsMask* mask, int* px, int* py, int* pz) {
    int nx = mask->dims[0];
    int ny = mask->dims[1];
    int nz = mask->dims[2];
    int nxp = nx + 2;
    int nyp = ny + 2;
    int nzp = nz + 2;

    size_t padded_size = (size_t)nxp * (size_t)nyp * (size_t)nzp;
    uint8_t* padded = (uint8_t*)calloc(padded_size, sizeof(uint8_t));
    if (!padded) {
        return NULL;
    }

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                size_t src_idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
                size_t dst_idx =
                    (size_t)(z + 1) * (size_t)nyp * (size_t)nxp +
                    (size_t)(y + 1) * (size_t)nxp +
                    (size_t)(x + 1);
                padded[dst_idx] = mask->data[src_idx];
            }
        }
    }

    *px = nxp;
    *py = nyp;
    *pz = nzp;
    return padded;
}

static uint8_t* pad_mask_2d(const FlashRadiomicsMask* mask, int* px, int* py) {
    int nx = mask->dims[0];
    int ny = mask->dims[1];
    int nxp = nx + 2;
    int nyp = ny + 2;

    size_t padded_size = (size_t)nxp * (size_t)nyp;
    uint8_t* padded = (uint8_t*)calloc(padded_size, sizeof(uint8_t));
    if (!padded) {
        return NULL;
    }

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            size_t src_idx = (size_t)y * (size_t)nx + (size_t)x;
            size_t dst_idx = (size_t)(y + 1) * (size_t)nxp + (size_t)(x + 1);
            padded[dst_idx] = mask->data[src_idx];
        }
    }

    *px = nxp;
    *py = nyp;
    return padded;
}

static void eigenvalues_symmetric_2x2(double a, double b, double c, double eigenvalues[2]) {
    double trace = a + c;
    double delta = sqrt(fmax(0.0, (a - c) * (a - c) + 4.0 * b * b));
    eigenvalues[0] = 0.5 * (trace - delta);
    eigenvalues[1] = 0.5 * (trace + delta);
}

static double determinant_symmetric_3x3(
    double m00,
    double m01,
    double m02,
    double m11,
    double m12,
    double m22
) {
    return m00 * (m11 * m22 - m12 * m12)
        - m01 * (m01 * m22 - m12 * m02)
        + m02 * (m01 * m12 - m11 * m02);
}

static void eigenvalues_symmetric_3x3(
    double m00,
    double m01,
    double m02,
    double m11,
    double m12,
    double m22,
    double eigenvalues[3]
) {
    double p1 = m01 * m01 + m02 * m02 + m12 * m12;
    if (p1 == 0.0) {
        eigenvalues[0] = m00;
        eigenvalues[1] = m11;
        eigenvalues[2] = m22;
    } else {
        double q = (m00 + m11 + m22) / 3.0;
        double a00 = m00 - q;
        double a11 = m11 - q;
        double a22 = m22 - q;
        double p2 = a00 * a00 + a11 * a11 + a22 * a22 + 2.0 * p1;
        double p = sqrt(fmax(p2 / 6.0, 0.0));

        if (p > 0.0) {
            double b00 = a00 / p;
            double b01 = m01 / p;
            double b02 = m02 / p;
            double b11 = a11 / p;
            double b12 = m12 / p;
            double b22 = a22 / p;
            double r =
                determinant_symmetric_3x3(b00, b01, b02, b11, b12, b22) / 2.0;

            double phi;
            if (r <= -1.0) {
                phi = FLASH_RADIOMICS_PI / 3.0;
            } else if (r >= 1.0) {
                phi = 0.0;
            } else {
                phi = acos(r) / 3.0;
            }

            eigenvalues[2] = q + 2.0 * p * cos(phi);
            eigenvalues[0] = q + 2.0 * p * cos(phi + 2.0 * FLASH_RADIOMICS_PI / 3.0);
            eigenvalues[1] = 3.0 * q - eigenvalues[0] - eigenvalues[2];
        } else {
            eigenvalues[0] = q;
            eigenvalues[1] = q;
            eigenvalues[2] = q;
        }
    }

    // Sort ascending.
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (eigenvalues[i] > eigenvalues[j]) {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = tmp;
            }
        }
    }
}

static void compute_covariance_eigen_3d(
    const FlashRadiomicsMask* mask,
    const double spacing_zyx[3],
    double eigenvalues[3]
) {
    int nx = mask->dims[0];
    int ny = mask->dims[1];
    int nz = mask->dims[2];

    double sum_z = 0.0;
    double sum_y = 0.0;
    double sum_x = 0.0;
    int count = 0;

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
                if (!mask->data[idx]) {
                    continue;
                }
                sum_z += (double)z * spacing_zyx[0];
                sum_y += (double)y * spacing_zyx[1];
                sum_x += (double)x * spacing_zyx[2];
                count++;
            }
        }
    }

    if (count <= 0) {
        eigenvalues[0] = 0.0;
        eigenvalues[1] = 0.0;
        eigenvalues[2] = 0.0;
        return;
    }

    double mean_z = sum_z / count;
    double mean_y = sum_y / count;
    double mean_x = sum_x / count;

    double c00 = 0.0;
    double c01 = 0.0;
    double c02 = 0.0;
    double c11 = 0.0;
    double c12 = 0.0;
    double c22 = 0.0;

    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
                if (!mask->data[idx]) {
                    continue;
                }

                double dz = (double)z * spacing_zyx[0] - mean_z;
                double dy = (double)y * spacing_zyx[1] - mean_y;
                double dx = (double)x * spacing_zyx[2] - mean_x;

                c00 += dz * dz;
                c01 += dz * dy;
                c02 += dz * dx;
                c11 += dy * dy;
                c12 += dy * dx;
                c22 += dx * dx;
            }
        }
    }

    c00 /= count;
    c01 /= count;
    c02 /= count;
    c11 /= count;
    c12 /= count;
    c22 /= count;

    eigenvalues_symmetric_3x3(c00, c01, c02, c11, c12, c22, eigenvalues);
}

static void compute_covariance_eigen_2d(
    const FlashRadiomicsMask* mask,
    const double spacing_yx[2],
    double eigenvalues[2]
) {
    int nx = mask->dims[0];
    int ny = mask->dims[1];

    double sum_y = 0.0;
    double sum_x = 0.0;
    int count = 0;

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            size_t idx = (size_t)y * (size_t)nx + (size_t)x;
            if (!mask->data[idx]) {
                continue;
            }
            sum_y += (double)y * spacing_yx[0];
            sum_x += (double)x * spacing_yx[1];
            count++;
        }
    }

    if (count <= 0) {
        eigenvalues[0] = 0.0;
        eigenvalues[1] = 0.0;
        return;
    }

    double mean_y = sum_y / count;
    double mean_x = sum_x / count;

    double c00 = 0.0;
    double c01 = 0.0;
    double c11 = 0.0;

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            size_t idx = (size_t)y * (size_t)nx + (size_t)x;
            if (!mask->data[idx]) {
                continue;
            }

            double dy = (double)y * spacing_yx[0] - mean_y;
            double dx = (double)x * spacing_yx[1] - mean_x;
            c00 += dy * dy;
            c01 += dy * dx;
            c11 += dx * dx;
        }
    }

    c00 /= count;
    c01 /= count;
    c11 /= count;

    eigenvalues_symmetric_2x2(c00, c01, c11, eigenvalues);
}

static double sanitize_eigenvalue(double value) {
    if (value < 0.0 && value > -1e-10) {
        return 0.0;
    }
    return (value > 0.0) ? value : 0.0;
}

FlashRadiomicsResult* flash_radiomics_shape_cpu(
    FlashRadiomicsMask* mask,
    float* spacing
) {
    clock_t start_time = clock();
    FlashRadiomicsResult* result = allocate_result(SHAPE_3D_FEATURE_COUNT);
    if (!result) {
        return NULL;
    }

    if (!mask || !mask->data || !spacing) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Shape requires non-null mask and spacing"
        );
        return result;
    }
    if (mask->ndim != 3) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Shape is only available for 3D masks"
        );
        return result;
    }
    if (mask->dims[0] <= 0 || mask->dims[1] <= 0 || mask->dims[2] <= 0) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Invalid 3D mask dimensions for Shape"
        );
        return result;
    }

    int voxel_count = count_mask_elements(mask);
    if (voxel_count <= 0) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Shape requires at least one ROI voxel"
        );
        return result;
    }

    int px = 0;
    int py = 0;
    int pz = 0;
    uint8_t* padded_mask = pad_mask_3d(mask, &px, &py, &pz);
    if (!padded_mask) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
            "Failed to allocate padded mask for Shape"
        );
        return result;
    }

    int size_zyx[3] = {pz, py, px};
    int strides_zyx[3] = {py * px, px, 1};
    double spacing_zyx[3] = {spacing[2], spacing[1], spacing[0]};

    double surface_area = 0.0;
    double mesh_volume = 0.0;
    double diameters[4] = {0.0, 0.0, 0.0, 0.0};
    int status = calculate_coefficients(
        (char*)padded_mask,
        size_zyx,
        strides_zyx,
        spacing_zyx,
        &surface_area,
        &mesh_volume,
        diameters
    );
    free(padded_mask);

    if (status != 0) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
            "cShape 3D coefficient computation failed"
        );
        return result;
    }

    mesh_volume = fabs(mesh_volume);
    surface_area = fabs(surface_area);

    double voxel_volume = (double)voxel_count * spacing[0] * spacing[1] * spacing[2];
    double surface_volume_ratio = (mesh_volume > 0.0) ? (surface_area / mesh_volume) : 0.0;
    double sphericity = (surface_area > 0.0)
        ? (pow(36.0 * FLASH_RADIOMICS_PI * mesh_volume * mesh_volume, 1.0 / 3.0) / surface_area)
        : 0.0;
    double compactness1 = (surface_area > 0.0)
        ? (mesh_volume / (pow(surface_area, 1.5) * sqrt(FLASH_RADIOMICS_PI)))
        : 0.0;
    double compactness2 = (surface_area > 0.0)
        ? ((36.0 * FLASH_RADIOMICS_PI * mesh_volume * mesh_volume) / pow(surface_area, 3.0))
        : 0.0;
    double spherical_disproportion = (sphericity > 0.0) ? (1.0 / sphericity) : 0.0;

    double eigenvalues[3] = {0.0, 0.0, 0.0};
    compute_covariance_eigen_3d(mask, spacing_zyx, eigenvalues);
    double lambda_least = sanitize_eigenvalue(eigenvalues[0]);
    double lambda_minor = sanitize_eigenvalue(eigenvalues[1]);
    double lambda_major = sanitize_eigenvalue(eigenvalues[2]);

    double major_axis = 4.0 * sqrt(lambda_major);
    double minor_axis = 4.0 * sqrt(lambda_minor);
    double least_axis = 4.0 * sqrt(lambda_least);
    double elongation = (lambda_major > 0.0) ? sqrt(lambda_minor / lambda_major) : 0.0;
    double flatness = (lambda_major > 0.0) ? sqrt(lambda_least / lambda_major) : 0.0;

    const char* names[SHAPE_3D_FEATURE_COUNT] = {
        "shape_MeshVolume",
        "shape_VoxelVolume",
        "shape_SurfaceArea",
        "shape_SurfaceVolumeRatio",
        "shape_Sphericity",
        "shape_Compactness1",
        "shape_Compactness2",
        "shape_SphericalDisproportion",
        "shape_Maximum3DDiameter",
        "shape_Maximum2DDiameterSlice",
        "shape_Maximum2DDiameterColumn",
        "shape_Maximum2DDiameterRow",
        "shape_MajorAxisLength",
        "shape_MinorAxisLength",
        "shape_LeastAxisLength",
        "shape_Elongation",
        "shape_Flatness",
    };
    double values[SHAPE_3D_FEATURE_COUNT] = {
        mesh_volume,
        voxel_volume,
        surface_area,
        surface_volume_ratio,
        sphericity,
        compactness1,
        compactness2,
        spherical_disproportion,
        diameters[3],
        diameters[0],
        diameters[1],
        diameters[2],
        major_axis,
        minor_axis,
        least_axis,
        elongation,
        flatness,
    };

    for (int i = 0; i < SHAPE_3D_FEATURE_COUNT; i++) {
        snprintf(result->features[i].name, sizeof(result->features[i].name), "%s", names[i]);
        result->features[i].value = values[i];
    }

    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    return result;
}

FlashRadiomicsResult* flash_radiomics_shape2d_cpu(
    FlashRadiomicsMask* mask,
    float* spacing
) {
    clock_t start_time = clock();
    FlashRadiomicsResult* result = allocate_result(SHAPE_2D_FEATURE_COUNT);
    if (!result) {
        return NULL;
    }

    if (!mask || !mask->data || !spacing) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Shape2D requires non-null mask and spacing"
        );
        return result;
    }
    if (mask->ndim != 2) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Shape2D is only available for 2D masks"
        );
        return result;
    }
    if (mask->dims[0] <= 0 || mask->dims[1] <= 0) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Invalid 2D mask dimensions for Shape2D"
        );
        return result;
    }

    int pixel_count = count_mask_elements(mask);
    if (pixel_count <= 0) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_INVALID_PARAMETER,
            "Shape2D requires at least one ROI pixel"
        );
        return result;
    }

    int px = 0;
    int py = 0;
    uint8_t* padded_mask = pad_mask_2d(mask, &px, &py);
    if (!padded_mask) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION,
            "Failed to allocate padded mask for Shape2D"
        );
        return result;
    }

    int size_yx[2] = {py, px};
    int strides_yx[2] = {px, 1};
    double spacing_yx[2] = {spacing[1], spacing[0]};

    double perimeter = 0.0;
    double mesh_surface = 0.0;
    double maximum_diameter = 0.0;
    int status = calculate_coefficients2D(
        (char*)padded_mask,
        size_yx,
        strides_yx,
        spacing_yx,
        &perimeter,
        &mesh_surface,
        &maximum_diameter
    );
    free(padded_mask);

    if (status != 0) {
        set_result_error(
            result,
            FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED,
            "cShape 2D coefficient computation failed"
        );
        return result;
    }

    mesh_surface = fabs(mesh_surface);
    double pixel_surface = (double)pixel_count * spacing[0] * spacing[1];
    double perimeter_surface_ratio = (mesh_surface > 0.0) ? (perimeter / mesh_surface) : 0.0;
    double sphericity = (perimeter > 0.0)
        ? (2.0 * sqrt(FLASH_RADIOMICS_PI * mesh_surface) / perimeter)
        : 0.0;
    double spherical_disproportion = (sphericity > 0.0) ? (1.0 / sphericity) : 0.0;

    double eigenvalues[2] = {0.0, 0.0};
    compute_covariance_eigen_2d(mask, spacing_yx, eigenvalues);
    double lambda_minor = sanitize_eigenvalue(eigenvalues[0]);
    double lambda_major = sanitize_eigenvalue(eigenvalues[1]);

    double major_axis = 4.0 * sqrt(lambda_major);
    double minor_axis = 4.0 * sqrt(lambda_minor);
    double elongation = (lambda_major > 0.0) ? sqrt(lambda_minor / lambda_major) : 0.0;

    const char* names[SHAPE_2D_FEATURE_COUNT] = {
        "shape2D_MeshSurface",
        "shape2D_PixelSurface",
        "shape2D_Perimeter",
        "shape2D_PerimeterSurfaceRatio",
        "shape2D_Sphericity",
        "shape2D_SphericalDisproportion",
        "shape2D_MaximumDiameter",
        "shape2D_MajorAxisLength",
        "shape2D_MinorAxisLength",
        "shape2D_Elongation",
    };
    double values[SHAPE_2D_FEATURE_COUNT] = {
        mesh_surface,
        pixel_surface,
        perimeter,
        perimeter_surface_ratio,
        sphericity,
        spherical_disproportion,
        maximum_diameter,
        major_axis,
        minor_axis,
        elongation,
    };

    for (int i = 0; i < SHAPE_2D_FEATURE_COUNT; i++) {
        snprintf(result->features[i].name, sizeof(result->features[i].name), "%s", names[i]);
        result->features[i].value = values[i];
    }

    clock_t end_time = clock();
    result->compute_time_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    return result;
}
