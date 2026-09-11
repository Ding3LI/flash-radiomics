#ifndef FLASH_RADIOMICS_TYPES_H
#define FLASH_RADIOMICS_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Backend types
typedef enum {
    FLASH_RADIOMICS_CPU,
    FLASH_RADIOMICS_MPS,
    FLASH_RADIOMICS_CUDA,
    FLASH_RADIOMICS_AUTO
} FlashRadiomicsBackend;

// Image format types
typedef enum {
    FLASH_RADIOMICS_FORMAT_NIFTI,
    FLASH_RADIOMICS_FORMAT_DICOM_SERIES,
    FLASH_RADIOMICS_FORMAT_AUTO
} FlashRadiomicsImageFormat;

// Error codes
typedef enum {
    FLASH_RADIOMICS_SUCCESS = 0,
    FLASH_RADIOMICS_ERROR_FILE_NOT_FOUND = 1,
    FLASH_RADIOMICS_ERROR_INVALID_FORMAT = 2,
    FLASH_RADIOMICS_ERROR_DIMENSION_MISMATCH = 3,
    FLASH_RADIOMICS_ERROR_MEMORY_ALLOCATION = 4,
    FLASH_RADIOMICS_ERROR_INVALID_PARAMETER = 5,
    FLASH_RADIOMICS_ERROR_COMPUTATION_FAILED = 6,
    FLASH_RADIOMICS_ERROR_BACKEND_UNAVAILABLE = 7,
    FLASH_RADIOMICS_ERROR_IO_FAILED = 8,
    FLASH_RADIOMICS_ERROR_CORRUPTED_DATA = 9,
    FLASH_RADIOMICS_ERROR_UNSUPPORTED_OPERATION = 10
} FlashRadiomicsErrorCode;

// Backend availability bitmask
#define FLASH_RADIOMICS_BACKEND_CPU_AVAILABLE (1 << 0)
#define FLASH_RADIOMICS_BACKEND_MPS_AVAILABLE (1 << 1)
#define FLASH_RADIOMICS_BACKEND_CUDA_AVAILABLE (1 << 2)

// Image data structure
typedef struct {
    float* data;              // Flattened array: [y][x] for 2D, [z][y][x] for 3D
    int ndim;                 // Number of dimensions (2 or 3)
    int dims[3];              // [x, y, z] dimensions (z=1 for 2D)
    float spacing[3];         // [x, y, z] voxel/pixel spacing in mm
    float origin[3];          // [x, y, z] origin coordinates
    int error_code;           // 0 = success, see FlashRadiomicsErrorCode
    char error_msg[256];      // Error description
} FlashRadiomicsImage;

// Mask data structure
typedef struct {
    uint8_t* data;            // Boolean mask (0 or 1)
    int ndim;                 // Number of dimensions (2 or 3)
    int dims[3];              // Must match image dims [x, y, z] (z=1 for 2D)
    int label;                // ROI label value
    int error_code;           // 0 = success, see FlashRadiomicsErrorCode
    char error_msg[256];      // Error description
} FlashRadiomicsMask;

// Feature result structure
typedef struct {
    char name[64];            // Feature name (e.g., "firstorder_Mean")
    double value;             // Computed feature value
} FlashRadiomicsFeature;

// Result structure
typedef struct {
    FlashRadiomicsFeature* features;  // Array of features
    int count;                    // Number of features
    double compute_time_ms;       // Execution time in milliseconds
    int error_code;               // 0 = success, see FlashRadiomicsErrorCode
    char error_msg[256];          // Error description
} FlashRadiomicsResult;

// Forward declaration for CUDA context
struct CudaContext;

// Execution context for hardware abstraction
typedef struct {
    FlashRadiomicsBackend backend;    // Selected backend (CPU, MPS, CUDA, AUTO)
    int num_threads;              // Number of threads for CPU backend
    void* device_handle;          // Device handle for MPS backend (MTLDevice*)
    struct CudaContext* cuda_ctx; // CUDA context for GPU backend
    int available_backends;       // Bitmask of available backends
    int initialized;              // 1 if context is initialized, 0 otherwise
} FlashRadiomicsContext;

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_TYPES_H
