/* MPS (Metal Performance Shaders) detection for Apple Silicon */

#ifndef FLASH_RADIOMICS_MPS_DETECTION_H
#define FLASH_RADIOMICS_MPS_DETECTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __APPLE__
/**
 * Detect if MPS (Metal Performance Shaders) is available
 * Returns 1 if available, 0 otherwise
 */
int flash_radiomics_detect_mps_available(void);

/**
 * Create an MPS device handle
 * Returns device handle on success, NULL on failure
 * Caller must release with flash_radiomics_release_mps_device()
 */
void* flash_radiomics_create_mps_device(void);

/**
 * Release an MPS device handle
 */
void flash_radiomics_release_mps_device(void* device_handle);
#else
/* Non-macOS fallback: expose no-op stubs so shared libraries do not carry
 * unresolved MPS symbols on Linux/Windows builds. */
static inline int flash_radiomics_detect_mps_available(void) {
    return 0;
}

static inline void* flash_radiomics_create_mps_device(void) {
    return NULL;
}

static inline void flash_radiomics_release_mps_device(void* device_handle) {
    (void)device_handle;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // FLASH_RADIOMICS_MPS_DETECTION_H
