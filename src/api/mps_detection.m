/* MPS (Metal Performance Shaders) detection for Apple Silicon */

#include "mps_detection.h"
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_MAC
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif
#endif

static int flash_radiomics_env_truthy(const char* value) {
    if (!value || !*value) {
        return 0;
    }
    return (
        strcmp(value, "1") == 0 ||
        strcmp(value, "true") == 0 ||
        strcmp(value, "TRUE") == 0 ||
        strcmp(value, "yes") == 0 ||
        strcmp(value, "YES") == 0
    );
}

int flash_radiomics_detect_mps_available(void) {
#ifdef __APPLE__
#if TARGET_OS_MAC
    @autoreleasepool {
        if (flash_radiomics_env_truthy(getenv("FLASH_RADIOMICS_FORCE_MPS_AVAILABLE"))) {
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            return 1;
        }
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices && devices.count > 0) {
            return 1;
        }
    }
#endif
#endif
    return 0;
}

void* flash_radiomics_create_mps_device(void) {
#ifdef __APPLE__
#if TARGET_OS_MAC
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            return (void*)CFBridgingRetain(device);
        }
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices && devices.count > 0) {
            return (void*)CFBridgingRetain(devices[0]);
        }
    }
#endif
#endif
    return NULL;
}

void flash_radiomics_release_mps_device(void* device_handle) {
    if (device_handle) {
#ifdef __APPLE__
#if TARGET_OS_MAC
        CFBridgingRelease(device_handle);
#endif
#endif
    }
}
