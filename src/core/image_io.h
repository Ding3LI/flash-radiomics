#ifndef FLASH_RADIOMICS_IMAGE_IO_H
#define FLASH_RADIOMICS_IMAGE_IO_H

#include "types.h"

// Load image from file or DICOM series directory
FlashRadiomicsImage* flash_radiomics_load_image(const char* path, FlashRadiomicsImageFormat format);

// Load segmentation mask from file or DICOM series directory
FlashRadiomicsMask* flash_radiomics_load_mask(const char* path, int label, FlashRadiomicsImageFormat format);

// Extract 2D slice from 3D volume
// axis: 0=sagittal (YZ plane), 1=coronal (XZ plane), 2=axial (XY plane)
FlashRadiomicsImage* flash_radiomics_extract_slice(FlashRadiomicsImage* img, int slice_index, int axis);

// Extract 2D mask slice from 3D mask
FlashRadiomicsMask* flash_radiomics_extract_mask_slice(FlashRadiomicsMask* mask, int slice_index, int axis);

// Validate image and mask compatibility
int flash_radiomics_validate_inputs(FlashRadiomicsImage* img, FlashRadiomicsMask* mask);

// Free memory
void flash_radiomics_free_image(FlashRadiomicsImage* img);
void flash_radiomics_free_mask(FlashRadiomicsMask* mask);

#endif // FLASH_RADIOMICS_IMAGE_IO_H
