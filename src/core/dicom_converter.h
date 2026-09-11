#ifndef FLASH_RADIOMICS_DICOM_CONVERTER_H
#define FLASH_RADIOMICS_DICOM_CONVERTER_H

#include "types.h"

// Convert DICOM file or series to NIfTI format
// Supports single DICOM files (2D) and DICOM series directories (3D)
// dicom_path: Path to single DICOM file or directory containing DICOM series
// output_nifti: Path to output NIfTI file
int flash_radiomics_dicom_to_nifti(const char* dicom_path, const char* output_nifti);

#endif // FLASH_RADIOMICS_DICOM_CONVERTER_H
