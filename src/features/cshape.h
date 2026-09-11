#ifndef FLASH_RADIOMICS_CSHAPE_H
#define FLASH_RADIOMICS_CSHAPE_H

int calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters);
int calculate_coefficients2D(char *mask, int *size, int *strides, double *spacing,
                             double *perimeter, double *surface, double *diameter);

#endif // FLASH_RADIOMICS_CSHAPE_H
