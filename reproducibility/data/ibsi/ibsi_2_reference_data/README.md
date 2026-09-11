The Image Biomarker Standardisation Initiative (IBSI) standardized the use of eight types of convolutional filters for use in radiomics and quantitative image analysis in general (**publication forthcoming**). These are mean filters, Laplacian-of-Gaussian filters, Laws kernels, Gabor kernels, separable wavelets, decomposed separable wavelets, non-separable wavelets and decomposed non-separable wavelets.

This repository contains reference response maps and reference feature values for standardized convolutional filters. These can be used to verify whether your software is compliant with the IBSI standard. Each of these reference objects are computed using data and configuration parameters defined in (**publication forthcoming**) and the IBSI reference manual (https://doi.org/10.48550/arXiv.2006.05470).

The reference response maps are available in NIfTI format. A filter implementation should produce a response map where all voxel intensities differ by at most a tolerance margin from the reference response map. This tolerance margin is 1% of the range of intensities in the reference map.

The reference feature values are in a comma-separated value file, which lists both the reference value and its tolerance margin. A filter implementation should produce a feature value that lies within the tolerance margin around the reference value.


