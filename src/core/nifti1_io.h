/* nifti1_io.h - Basic NIfTI-1 I/O routines
 * Simplified version for flash_radiomics project
 * Based on the NIFTI-1 Data Format specification
 */

#ifndef NIFTI1_IO_H
#define NIFTI1_IO_H

#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NIfTI-1 header structure (348 bytes) */
typedef struct {
    int   sizeof_hdr;    /* MUST be 348 */
    char  data_type[10];
    char  db_name[18];
    int   extents;
    short session_error;
    char  regular;
    char  dim_info;
    
    short dim[8];        /* Data array dimensions */
    float intent_p1;
    float intent_p2;
    float intent_p3;
    short intent_code;
    short datatype;      /* Defines data type */
    short bitpix;        /* Number of bits per voxel */
    short slice_start;
    float pixdim[8];     /* Grid spacings */
    float vox_offset;    /* Offset into .nii file */
    float scl_slope;     /* Data scaling: slope */
    float scl_inter;     /* Data scaling: intercept */
    short slice_end;
    char  slice_code;
    char  xyzt_units;
    float cal_max;
    float cal_min;
    float slice_duration;
    float toffset;
    int   glmax;
    int   glmin;
    
    char  descrip[80];   /* Any text you like */
    char  aux_file[24];  /* Auxiliary filename */
    
    short qform_code;    /* NIFTI_XFORM_* code */
    short sform_code;    /* NIFTI_XFORM_* code */
    
    float quatern_b;     /* Quaternion b param */
    float quatern_c;     /* Quaternion c param */
    float quatern_d;     /* Quaternion d param */
    float qoffset_x;     /* Quaternion x shift */
    float qoffset_y;     /* Quaternion y shift */
    float qoffset_z;     /* Quaternion z shift */
    
    float srow_x[4];     /* 1st row affine transform */
    float srow_y[4];     /* 2nd row affine transform */
    float srow_z[4];     /* 3rd row affine transform */
    
    char intent_name[16];/* Name or meaning of data */
    char magic[4];       /* MUST be "ni1\0" or "n+1\0" */
} nifti_1_header;

/* NIfTI-1 data types */
#define DT_NONE                    0
#define DT_BINARY                  1
#define DT_UNSIGNED_CHAR           2
#define DT_SIGNED_SHORT            4
#define DT_SIGNED_INT              8
#define DT_FLOAT                  16
#define DT_COMPLEX                32
#define DT_DOUBLE                 64
#define DT_RGB                   128
#define DT_INT8                  256
#define DT_UINT16                512
#define DT_UINT32                768
#define DT_INT64                1024
#define DT_UINT64               1280
#define DT_FLOAT128             1536
#define DT_COMPLEX128           1792
#define DT_COMPLEX256           2048

/* NIfTI image structure */
typedef struct {
    int ndim;            /* Number of dimensions */
    int nx;              /* Dimensions of grid array */
    int ny;
    int nz;
    int nt;
    int nu;
    int nv;
    int nw;
    int nvox;            /* Number of voxels = nx*ny*nz*...*nw */
    int nbyper;          /* Bytes per voxel */
    int datatype;        /* Type of data in voxels */
    
    float dx;            /* Grid spacings */
    float dy;
    float dz;
    float dt;
    float du;
    float dv;
    float dw;
    
    float scl_slope;     /* Scaling parameters */
    float scl_inter;
    
    float cal_min;       /* Calibration parameters */
    float cal_max;
    
    int qform_code;      /* Codes for (x,y,z) space meaning */
    int sform_code;
    
    float quatern_b;     /* Quaternion parameters */
    float quatern_c;
    float quatern_d;
    float qoffset_x;
    float qoffset_y;
    float qoffset_z;
    
    float qfac;          /* Scaling factor for qform */
    
    float qto_xyz[4][4]; /* qform: transform (i,j,k) to (x,y,z) */
    float qto_ijk[4][4]; /* qform: transform (x,y,z) to (i,j,k) */
    
    float sto_xyz[4][4]; /* sform: transform (i,j,k) to (x,y,z) */
    float sto_ijk[4][4]; /* sform: transform (x,y,z) to (i,j,k) */
    
    float toffset;       /* Time axis shift */
    
    int xyz_units;       /* dx,dy,dz units */
    int time_units;      /* dt units */
    
    int nifti_type;      /* 0=analyze, 1=nifti-1 (1 file), 2=nifti-1 (2 files) */
    
    char *fname;         /* Header filename */
    char *iname;         /* Image filename */
    int iname_offset;    /* Offset into iname where data starts */
    int swapsize;        /* Swap unit in image data (might be 0) */
    int byteorder;       /* Byte order on disk */
    
    void *data;          /* Pointer to data */
    
    char descrip[80];    /* Text description */
} nifti_image;

/* Function prototypes */
nifti_image* nifti_image_read(const char *hname, int read_data);
void nifti_image_free(nifti_image *nim);
int nifti_image_load(nifti_image *nim);
void nifti_image_unload(nifti_image *nim);

/* Helper functions */
int is_nifti_file(const char *hname);
void swap_nifti_header(nifti_1_header *h, int is_nifti);

#ifdef __cplusplus
}
#endif

#endif /* NIFTI1_IO_H */
