/* nifti1_io.c - Basic NIfTI-1 I/O routines
 * Simplified implementation for flash_radiomics project
 */

#include "nifti1_io.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <zlib.h>

/* Byte swapping macros */
#define SWAP_2(x) ((((x) & 0xff) << 8) | (((x) >> 8) & 0xff))
#define SWAP_4(x) ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >> 8) | \
                   (((x) & 0x0000ff00) << 8) | (((x) & 0x000000ff) << 24))

static void swap_2bytes(void *ptr, int n) {
    uint16_t *p = (uint16_t *)ptr;
    for (int i = 0; i < n; i++) {
        p[i] = SWAP_2(p[i]);
    }
}

static void swap_4bytes(void *ptr, int n) {
    uint32_t *p = (uint32_t *)ptr;
    for (int i = 0; i < n; i++) {
        p[i] = SWAP_4(p[i]);
    }
}

static void swap_8bytes(void *ptr, int n) {
    uint64_t *p = (uint64_t *)ptr;
    for (int i = 0; i < n; i++) {
        uint64_t val = p[i];
        p[i] = ((val & 0xff00000000000000ULL) >> 56) |
               ((val & 0x00ff000000000000ULL) >> 40) |
               ((val & 0x0000ff0000000000ULL) >> 24) |
               ((val & 0x000000ff00000000ULL) >> 8) |
               ((val & 0x00000000ff000000ULL) << 8) |
               ((val & 0x0000000000ff0000ULL) << 24) |
               ((val & 0x000000000000ff00ULL) << 40) |
               ((val & 0x00000000000000ffULL) << 56);
    }
}

/* Check if system is big-endian */
static int is_big_endian(void) {
    uint16_t test = 0x0001;
    return (*(uint8_t *)&test) == 0;
}

/* Swap NIfTI header bytes if needed */
void swap_nifti_header(nifti_1_header *h, int is_nifti) {
    swap_4bytes(&h->sizeof_hdr, 1);
    swap_4bytes(&h->extents, 1);
    swap_2bytes(&h->session_error, 1);
    swap_2bytes(h->dim, 8);
    swap_4bytes(&h->intent_p1, 3);
    swap_2bytes(&h->intent_code, 1);
    swap_2bytes(&h->datatype, 1);
    swap_2bytes(&h->bitpix, 1);
    swap_2bytes(&h->slice_start, 1);
    swap_4bytes(h->pixdim, 8);
    swap_4bytes(&h->vox_offset, 1);
    swap_4bytes(&h->scl_slope, 1);
    swap_4bytes(&h->scl_inter, 1);
    swap_2bytes(&h->slice_end, 1);
    swap_4bytes(&h->cal_max, 1);
    swap_4bytes(&h->cal_min, 1);
    swap_4bytes(&h->slice_duration, 1);
    swap_4bytes(&h->toffset, 1);
    swap_4bytes(&h->glmax, 1);
    swap_4bytes(&h->glmin, 1);
    swap_2bytes(&h->qform_code, 1);
    swap_2bytes(&h->sform_code, 1);
    swap_4bytes(&h->quatern_b, 1);
    swap_4bytes(&h->quatern_c, 1);
    swap_4bytes(&h->quatern_d, 1);
    swap_4bytes(&h->qoffset_x, 1);
    swap_4bytes(&h->qoffset_y, 1);
    swap_4bytes(&h->qoffset_z, 1);
    swap_4bytes(h->srow_x, 4);
    swap_4bytes(h->srow_y, 4);
    swap_4bytes(h->srow_z, 4);
}

/* Check if file is gzipped */
static int is_gzipped(const char *fname) {
    const char *ext = strrchr(fname, '.');
    return (ext && strcmp(ext, ".gz") == 0);
}

/* Check if file is NIfTI format */
int is_nifti_file(const char *hname) {
    nifti_1_header hdr;
    int is_gz = is_gzipped(hname);
    
    if (is_gz) {
        /* Read from gzipped file */
        gzFile gzfp = gzopen(hname, "rb");
        if (!gzfp) return 0;
        
        if (gzread(gzfp, &hdr, sizeof(hdr)) < (int)sizeof(hdr)) {
            gzclose(gzfp);
            return 0;
        }
        gzclose(gzfp);
    } else {
        /* Read from regular file */
        FILE *fp = fopen(hname, "rb");
        if (!fp) return 0;
        
        if (fread(&hdr, 1, sizeof(hdr), fp) < sizeof(hdr)) {
            fclose(fp);
            return 0;
        }
        fclose(fp);
    }
    
    /* Check magic string */
    if (strncmp(hdr.magic, "n+1", 3) == 0 || strncmp(hdr.magic, "ni1", 3) == 0) {
        return 1;
    }
    
    /* Try byte-swapped version */
    swap_nifti_header(&hdr, 1);
    if (strncmp(hdr.magic, "n+1", 3) == 0 || strncmp(hdr.magic, "ni1", 3) == 0) {
        return 1;
    }
    
    return 0;
}

/* Get bytes per voxel for datatype */
static int nifti_datatype_sizes(int datatype) {
    switch (datatype) {
        case DT_BINARY:         return 0;  /* 1 bit per voxel */
        case DT_UNSIGNED_CHAR:
        case DT_INT8:           return 1;
        case DT_SIGNED_SHORT:
        case DT_UINT16:         return 2;
        case DT_SIGNED_INT:
        case DT_UINT32:
        case DT_FLOAT:          return 4;
        case DT_COMPLEX:
        case DT_DOUBLE:
        case DT_INT64:
        case DT_UINT64:         return 8;
        case DT_FLOAT128:       return 16;
        case DT_RGB:            return 3;
        case DT_COMPLEX128:     return 16;
        case DT_COMPLEX256:     return 32;
        default:                return 0;
    }
}

/* Read NIfTI image */
nifti_image* nifti_image_read(const char *hname, int read_data) {
    nifti_1_header hdr;
    nifti_image *nim;
    int need_swap = 0;
    int is_gz = is_gzipped(hname);
    
    if (!hname) return NULL;
    
    /* Open and read header */
    if (is_gz) {
        gzFile gzfp = gzopen(hname, "rb");
        if (!gzfp) {
            fprintf(stderr, "::ERROR:: Cannot open gzipped NIfTI file: %s\n", hname);
            return NULL;
        }
        
        if (gzread(gzfp, &hdr, sizeof(hdr)) < (int)sizeof(hdr)) {
            fprintf(stderr, "::ERROR:: Cannot read NIfTI header from: %s\n", hname);
            gzclose(gzfp);
            return NULL;
        }
        gzclose(gzfp);
    } else {
        FILE *fp = fopen(hname, "rb");
        if (!fp) {
            fprintf(stderr, "::ERROR:: Cannot open NIfTI file: %s\n", hname);
            return NULL;
        }
        
        if (fread(&hdr, 1, sizeof(hdr), fp) < sizeof(hdr)) {
            fprintf(stderr, "::ERROR:: Cannot read NIfTI header from: %s\n", hname);
            fclose(fp);
            return NULL;
        }
        fclose(fp);
    }
    
    /* Check if we need to swap bytes */
    if (hdr.sizeof_hdr != 348) {
        need_swap = 1;
        swap_nifti_header(&hdr, 1);
        if (hdr.sizeof_hdr != 348) {
            fprintf(stderr, "::ERROR:: Invalid NIfTI header size: %d\n", hdr.sizeof_hdr);
            return NULL;
        }
    }
    
    /* Allocate nifti_image structure */
    nim = (nifti_image *)calloc(1, sizeof(nifti_image));
    if (!nim) {
        return NULL;
    }
    
    /* Fill in basic info */
    nim->ndim = hdr.dim[0];
    nim->nx = (hdr.dim[1] > 0) ? hdr.dim[1] : 1;
    nim->ny = (hdr.dim[2] > 0) ? hdr.dim[2] : 1;
    nim->nz = (hdr.dim[3] > 0) ? hdr.dim[3] : 1;
    nim->nt = (hdr.dim[4] > 0) ? hdr.dim[4] : 1;
    nim->nu = (hdr.dim[5] > 0) ? hdr.dim[5] : 1;
    nim->nv = (hdr.dim[6] > 0) ? hdr.dim[6] : 1;
    nim->nw = (hdr.dim[7] > 0) ? hdr.dim[7] : 1;
    
    nim->nvox = nim->nx * nim->ny * nim->nz * nim->nt * nim->nu * nim->nv * nim->nw;
    
    nim->dx = hdr.pixdim[1];
    nim->dy = hdr.pixdim[2];
    nim->dz = hdr.pixdim[3];
    nim->dt = hdr.pixdim[4];
    nim->du = hdr.pixdim[5];
    nim->dv = hdr.pixdim[6];
    nim->dw = hdr.pixdim[7];
    
    nim->datatype = hdr.datatype;
    nim->nbyper = nifti_datatype_sizes(hdr.datatype);
    
    nim->scl_slope = hdr.scl_slope;
    nim->scl_inter = hdr.scl_inter;
    nim->cal_min = hdr.cal_min;
    nim->cal_max = hdr.cal_max;
    
    nim->qform_code = hdr.qform_code;
    nim->sform_code = hdr.sform_code;
    
    nim->quatern_b = hdr.quatern_b;
    nim->quatern_c = hdr.quatern_c;
    nim->quatern_d = hdr.quatern_d;
    nim->qoffset_x = hdr.qoffset_x;
    nim->qoffset_y = hdr.qoffset_y;
    nim->qoffset_z = hdr.qoffset_z;
    
    /* Copy sform matrix */
    for (int i = 0; i < 4; i++) {
        nim->sto_xyz[0][i] = hdr.srow_x[i];
        nim->sto_xyz[1][i] = hdr.srow_y[i];
        nim->sto_xyz[2][i] = hdr.srow_z[i];
    }
    nim->sto_xyz[3][0] = nim->sto_xyz[3][1] = nim->sto_xyz[3][2] = 0.0f;
    nim->sto_xyz[3][3] = 1.0f;
    
    strncpy(nim->descrip, hdr.descrip, 79);
    nim->descrip[79] = '\0';
    
    /* Determine file type */
    if (strncmp(hdr.magic, "n+1", 3) == 0) {
        nim->nifti_type = 1;  /* Single file .nii */
        nim->iname_offset = (int)hdr.vox_offset;
    } else if (strncmp(hdr.magic, "ni1", 3) == 0) {
        nim->nifti_type = 2;  /* Two files .hdr/.img */
        nim->iname_offset = 0;
    } else {
        nim->nifti_type = 0;  /* Analyze format */
        nim->iname_offset = 0;
    }
    
    /* Store filenames */
    nim->fname = strdup(hname);
    nim->iname = strdup(hname);
    
    /* For .hdr files, change extension to .img */
    if (nim->nifti_type == 2) {
        char *ext = strrchr(nim->iname, '.');
        if (ext && strcmp(ext, ".hdr") == 0) {
            strcpy(ext, ".img");
        }
    }
    
    nim->byteorder = need_swap ? (!is_big_endian()) : is_big_endian();
    nim->swapsize = nim->nbyper;
    
    /* Read data if requested */
    if (read_data) {
        if (nifti_image_load(nim) < 0) {
            nifti_image_free(nim);
            return NULL;
        }
    }
    
    return nim;
}

/* Load image data */
int nifti_image_load(nifti_image *nim) {
    size_t bytes_to_read;
    int is_gz;
    
    if (!nim || !nim->iname) return -1;
    if (nim->data) return 0;  /* Already loaded */
    
    bytes_to_read = (size_t)nim->nvox * nim->nbyper;
    if (bytes_to_read == 0) return -1;
    
    nim->data = malloc(bytes_to_read);
    if (!nim->data) {
        fprintf(stderr, "::ERROR:: Cannot allocate %zu bytes for image data\n", bytes_to_read);
        return -1;
    }
    
    is_gz = is_gzipped(nim->iname);
    
    if (is_gz) {
        /* Read from gzipped file */
        gzFile gzfp = gzopen(nim->iname, "rb");
        if (!gzfp) {
            fprintf(stderr, "::ERROR:: Cannot open gzipped image file: %s\n", nim->iname);
            free(nim->data);
            nim->data = NULL;
            return -1;
        }
        
        /* Seek to data offset */
        if (nim->iname_offset > 0) {
            if (gzseek(gzfp, nim->iname_offset, SEEK_SET) < 0) {
                fprintf(stderr, "::ERROR:: Cannot seek to data offset in: %s\n", nim->iname);
                gzclose(gzfp);
                free(nim->data);
                nim->data = NULL;
                return -1;
            }
        }
        
        /* Read data */
        if (gzread(gzfp, nim->data, bytes_to_read) < (int)bytes_to_read) {
            fprintf(stderr, "::ERROR:: Cannot read image data from: %s\n", nim->iname);
            gzclose(gzfp);
            free(nim->data);
            nim->data = NULL;
            return -1;
        }
        
        gzclose(gzfp);
    } else {
        /* Read from regular file */
        FILE *fp = fopen(nim->iname, "rb");
        if (!fp) {
            fprintf(stderr, "::ERROR:: Cannot open image file: %s\n", nim->iname);
            free(nim->data);
            nim->data = NULL;
            return -1;
        }
        
        /* Seek to data offset */
        if (fseek(fp, nim->iname_offset, SEEK_SET) != 0) {
            fprintf(stderr, "::ERROR:: Cannot seek to data offset in: %s\n", nim->iname);
            fclose(fp);
            free(nim->data);
            nim->data = NULL;
            return -1;
        }
        
        /* Read data */
        if (fread(nim->data, 1, bytes_to_read, fp) < bytes_to_read) {
            fprintf(stderr, "::ERROR:: Cannot read image data from: %s\n", nim->iname);
            fclose(fp);
            free(nim->data);
            nim->data = NULL;
            return -1;
        }
        
        fclose(fp);
    }
    
    /* Swap bytes if needed */
    if (nim->byteorder != is_big_endian() && nim->swapsize > 1) {
        switch (nim->swapsize) {
            case 2: swap_2bytes(nim->data, nim->nvox); break;
            case 4: swap_4bytes(nim->data, nim->nvox); break;
            case 8: swap_8bytes(nim->data, nim->nvox); break;
        }
    }
    
    return 0;
}

/* Unload image data */
void nifti_image_unload(nifti_image *nim) {
    if (nim && nim->data) {
        free(nim->data);
        nim->data = NULL;
    }
}

/* Free nifti_image structure */
void nifti_image_free(nifti_image *nim) {
    if (!nim) return;
    
    if (nim->data) free(nim->data);
    if (nim->fname) free(nim->fname);
    if (nim->iname) free(nim->iname);
    
    free(nim);
}
