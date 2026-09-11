#include "dicom_converter.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

/* Helper function to check if file has DICOM extension */
static int is_dicom_file(const char* filename) {
    if (!filename) return 0;
    
    const char* ext = strrchr(filename, '.');
    if (!ext) {
        /* DICOM files may not have extension */
        return 1;
    }
    
    /* Common DICOM extensions */
    if (strcmp(ext, ".dcm") == 0 || strcmp(ext, ".DCM") == 0 ||
        strcmp(ext, ".dicom") == 0 || strcmp(ext, ".DICOM") == 0) {
        return 1;
    }
    
    return 0;
}

/* Count DICOM files in directory */
static int count_dicom_files(const char* dicom_dir) {
    DIR* dir;
    struct dirent* entry;
    int count = 0;
    
    dir = opendir(dicom_dir);
    if (!dir) {
        return 0;
    }
    
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        /* Check if it's a regular file */
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", dicom_dir, entry->d_name);
        
        struct stat statbuf;
        if (stat(filepath, &statbuf) == 0 && S_ISREG(statbuf.st_mode)) {
            if (is_dicom_file(entry->d_name)) {
                count++;
            }
        }
    }
    
    closedir(dir);
    return count;
}

/* Convert DICOM file or series to NIfTI using Python SimpleITK
 * This function creates a temporary Python script and executes it
 * to perform the conversion using SimpleITK
 * Supports both single DICOM files (2D) and DICOM series directories (3D)
 */
int flash_radiomics_dicom_to_nifti(const char* dicom_path, const char* output_nifti) {
    if (!dicom_path || !output_nifti) {
        return FLASH_RADIOMICS_ERROR_INVALID_PARAMETER;
    }
    
    /* Check if path exists */
    struct stat statbuf;
    if (stat(dicom_path, &statbuf) != 0) {
        fprintf(stderr, "::ERROR:: DICOM path not found: %s\n", dicom_path);
        return FLASH_RADIOMICS_ERROR_FILE_NOT_FOUND;
    }
    
    /* If it's a directory, check if it contains DICOM files */
    if (S_ISDIR(statbuf.st_mode)) {
        int num_files = count_dicom_files(dicom_path);
        if (num_files == 0) {
            fprintf(stderr, "::ERROR:: No DICOM files found in directory: %s\n", dicom_path);
            return FLASH_RADIOMICS_ERROR_INVALID_FORMAT;
        }
    }
    /* If it's a file, verify it's a DICOM file */
    else if (S_ISREG(statbuf.st_mode)) {
        const char* filename = strrchr(dicom_path, '/');
        filename = filename ? filename + 1 : dicom_path;
        if (!is_dicom_file(filename)) {
            fprintf(stderr, "::WARN:: File may not be a DICOM file: %s\n", dicom_path);
        }
    }
    else {
        fprintf(stderr, "::ERROR:: DICOM path is neither a file nor directory: %s\n", dicom_path);
        return FLASH_RADIOMICS_ERROR_INVALID_FORMAT;
    }
    
    /* Create temporary Python script for conversion */
    char script_path[] = "/tmp/flash_radiomics_dicom_convert_XXXXXX";
    int fd = mkstemp(script_path);
    if (fd == -1) {
        fprintf(stderr, "::ERROR:: Failed to create temporary script file\n");
        return FLASH_RADIOMICS_ERROR_IO_FAILED;
    }
    
    /* Write Python script */
    FILE* script_file = fdopen(fd, "w");
    if (!script_file) {
        close(fd);
        unlink(script_path);
        return FLASH_RADIOMICS_ERROR_IO_FAILED;
    }
    
    fprintf(script_file,
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import os\n"
        "try:\n"
        "    import SimpleITK as sitk\n"
        "except ImportError:\n"
        "    print('::ERROR:: SimpleITK not installed', file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "\n"
        "def convert_dicom_to_nifti(dicom_path, output_nifti):\n"
        "    try:\n"
        "        # Check if path is a directory or a single file\n"
        "        if os.path.isdir(dicom_path):\n"
        "            # Read DICOM series from directory\n"
        "            reader = sitk.ImageSeriesReader()\n"
        "            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)\n"
        "            \n"
        "            if len(dicom_names) == 0:\n"
        "                print(f'::ERROR:: No DICOM series found in {dicom_path}', file=sys.stderr)\n"
        "                return False\n"
        "            \n"
        "            reader.SetFileNames(dicom_names)\n"
        "            image = reader.Execute()\n"
        "        else:\n"
        "            # Read single DICOM file (2D image)\n"
        "            image = sitk.ReadImage(dicom_path)\n"
        "        \n"
        "        # Write as NIfTI\n"
        "        sitk.WriteImage(image, output_nifti)\n"
        "        return True\n"
        "        \n"
        "    except Exception as e:\n"
        "        print(f'::ERROR:: DICOM conversion failed: {e}', file=sys.stderr)\n"
        "        return False\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    if len(sys.argv) != 3:\n"
        "        print('Usage: script.py <dicom_path> <output_nifti>', file=sys.stderr)\n"
        "        sys.exit(1)\n"
        "    \n"
        "    dicom_path = sys.argv[1]\n"
        "    output_nifti = sys.argv[2]\n"
        "    \n"
        "    if convert_dicom_to_nifti(dicom_path, output_nifti):\n"
        "        sys.exit(0)\n"
        "    else:\n"
        "        sys.exit(1)\n"
    );
    
    fclose(script_file);
    
    /* Make script executable */
    chmod(script_path, 0755);
    
    /* Execute Python script */
    char command[2048];
    snprintf(command, sizeof(command), "python3 %s '%s' '%s' 2>&1",
             script_path, dicom_path, output_nifti);
    
    int result = system(command);
    
    /* Clean up temporary script */
    unlink(script_path);
    
    if (result != 0) {
        fprintf(stderr, "::ERROR:: DICOM to NIfTI conversion failed\n");
        return FLASH_RADIOMICS_ERROR_IO_FAILED;
    }
    
    /* Verify output file was created */
    if (stat(output_nifti, &statbuf) != 0) {
        fprintf(stderr, "::ERROR:: Output NIfTI file not created: %s\n", output_nifti);
        return FLASH_RADIOMICS_ERROR_IO_FAILED;
    }
    
    return FLASH_RADIOMICS_SUCCESS;
}
