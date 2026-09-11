cmake_minimum_required(VERSION 3.18)
project(flash_radiomics C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(
    FLASH_RADIOMICS_NATIVE_CPU
    "Use native CPU instruction tuning (-march=native or -mcpu=native). Disable for portable cluster builds."
    OFF
)
if(APPLE)
    set(CMAKE_C_FLAGS_RELEASE "-O3 -fPIC" CACHE STRING "Flags used by the C compiler during release builds" FORCE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    if(FLASH_RADIOMICS_NATIVE_CPU)
        set(CMAKE_C_FLAGS_RELEASE "-O3 -mcpu=native -fPIC" CACHE STRING "Flags used by the C compiler during release builds" FORCE)
    else()
        set(CMAKE_C_FLAGS_RELEASE "-O3 -fPIC" CACHE STRING "Flags used by the C compiler during release builds" FORCE)
    endif()
else()
    if(FLASH_RADIOMICS_NATIVE_CPU)
        set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -fPIC" CACHE STRING "Flags used by the C compiler during release builds" FORCE)
    else()
        set(CMAKE_C_FLAGS_RELEASE "-O3 -fPIC" CACHE STRING "Flags used by the C compiler during release builds" FORCE)
    endif()
endif()
message(STATUS "FLASH_RADIOMICS_NATIVE_CPU=${FLASH_RADIOMICS_NATIVE_CPU}")
message(STATUS "C release flags: ${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fPIC" CACHE STRING "Flags used by the C compiler during debug builds" FORCE)

option(ENABLE_CUDA "Enable CUDA GPU acceleration support" ON)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
            message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_TARGET_DIR}")
            
            set(CMAKE_CUDA_STANDARD 14)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC")
            
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

            set(CUDA_AVAILABLE TRUE)
            message(STATUS "CUDA support enabled - Building with GPU acceleration")
        else()
            set(CUDA_AVAILABLE FALSE)
            message(STATUS "CUDA Toolkit not found - Building without CUDA support")
        endif()
    else()
        set(CUDA_AVAILABLE FALSE)
        message(STATUS "CUDA compiler not found - Building without CUDA support")
    endif()
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA support disabled by user - Building without CUDA support")
endif()

set(CORE_SOURCES
    src/core/image_io.c
    src/core/dicom_converter.c
    src/core/utils.c
    src/core/nifti1_io.c
)

set(FEATURE_CPU_SOURCES
    src/features/firstorder_cpu.c
    src/features/glcm_cpu.c
    src/features/glrlm_cpu.c
    src/features/glszm_cpu.c
    src/features/gldm_cpu.c
    src/features/ngtdm_cpu.c
    src/features/cshape.c
    src/features/shape_cpu.c
)

set(API_SOURCES
    src/api/flash_radiomics_api.c
)

if(APPLE)
    set(MPS_DETECTION_SOURCE
        src/api/mps_detection.m
    )
endif()

if(APPLE)
    add_library(flash_radiomics_cpu SHARED
        ${CORE_SOURCES}
        ${FEATURE_CPU_SOURCES}
        ${API_SOURCES}
        ${MPS_DETECTION_SOURCE}
    )
else()
    add_library(flash_radiomics_cpu SHARED
        ${CORE_SOURCES}
        ${FEATURE_CPU_SOURCES}
        ${API_SOURCES}
    )
endif()

target_include_directories(flash_radiomics_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/features
    ${CMAKE_CURRENT_SOURCE_DIR}/src/api
)

if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    
    target_link_libraries(flash_radiomics_cpu PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        z
        m
    )
else()
    target_link_libraries(flash_radiomics_cpu PUBLIC
        z
        m
    )
endif()

if(APPLE)
    enable_language(OBJC)
    
    set(FEATURE_MPS_SOURCES
        src/mps/firstorder_mps.m
        src/mps/glcm_mps.m
        src/mps/glrlm_mps.m
        src/mps/glszm_mps.m
    )
    
    add_library(flash_radiomics_mps SHARED
        ${CORE_SOURCES}
        ${FEATURE_CPU_SOURCES}
        ${FEATURE_MPS_SOURCES}
        ${API_SOURCES}
        ${MPS_DETECTION_SOURCE}
    )
    
    target_include_directories(flash_radiomics_mps PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mps
        ${CMAKE_CURRENT_SOURCE_DIR}/src/api
    )
    
    target_compile_definitions(flash_radiomics_mps PRIVATE FLASH_RADIOMICS_MPS_ENABLED)
    
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    
    target_link_libraries(flash_radiomics_mps PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        z
        m
    )
endif()

if(CUDA_AVAILABLE)
    set(FEATURE_CUDA_SOURCES
        src/cuda/cuda_common.cu
        src/cuda/reduction_kernels.cu
        src/cuda/firstorder_cuda.cu
        src/cuda/glcm_cuda.cu
        src/cuda/glrlm_cuda.cu
        src/cuda/glszm_cuda.cu
        src/cuda/gldm_cuda.cu
        src/cuda/ngtdm_cuda.cu
        src/cuda/shape_cuda.cu
        src/cuda/voxel_batch_texture_cuda.cu
    )
    
    add_library(flash_radiomics_cuda SHARED
        ${CORE_SOURCES}
        ${FEATURE_CPU_SOURCES}
        ${FEATURE_CUDA_SOURCES}
        ${API_SOURCES}
    )
    
    if(APPLE)
        target_sources(flash_radiomics_cuda PRIVATE ${MPS_DETECTION_SOURCE})
    endif()
    
    target_include_directories(flash_radiomics_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/features
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/src/api
    )
    
    target_compile_definitions(flash_radiomics_cuda PRIVATE FLASH_RADIOMICS_CUDA_ENABLED)
    
    target_link_libraries(flash_radiomics_cuda PUBLIC
        CUDA::cudart
        z
        m
    )
    if(APPLE)
        target_link_libraries(flash_radiomics_cuda PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )
    endif()
    
    set_target_properties(flash_radiomics_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    message(STATUS "CUDA library target configured: flash_radiomics_cuda")
endif()

install(TARGETS flash_radiomics_cpu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(APPLE)
    install(TARGETS flash_radiomics_mps
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

if(CUDA_AVAILABLE)
    install(TARGETS flash_radiomics_cuda
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY src/core/ DESTINATION include/flash_radiomics
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY src/api/ DESTINATION include/flash_radiomics
    FILES_MATCHING PATTERN "*.h")
