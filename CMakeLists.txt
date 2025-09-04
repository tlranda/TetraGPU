cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

project(tetra_topo
    VERSION 0.0.1
    DESCRIPTION "GPU Tetrahedron Mesh Processing"
    LANGUAGES CXX CUDA)

file(GLOB sources src/*.cu src/*.cpp)

# Options
option(VALIDATE_GPU "Include CPU validation of GPU results" OFF)
if (NOT VALIDATE_GPU)
    # Remove validate.cpp from file globbing
    list(REMOVE_ITEM sources src/validate.cpp)
endif (NOT VALIDATE_GPU)
set(CMAKE_OUTPUT_NAME "main" CACHE STRING "Name of the output executable")
add_executable(main ${sources})
set_target_properties(main PROPERTIES OUTPUT_NAME ${CMAKE_OUTPUT_NAME})
set_target_properties(main PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
include_directories(${CMAKE_BINARY_DIR})

# CUDA Dependency
# N01 has a whack installation that has to be adjusted
execute_process(COMMAND hostname
                OUTPUT_VARIABLE HOSTNAME
                OUTPUT_STRIP_TRAILING_WHITESPACE)
#if (HOSTNAME STREQUAL "n01")
#    message(STATUS "Adding custom NVIDIA setup for n01")
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
#    include_directories(/usr/local/cuda-12.2/include /home/tlranda/tools/gcc7/include/c++/7.5.0 /home/tlranda/tools/gcc7/include/c++/7.5.0/x86_64-pc-linux-gnu)
#    link_directories(/usr/local/cuda-12.2/libnvvp /usr/local/cuda-12.2/lib64 /home/tlranda/tools/gcc7/lib)
#    target_link_libraries(main PRIVATE -L/home/tlranda/tools/gcc7/lib -lstdc++)
#elseif(HOSTNAME STREQUAL "voltron")
#    message(STATUS "Adding custom NVIDIA setup for voltron")
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
#    include_directories(/usr/local/cuda-12.5/include)
#    link_directories(/usr/local/cuda-12.5/libnvvp /usr/local/cuda-12.5/lib64)
#    target_link_libraries(main PRIVATE -lstdc++)
target_link_libraries(main PRIVATE -lcudart)
#else()
#    message(STATUS "Not host 'n01', no ccbin aded")
#    find_package(CUDA PATHS /usr/local/)
#    if(NOT CUDA_FOUND)
#        message("CUDA INSTALLATION NOT FOUND!")
#        return()
#    endif()
set(CMAKE_CUDA_HOST_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
#endif()

# VTK Dependency
find_package(VTK)
if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK NOT FOUND! Set VTK_DIR environment variable to manually identify it")
endif()
message(STATUS "VTK_VERSION: ${VTK_VERSION}")
message("VTK_DIR: ${VTK_DIR}")
# message("VTK_LIBRARIES: ${VTK_LIBRARIES}")

# C++ / CUDA settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_ARCHITECTURES "61")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
target_compile_options(main PRIVATE -Wall -Wextra -Wno-pedantic) # -Werror

# Set a default build type if none was specified
set(default_build_type "Release")
#if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
#set(default_build_type "Debug")
#endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif()

# Adjustments for build types
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Adjusting CUDA FLAGS for debug build...")
    target_compile_options(main PRIVATE -g -lineinfo -O0)
    #target_compile_options(main CUDA::CUDA_OPTIONS PRIVATE -G -O0)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0 -g -lineinfo")
    #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -O0 -g -lineinfo")
else()
    message("Adjusting host flags to O3 for release build...")
    target_compile_options(main PRIVATE -O3)
    #target_compile_options(main CUDA::CUDA_OPTIONS PRIVATE -O3)
endif()

# Final build
configure_file(include/datatypes.h.in datatypes.h)
target_link_libraries(main PRIVATE VTK::IOXML)
target_include_directories(main PRIVATE include)
install(TARGETS main)

