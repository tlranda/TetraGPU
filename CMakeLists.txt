cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(tetra_topo
    VERSION 0.0.1
    DESCRIPTION "GPU Tetrahedron Mesh Processing"
    LANGUAGES CXX)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    set(default_build_type "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(VTK PATHS ~/usr/local)
find_package(VTK)

if(NOT VTK_FOUND)
    message("VTK NOT FOUND!")
    return()
endif()

message(STATUS "VTK_VERSION: ${VTK_VERSION}")
message("VTK_DIR: ${VTK_DIR}")
# message("VTK_LIBRARIES: ${VTK_LIBRARIES}")

add_executable(main main.cpp)
target_compile_options(main PRIVATE -g -O0)
target_link_libraries(main PRIVATE VTK::IOXML)

