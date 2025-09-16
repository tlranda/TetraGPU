#!/bin/bash

# Arguments to this script each define separate files to use as inputs
#
# SOME ENVIRONMENT VARIABLES ARE READ BY THIS SCRIPT:
# RUNTIME_ARGS := Additional arguments to every single input file tested
# VALIDATE := Use the validation build and add --validate to runtime arguments
# DEBUG := Use Debug build type
# VTK_DIR := Help CMake find the VTK install directory
# CUDA_DIR := Help CMake find the CUDA install directory
# TETRA_CMAKE_ARGS := Additional arguments to CMake (ie: -DCMAKE_CUDA_HOST_COMPILER, -DCMake_CXX_COMPILER)
# EXE := The filename used for binary export
# MAIN := The main driver file to use
# COMPILE_ONLY := Only run cmake wrt other arguments, do not execute the program

validate="${VALIDATE-0}";
echo "validate='${validate}'";
debug="${DEBUG-0}";
echo "debug='${debug}'";
vtk_dir="${VTK_DIR-0}";
echo "VTK_DIR='${vtk_dir}'";
cuda_dir="${CUDA_DIR-0}";
echo "CUDA_DIR='${cuda_dir}'";
cmake_args="${TETRA_CMAKE_ARGS-0}";
echo "cmake_args='${cmake_args}'";
exe="${EXE-0}";
echo "exe='${exe}'";
main="${MAIN-0}";
echo "main='${main}'";
compile_only="${COMPILE_ONLY-0}";
echo "compile_only='${compile_only}'";

if [ $# -eq 0 ]; then
    set -- "${@:1}" "Bucket.vtu";
fi

if [[ "${validate}" == "0" ]]; then
    build_dir="build_${HOSTNAME}";
else
    build_dir="build_${HOSTNAME}_validate";
    if [[ "${RUNTIME_ARGS}" != *"--validate"* ]]; then
        RUNTIME_ARGS="${RUNTIME_ARGS} --validate";
    fi
fi

if [[ "${debug}" == "0" ]]; then
    build_type="Release";
else
    build_type="Debug";
fi

if [[ "${vtk_dir}" != "0" ]]; then
    cmake_command="VTK_DIR=${vtk_dir}";
else
    cmake_command="";
fi
if [[ "${cuda_dir}" != "0" ]]; then
    cmake_command="${cmake_command} CUDA_DIR=${cuda_dir}";
fi
cmake_command="${cmake_command} cmake -B ${build_dir} -DCMAKE_BUILD_TYPE=${build_type}";
if [[ "${cmake_args}" != "0" ]]; then
    cmake_command="${cmake_command} ${cmake_args}";
fi
if [[ "${validate}" != "0" ]]; then
    cmake_command="${cmake_command} -DVALIDATE_GPU=ON";
fi
if [[ "${debug}" != "0" ]]; then
    cmake_command="${cmake_command} -DCMAKE_BUILD_TYPE=\"Debug\"";
fi
if [[ "${exe}" == "0" ]]; then
    exe="main";
fi
cmake_command="${cmake_command} -DCMAKE_OUTPUT_NAME=\"${exe}\"";
if [[ "${main}" != "0" ]]; then
    lines=$(diff src/main.cu ${main} | wc -l);
    if [[ ${lines} -ne 0 ]]; then
        echo "Updating main file";
        cp ${main} src/main.cu;
    else
        echo "Main file up-to-date";
    fi
fi

echo $cmake_command;
eval $cmake_command;
if [ $? -ne 0 ]; then
    echo "CMake return $?";
    exit;
fi

cd ${build_dir} && make VERBOSE=1;
if [ $? -ne 0 ]; then
    echo "Make return $?";
    exit $?;
fi
if [[ "${compile_only}" != "0" ]]; then
    exit 0;
fi
cd ..;

if [[ "${debug}" == "0" ]]; then
    for arg in $@; do
        task="time ./${build_dir}/${exe} --input $arg ${RUNTIME_ARGS} ";
        echo "${task}";
        eval "${task}";
    done;
else
    echo "Suggested run command:";
    echo "--input $1 ${RUNTIME_ARGS}";
    cuda-gdb ${build_dir}/${exe};
fi

