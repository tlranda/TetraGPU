#!/bin/bash

# Arguments to this script each define separate files to use as inputs
#
# SOME ENVIRONMENT VARIABLES ARE READ BY THIS SCRIPT:
# RUNTIME_ARGS := Additional arguments to every single input file tested
# VALIDATE := Use the validation build and add --validate to runtime arguments

validate="${VALIDATE-0}";
echo "validate='${validate}'";

if [ $# -eq 0 ]; then
    set -- "${@:1}" "Bucket.vtu";
fi

if [[ "${validate}" == "0" ]]; then
    build_dir="build_${HOSTNAME}";
else
    build_dir="build_${HOSTNAME}_validate";
fi

cmake_command="CUDA_DIR=/usr/local/cuda-12.2 VTK_DIR=/home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7 cmake -B ${build_dir} -DCMAKE_CUDA_HOST_COMPILER=/home/tlranda/tools/gcc7/bin/g++ -DCMAKE_CXX_COMPILER=/home/tlranda/tools/gcc7/bin/g++";
if [[ "${validate}" != "0" ]]; then
    cmake_command="${cmake_command} -DVALIDATE_GPU=ON";
fi
echo $cmake_command;
eval $cmake_command;
if [ $? -ne 0 ]; then
    echo "CMake return $?";
    exit;
fi

cd ${build_dir} && make;
if [ $? -ne 0 ]; then
    echo "Make return $?";
    exit;
fi
cd ..;

for arg in $@; do
    task="time ./${build_dir}/main --input $arg -t 24 ${RUNTIME_ARGS} ";
    if [[ "${validate}" != "0" ]]; then
        task="${task} --validate";
    fi
    echo "${task}";
    eval "${task}";
done;

