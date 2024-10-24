#!/bin/bash

if [ $# -eq 0 ]; then
    set -- "${@:1}" "Bucket.vtu";
fi

CUDA_DIR=/usr/local/cuda-12.2 VTK_DIR=/home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7 cmake -B build_${HOSTNAME} -DCMAKE_CUDA_HOST_COMPILER=/home/tlranda/tools/gcc7/bin/g++ -DCMAKE_CXX_COMPILER=/home/tlranda/tools/gcc7/bin/g++;
if [ $? -ne 0 ]; then
    echo "CMake return $?";
    exit;
fi

cd build_${HOSTNAME} && make;
if [ $? -ne 0 ]; then
    echo "Make return $?";
    exit;
fi
cd ..;

for arg in $@; do
    task="time ./build_${HOSTNAME}/main --input $arg -t 24";
    echo "${task}";
    eval "${task}";
done;

