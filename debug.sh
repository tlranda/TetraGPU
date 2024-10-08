#!/bin/bash

export CUDA_DIR=/usr/local/cuda-12.2
export VTK_DIR=/home/tlranda/tools/VTK/VTK-9.3.1/build
cmake -B build_n01 -DCMAKE_CUDA_HOST_COMPILER=/home/tlranda/tools/gcc7/bin/g++;
echo "CMake return $?";
cd build_n01;
make && gdb ./main;

