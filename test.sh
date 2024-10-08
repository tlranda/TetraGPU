#!/bin/bash

#export 
#export 
CUDA_DIR=/usr/local/cuda-12.2 VTK_DIR=/home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7 cmake -B build_n01 -DCMAKE_CUDA_HOST_COMPILER=/home/tlranda/tools/gcc7/bin/g++ -DCMAKE_CXX_COMPILER=/home/tlranda/tools/gcc7/bin/g++;
echo "CMake return $?";
cd build_n01;
make && time ./main --input ../Bucket.vtu -t 24;

