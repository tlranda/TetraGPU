#!/bin/bash

VTK_DIR=/home/tlranda/tools/VTK/VTK-9.3.1/build cmake -B build_n01;
cd build_n01;
make && gdb ./bin/main;

