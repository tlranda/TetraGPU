# TetraGPU: Tetrahedral Mesh Processing for GPUs

This repository represents an effort to bring generalizable tetrahedral mesh
processing to GPU architectures. General triangular mesh processing exists for
GPUs, however tetrahedral algorithms require different solutions.

The current goal for this project is to fully implement the simplex
relationships necessary to test the initial hypotheses against select algorithms.

## Setup and Install

### Requirements
* CMake >= 3.12
* [VTK == 9.3.1](htts://gitlab.kitware.com/vtk/vtk/-/tree/v9.3.1)
* NVIDIA GPU (Tested on minimum SM\_52 Compute-Capability @ CUDA 12.2.128)

### BASH Shell Environment
* Define the environment variable `VTK_DIR` such that it points to a built and
installed VTK base directory
    * If building from source, this path may resemble: `<...>/VTK/VTK-#.#.#/build`
    * ie: `$ export VTK_DIR=<path/to/VTK/build>`

### CMake Modifications
* The included CMakeLists.txt should locate your CUDA installation normally,
however if you have an install like mine that requires some hints on host
compilers, you can replicate the structure for "n01" in the existing CMakeList

### Tetrahedral Dataset
* Supply a VTK-compatible tetrahedral XML unstructured grid file (.vtu)
* Alternatively, with Python3 and some common dependencies, you can run
`python3 make_mesh.py --help` to see how to generate a simple mesh
    * Since this file is NOT required to run the C++/CUDA code, there isn't a
fixed set of dependencies.
    * None of the dependent modules are explicitly noted to utilize behaviors
requiring specific versions; however they are tested against:
        * Numpy Version 1.23.0
        * Meshio Version 5.3.5
        * TQDM Version 4.62.3

### Building, Running, Debugging
* All builds can be managed directly by CMake, but for my own convenience I
use the included `test.sh` script. You may use it as well:
    * The script always builds/updates the repository source via CMake
    * If you set/export a nonempty value for `RUNTIME_ARGS`, the string will be
forwarded to `test.sh`'s executions of the program as additional arguments.
    * All other arguments to `test.sh` are assumed to be `.vtu` files used as
arguments to the compiled program. Each file will require a separate program
execution (serialized)
* Other environment tweaks affecting `test.sh`:
    * If you set/export a nonempty value for `VALIDATE`, a separate build
directory is used and additional sources / definitions are provided at
compile-time to use the CPU to validate all results returned from the GPU when
the runtime flag `--validate` is provided (this flag is added by default by
`test.sh`'s executions of the program).
    * If you set/export a nonempty value for `DEBUG`, CMake uses Debug build
rules instead of the Release target. `test.sh` will suggest arguments to run
the program and launch CUDA-GDB instead of directly executing the program.

Example usage may look like this:
```/bin/bash
$ ./test.sh;
validate='0'
debug='0'
VTK_DIR='/home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7'
CUDA_DIR='/usr/local/cuda-12.2'
cmake_args='0'
VTK_DIR=/home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7 CUDA_DIR=/usr/local/cuda-12.2 cmake -B build_n01 -DCMAKE_BUILD_TYPE=Release
-- CMAKE OUTPUT LARGELY OMITTED, MAY VARY BASED ON PRIOR BUILD STATUS --
-- Adding custom NVIDIA setup for n01
-- VTK_VERSION: 9.3.1
VTK_DIR: /home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7
Adjusting host flags to O3 for release build...
-- Configuring done (0.4s)
-- Generating done (0.1s)
-- Build files have been written to: /home/tlranda/TetraGPU/build_n01
[100%] Built target main
time ./build_n01/main --input Bucket.vtu -t 24  
ℹ️  Dataset: Bucket.vtu
ℹ️  CPU threads: 24
ℹ️  Bit flag [1: TE]: false
ℹ️  Bit flag [2: EV]: false
ℹ️  Bit flag [3: ET]: false
ℹ️  Bit flag [4: TF]: false
ℹ️  Bit flag [5: FV]: false
ℹ️  Bit flag [6: FE]: false
⏳ Argument parsing: 0.000107
⏳ GPU context creation with dummy kernel: 0.255451
⏳ GPU trivial kernel launch: 0.000010
📍 Parsing vtu file: Bucket.vtu
🆗 Dataset loaded with 84156 tetrahedra and 19917 vertices
⏳ TV from VTK: 0.035009
📍 Building edges...
⏳ VE [CPU]: 0.022587
🆗 Built 114520 edges.
📍 Building faces...
⏳ VF [CPU]: 0.029375
🆗 Built 178760 faces.
❗ Open interval 6 cannot report time until it is closed by another tick()

real	0m0.426s
user	0m0.111s
sys	0m0.284s
```

## Current State of the Repository:

This is a pre-production, in-development academic research project.

It is feature incomplete.

Features that are complete may not be sufficiently robust for production usage.

Currently, the project is developed by one developer; meaningful contributions
are not expected from random passersby and could require significant effort to
accept. As such, please email me if you are interested in contributing PRIOR to
beginning your own work so we may better coordinate efforts.

