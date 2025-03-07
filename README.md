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
* To make dataset replication as simple as possible, you should be able to
recreate the datasets by running `./artificial_dataset.sh`, which will drive
the Python script on your behalf. This process will take a few minutes -- we
note that TQDM's time-to-completion estimate is overly optimistic on the final
dataset due to the nonlinear time complexity of the timed loop.

### Building, Running, Debugging
* All builds can be managed directly by CMake, but for my own convenience I
use the included `test.sh` script. You may use it as well:
    * The environment variable "MAIN" must be set to a driver (.cu file under
the src/alg directory) if you have not run the script before. This ensures
multiple drivers can be swapped in and out transparently for development and
may change in the future.
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
$ MAIN=src/alg/test_driver.cu ./test.sh;
validate='0'
debug='0'
VTK_DIR='/home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7'
CUDA_DIR='/usr/local/cuda-12.2'
cmake_args='-DCMAKE_CUDA_HOST_COMPILER=/home/tlranda/tools/gcc7/bin/g++ -DCMAKE_CXX_COMPILER=/home/tlranda/tools/gcc7/bin/g++'
exe='0'
main='src/alg/test_driver.cu'
Main file up-to-date
VTK_DIR=/home/tlranda/tools/VTK/VTK-9.3.1/build_gcc7 CUDA_DIR=/usr/local/cuda-12.2 cmake -B build_n01 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=/home/tlranda/tools/gcc7/bin/g++ -DCMAKE_CXX_COMPILER=/home/tlranda/tools/gcc7/bin/g++ -DCMAKE_OUTPUT_NAME="main"
-- CMAKE OUTPUT LARGELY OMITTED, MAY VARY BASED ON PRIOR BUILD STATUS --
-- Adding custom NVIDIA setup for n01
-- VTK_VERSION: 9.3.1
[100%] Built target main
time ./build_n01/main --input Bucket.vtu -t 24  
ℹ️  Dataset: Bucket.vtu
ℹ️  CPU threads: 24
ℹ️  Export: 
ℹ️  Bit flag [1: TE]: false
ℹ️  Bit flag [2: EV]: false
ℹ️  Bit flag [3: ET]: false
ℹ️  Bit flag [4: TF]: false
ℹ️  Bit flag [5: FV]: false
ℹ️  Bit flag [6: FE]: false
ℹ️  Bit flag [7: FT]: false
ℹ️  Bit flag [8: EF]: false
ℹ️  Bit flag [9: VT]: false
ℹ️  Bit flag [10: TT]: false
ℹ️  Bit flag [11: FF]: false
ℹ️  Bit flag [12: EE]: false
ℹ️  Bit flag [13: VV]: true
⏳ Argument parsing: 0.000199
📍 Parsing vtu file: Bucket.vtu
🆗 Dataset loaded with 84156 tetrahedra and 19917 vertices
Has 1 arrays
	Array 0 is named Result
⏳ TV from VTK: 0.119694
📍 Building edges...
⏳ VE [CPU]: 0.036659
🆗 Built 114520 edges.
📍 Building faces...
⏳ VF [CPU]: 0.032882
🆗 Built 178760 faces.
📍 Using CPU to compute VV
ℹ️  Longest vertex adjacency: 28
⏳ VV [CPU]: 0.045196
📍 Using GPU to compute VV
ℹ️  Approximated max VV adjacency: 153
ℹ️  Kernel launch configuration is 329 grid blocks with 1024 threads per block
ℹ️  The mesh has 84156 cells and 19917 vertices
ℹ️  Tids >= 336624 should auto-exit (272)
⏳ GPU kernel duration: 0.008258
⏳ GPU Device->Host transfer: 0.003743
⏳ GPU Device->Host translation: 0.008375
⏳ VV [GPU]: 0.288844
Attribute sum: 1.45217e+06
⏳ VTK test [CPU]: 0.000050
❗ Timer[Main] Open interval 7 cannot report time until it is closed by another tick()

real	0m1.052s
user	0m0.223s
sys	0m0.331s
```

## Current State of the Repository:

This is a pre-production, in-development academic research project.

It is feature incomplete.

Features that are complete may not be sufficiently robust for production usage.

Currently, the project is developed by one developer; meaningful contributions
are not expected from random passersby and could require significant effort to
accept. As such, please email me if you are interested in contributing PRIOR to
beginning your own work so we may better coordinate efforts.

