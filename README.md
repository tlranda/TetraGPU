# Setup and Install

Requirements
* CMake >= 3.12
* [VTK == 9.3.1](htts://gitlab.kitware.com/vtk/vtk/-/tree/v9.3.1)

Build
* Define the environment variable `VTK_DIR` such that it points to the built/installed VTK base
    * If building from source, this path may resemble: `.../VTK/VTK-9.3.1/build`
    * ie: `$ VTK_DIR=<path/to/VTK/build> cmake -B build`
* Run the makefile that CMake generates for you
    * `$ make`

Dataset
* Supply a VTK-compatible tetrahedral XML unstructured grid file (.vtu)
* Alternatively, with Python3's [meshio, numpy and tqdm] dependencies, you can
run `python3 make_mesh.py --help` to see how to generate a simple mesh

Run
* Generally: `$ ./bin/main --input <path/to/vtu_file.vtu>`
* If you want to utilize my testing framework, use `$ ./test.sh <each .vtu you want to test>`
* For debugging, use `$ ./debug.sh`

