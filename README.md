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

Run
* `$ ./bin/main --input <path/to/vtu_file.vtu>`
