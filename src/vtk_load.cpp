#include "vtk_load.h"

/* This is largely adapted from Guoxi's demo he kindly provided on using VTK.
   Our only goal is to get the .vtu mesh into memory and format it as a TV
   array in a manner that I like to use in all subsequent parts of the program.
   It may or may not be optimal and may or may not check every error correctly,
   but we do try to check for errors as able to at least flag the reader that
   things might not be OK.

   For now, I'm only outputting error message text and we just proceed into
   what is likely undefined behavior (crash expected) very soon when the
   processing here fails. In my experience, VTK itself likely hits this
   behavior as I attempt to construct TV, which is fine. We weren't going to
   go much further anyways and the trace/my message are usually sufficient
   warning.
*/

int checkCellTypes(vtkPointSet *object) {
    /*
     * Check a PointSet to ensure that it meets the minimum type requirements
     * for this work: Cells must be homogeneous type and 3D simplices.
     *
     * Returns 0 when no errors, else returns a negative error value
     */
    auto cellTypes = vtkSmartPointer<vtkCellTypes>::New();
    object->GetCellTypes(cellTypes);

    size_t nTypes = cellTypes->GetNumberOfTypes();

    if(nTypes == 0) return 0; // no error on an empty set
    if(nTypes > 1) return -1; // if cells are not homogeneous

    // nTypes == 1 by tautology, (size_t is unsigned)
    const auto &cellType = cellTypes->GetCellType(0);
    // if cells are not simplices
    if(cellType != VTK_VERTEX &&
       cellType != VTK_LINE &&
       cellType != VTK_TRIANGLE &&
       cellType != VTK_TETRA) {
        return -2;
    }

    return 0;
}

std::unique_ptr<TV_Data> get_TV_from_VTK(const arguments args) {
    // VTK loads the file data
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
        vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(args.fileName.c_str());
    reader->Update();
    vtkUnstructuredGrid *unstructuredGrid = reader->GetOutput();

    // Points
    vtkPoints *points = unstructuredGrid->GetPoints();
    if (!points) {
        std::cerr << WARN_EMOJI << "DataSet has uninitialized `vtkPoints`." << std::endl;
    }
    int pointDataType = points->GetDataType();
    if (pointDataType != VTK_FLOAT && pointDataType != VTK_DOUBLE) {
        std::cerr << WARN_EMOJI << "Unable to initialize triangulation for point precision "
                     "other than 'float' or 'double'." << std::endl;
    }

    // get information from the input point set
    vtkIdType nPoints = points->GetNumberOfPoints();
    //void *pointDataArray = points->GetVoidPointer(0); // this gets the base pointer for the array of XYZ coordinate values

    // check if cell types are simplices
    int cellTypeStatus = checkCellTypes(unstructuredGrid);
    if (cellTypeStatus == -1) {
        std::cout << WARN_EMOJI << "Inhomogeneous cell dimensions detected." << std::endl;
    }
    else if (cellTypeStatus == -2) {
        std::cout << WARN_EMOJI << "Cells are not simplices." << std::endl << "Consider "
                     "using `vtkTetrahedralize` in pre-processing." <<
                     std::endl;
    }

    // Cells
    vtkCellArray *cells = unstructuredGrid->GetCells();
    if (!cells) {
        std::cerr << WARN_EMOJI << "DataSet has uninitialized `vtkCellArray`." << std::endl;
    }
    vtkIdType nCells = cells->GetNumberOfCells();
    if (nCells < 0) {
        std::cerr << WARN_EMOJI << "No cells detected in the dataset." << std::endl;
    }
    else {
        std::cout << OK_EMOJI << "Dataset loaded with " << nCells
                  << " tetrahedra and " << nPoints << " vertices" << std::endl;
    }

    // connectivity array stores vertex indices of each cell (e.g., tetrahedron)
    // it can be viewed as the TV relation
    //
    // vtkIdType is the largest useful integer type available to the system,
    // so it's usually a long long int, but may downscale to long int or int
    // You can check the VTK defined symbols VTK_ID_TYPE_IMPL,
    // VTK_SIZEOF_ID_TYPE, VTK_ID_MIN, VTK_ID_MAX, VTK_ID_TYPE_PRId to confirm
    // (Based on <VTK_INSTALL>/Common/Core/vtkType.h:{295-321})
    vtkIdType * connectivity = static_cast<vtkIdType *>(cells->GetConnectivityArray()->GetVoidPointer(0));
    // offsets array tells the starting index of a cell in the connectivity array
    vtkIdType * offsets = static_cast<vtkIdType *>(cells->GetOffsetsArray()->GetVoidPointer(0));
    // Theoretically: *(connectivity[offsets[CELL_ID]])@4 = {v0,v1,v2,v3}
    // If VTK data is constructed orderly, which I observed, this means offsets
    // are always 4*index, ie offsets[0] = 0, offsets[1] = 4, offsets[4] = 16

    // Transfer into simpler data structure with unique ownership
    // We preserve the locality provided by the input, which means we assume
    // that adjacent cellIDs are close on the mesh and that their vertex IDs
    // are ordered to promote spatial locality between neighbor cells
    std::unique_ptr<TV_Data> data = std::make_unique<TV_Data>(nPoints, nCells);
    //#pragma omp parallel for num_threads(args.threadNumber)
    for (vtkIdType cellIndex = 0; cellIndex < nCells; cellIndex++) {
        std::array<vtkIdType,4> cell_vertices{
            connectivity[offsets[cellIndex]],
            connectivity[offsets[cellIndex]+1],
            connectivity[offsets[cellIndex]+2],
            connectivity[offsets[cellIndex]+3]};
        // YOU HAVE TO SORT HERE OR ELSE SOME DATASETS WILL BREAK INVARIANTS
        // HELD BY SUBSEQUENT CODE
        std::sort(cell_vertices.begin(), cell_vertices.end());
        // Because std::arrays are stack-allocated OVERRIDE the memory do not
        // replace it with memory from this frame
        for (int vertexIndex = 0; vertexIndex < 4; vertexIndex++) {
            data->cells[cellIndex][vertexIndex] = cell_vertices[vertexIndex];
        }
    }

    // Retrieve vertex attributes
    vtkPointData* pd = unstructuredGrid->GetPointData();
    if (!pd) {
        std::cerr << EXCLAIM_EMOJI << "Unable to retrieve point data from the dataset" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Has " << pd->GetNumberOfArrays() << " arrays" << std::endl;
    for (int i = 0; i < pd->GetNumberOfArrays(); i++) {
        std::cout << "\tArray " << i << " is named " << (pd->GetArrayName(i) ? pd->GetArrayName(i) : "NULL (not specified)") << std::endl;
    }
    vtkDataArray* vertexAttributes = pd->GetArray(0);
    if (!vertexAttributes) {
        std::cerr << EXCLAIM_EMOJI << "No vertex attributes found in the dataset" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<double> vertexAttributeValues(nPoints);
    for (vtkIdType i = 0; i < nPoints; i++) vertexAttributeValues[i] = vertexAttributes->GetTuple1(i);
    data->vertexAttributes = std::move(vertexAttributeValues);

    return data;
}

