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

std::shared_ptr<TV_Data> get_TV_from_VTK(const runtime_arguments args) {
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

    std::shared_ptr<TV_Data> data = std::make_shared<TV_Data>(nPoints, nCells);
    // TODO: Embarassingly parallel but not enough for naive pragma omp parallel
    for (vtkIdType cellIndex = 0; cellIndex < nCells; cellIndex++) {
        std::array<vtkIdType,nbVertsInCell> cell_vertices{
            connectivity[offsets[cellIndex]+0],
            connectivity[offsets[cellIndex]+1],
            connectivity[offsets[cellIndex]+2],
            connectivity[offsets[cellIndex]+3]};
        // Because std::arrays are stack-allocated OVERRIDE the memory do not
        // replace it with memory from this frame
        for (int vertexIndex = 0; vertexIndex < nbVertsInCell; vertexIndex++) {
            data->cells[(cellIndex*nbVertsInCell)+vertexIndex] = cell_vertices[vertexIndex];
        }
    }

    // Retrieve vertex attributes
    vtkPointData* pd = unstructuredGrid->GetPointData();
    if (!pd) {
        std::cerr << EXCLAIM_EMOJI << "Unable to retrieve point data from the dataset" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Has " << pd->GetNumberOfArrays() << " point arrays" << std::endl;
    int use_this_array = -1;
    for (int i = 0; i < pd->GetNumberOfArrays(); i++) {
        std::cout << "\tPoint Array " << i << " is named " << (pd->GetArrayName(i) ? pd->GetArrayName(i) : "NULL (not specified)") << std::endl;
        if (use_this_array == -1 && args.arrayname != "" && pd->GetArrayName(i) == args.arrayname) {
            use_this_array = i;
            std::cout << "Found user's requested point array for vertex attributes: " << args.arrayname << std::endl;
        }
    }
    if (use_this_array == -1) {
        use_this_array = 0;
        std::cout << "Auto select 0th point array as vertex attributes: " << pd->GetArrayName(0) << std::endl;
    }
    vtkDataArray* vertexAttributes = pd->GetArray(use_this_array);
    if (!vertexAttributes) {
        std::cerr << EXCLAIM_EMOJI << "No vertex attributes found in the dataset" << std::endl;
        exit(EXIT_FAILURE);
    }
    double * copyVertexAttributes = new double[nPoints];
    for (vtkIdType i = 0; i < nPoints; i++) copyVertexAttributes[i] = vertexAttributes->GetTuple1(i);
    data->vertexAttributes = copyVertexAttributes;

    // Retrieve partitioning IDs
    Timer vtkTimer(false, "VTK Preprocessing");
    if (args.partitioningname == "") {
        // Default everything into a single partition
        int * meshPartitionIDs = new int[nPoints]();
        data->partitionIDs = meshPartitionIDs;
        int * mesh_n_per_partition = new int[1];
        mesh_n_per_partition[0] = nPoints;
        data->n_per_partition = mesh_n_per_partition;
    }
    else {
        // Re-use pd from scalar fetch, similar process
        use_this_array = -1;
        for (int i = 0; i < pd->GetNumberOfArrays(); i++) {
            if (args.partitioningname == pd->GetArrayName(i)) {
                use_this_array = i;
                std::cout << "Found user's requested partitioning: " << args.partitioningname << std::endl;
                break;
            }
        }
        if (use_this_array == -1) {
            std::cerr << EXCLAIM_EMOJI << "Unable to find partitioning name '" << args.partitioningname << "' in dataset" << std::endl;
            exit(EXIT_FAILURE);
        }
        vtkDataArray* partitionAttribute = pd->GetArray(use_this_array);
        int * meshPartitionIDs = new int[nPoints];
        // Determine the number of partitions and their counts
        // TODO: Parallelism SHOULD help here, but slapping omp parallel is disastrously bad for performance
        for (vtkIdType i = 0; i < nPoints; i++) {
            meshPartitionIDs[i] = partitionAttribute->GetTuple1(i)-1;
        }
        std::unordered_set<int> partition_set(meshPartitionIDs, meshPartitionIDs+nPoints);
        data->n_partitions = partition_set.size();
        int * mesh_n_per_partition = new int[data->n_partitions]();
        // TODO: Parallelism should help, but omp parallel has virtually zero impact on performance
        for (vtkIdType i = 0; i < nPoints; i++) {
            mesh_n_per_partition[meshPartitionIDs[i]]++;
        }
        std::cout << "Using " << data->n_partitions << " partitions from " << args.partitioningname << std::endl;
        data->partitionIDs = meshPartitionIDs;
        data->n_per_partition = mesh_n_per_partition;
    }

    // Retrieve cell data
    if (data->n_partitions == 1) {
        std::cout << "No cell data retrieval for cell partition data" << std::endl;
    }
    else {
        vtkCellData* cd = unstructuredGrid->GetCellData();
        if (!cd) {
            std::cerr << EXCLAIM_EMOJI << "Unable to retrieve cell data from the dataset" << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << "Has " << cd->GetNumberOfArrays() << " cell arrays" << std::endl;
        data->partitionCells = new int[data->n_partitions * nCells]();
        if (cd->GetNumberOfArrays() == 0) {
            // Do the precompute here, multithreaded
            Timer ecp(false, "External Cell Preprocess");
            //#pragma omp parallel
            for (int c = 0; c < nCells; c++) {
                for (int v = 0; v < nbVertsInCell; v++) {
                    int vertexID = data->cells[(c*nbVertsInCell)+v],
                        partitionID = data->partitionIDs[vertexID];
                    data->partitionCells[(partitionID * nCells) + c] = 1;
                    /* Extra-verbose for real-time checking
                    vtkDataArray* cellInfo = cd->GetArray(partitionID);
                    int datacheck = cellInfo->GetTuple1(c);
                    if (datacheck != 1) {
                        std::cout << "Mismatch for partition " << partitionID << " cell " << c << std::endl;
                    }
                    */
                }
            }
            /* DEBUG ONLY: Check that answer matches Python saved answer
            if (cd->GetNumberOfArrays() != 0) {
                int mismatch = 0, first_mismatch = -1, first_partition = -1;
                for (int i = 0; i < cd->GetNumberOfArrays(); i++) {
                    cellInfo = cd->GetArray(i);
                    for (int c = 0; c < data->nCells; c++) {
                        if (cellInfo->GetTuple1(c) != data->partitionCells[(i * nCells) + c]) {
                            mismatch++;
                            if (first_mismatch == -1) {
                                first_mismatch = c;
                                first_partition = i;
                            }
                        }
                    }
                }
                std::cout << "Mismatch count between disk data and instantaneous data: " << mismatch << std::endl;
                std::cout << "First mismatch @ " << first_mismatch << " in partition " << first_partition << std::endl;
            }
            */
            ecp.tick_announce();
            //exit(EXIT_SUCCESS);
        }
        else {
            int * pointer = data->partitionCells;
            for (int p = 0; p < data->n_partitions; p++) {
                char search[32] = {0};
                sprintf(search, "partition_cells_%d", p);
                std::cout << "Searching for array: " << search << std::endl;
                bool found = false;
                for (int i = 0; i < cd->GetNumberOfArrays(); i++) {
                    const char * arrayName = cd->GetArrayName(i);
                    if (p == 0) {
                        std::cout << "\tCell Array " << i << " is named " << (arrayName ? arrayName : "NULL (not specified)") << std::endl;
                    }
                    if (strcmp(search, arrayName) == 0) {
                        std::cout << "Copying cell data array " << arrayName << " for partition " << p << std::endl;
                        vtkDataArray* cellInfo = cd->GetArray(i);
                        // TODO: Parallelize
                        for (int c = 0; c < data->nCells; c++) {
                            *pointer = cellInfo->GetTuple1(c);
                            pointer++;
                        }
                        found = true;
                    }
                }
                if (!found) {
                    // Cannot use partial / missing data
                    std::cerr << WARN_EMOJI << "Failed to find partition cells (cellData) array " << search << ", falling back to slower CPU-side precompute" << std::endl;
                    delete[] data->partitionCells;
                    data->partitionCells = nullptr;
                    break;
                }
            }
        }
    }
    vtkTimer.tick_announce();

    return data;
}

