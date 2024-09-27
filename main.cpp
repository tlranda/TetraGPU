#include <iostream> // std::c{err,out}, std::endl
#include <cstdlib> // atoi
#include <getopt.h> // argument parsing getopt_long()
#include <unistd.h>

// VTK library requirements
#include <vtkCellTypes.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

// Argument values to be stored here, with defaults if they are optional
typedef struct argstruct {
    std::string fileName;
    int threadNumber = 1;
} arguments;
// Getopt option declarations
const char * optionstring = "hi:t:";
static struct option long_options[] = {
    {"help", no_argument, 0, 'h'},
    {"input", required_argument, 0, 'i'},
    {"threads", required_argument, 0, 't'},
    {0,0,0,0}
};

void parse(int argc, char *argv[], arguments& args) {
    /*
     * Command-line parsing
     *
     * Output as many helpful error-state messages as possible if invalid.
     * If valid, set values in the provided struct appropriately
     */
    int c, bad_args = 0;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " -i <input.vtu> [options]" <<
                     std::endl << "Missing required input .vtu file!" <<
                     std::endl;
        bad_args += 1;
    }
    // Disable getopt's automatic error messages so we can catch it via '?'
    opterr = 0;

    while (1) {
        int option_index = 0;
        c = getopt_long(argc, argv, optionstring, long_options, &option_index);
        if (c == -1) break;
        switch (c) {
            case 0:
                // I need to remind myself of the actual significance of this case
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " -i <input.vtu> " <<
                             "[options]" << std::endl;
                std::cout << "\t-h | --help\n\t\t" <<
                             "Print this help message and exit" << std::endl;
                std::cout << "\t-i <input.vtu> | --input <input.vtu>\n\t\t" <<
                             "Tetrahedral mesh input (.vtu only)" << std::endl;
                std::cout << "\t-t <threads> | --threads <threads>\n\t\t" <<
                             "CPU thread limit for parallelism" << std::endl;
                exit(EXIT_SUCCESS);
            case 'i':
                args.fileName = std::string(optarg);
                break;
            case 't':
                args.threadNumber = atoi(optarg);
                if (args.threadNumber == 0) {
                    // Indicates 0 or an error in processing, fortunately
                    // 0 is an invalid value for us as well in this context.
                    std::cerr << "Thread argument must be integer >= 1" <<
                                 std::endl;
                    bad_args += 1;
                }
                break;
            case '?':
                std::cerr << "Unrecognized argument: " << argv[optind-1] <<
                             std::endl;
                bad_args += 1;
                break;
        }
    }

    // Filename must be given
    if (args.fileName.empty()) {
        std::cerr << "Must supply an input filename via -i | --input" <<
                     std::endl;
        bad_args += 1;
    }

    if (bad_args != 0) exit(EXIT_FAILURE);
}

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

/*
 * Data types for relationships and retrieved data
 * TV relationship: TV_Data
 * Cell: vtkIDType (from VTK, integer-type)
 * Vertex: vtkIDType (from VTK, integer-type)
 * Edge: EdgeData (high vertex & edge ID (arbitrarily enumerated))
 * TE relationship: TE_Data (map cell ID to its 6 edge IDs)
 * EV relationship: EV_Data (map edge ID to its 2 vertex IDs)
 * ET relationship: ET_Data (map edge ID to an arbitrary number of cell IDs)
 * VE relationship: VE_Data (map low vertex ID to edge data including other vertex)
 */
/*
    From here we can dynamically compute:
    VV = TV' * TV
    VE (precomputed)
            VF = --not defined--
    VT = TV'
    EV (precomputed)
    EE = EV * VE
            EF = --not defined--
    ET (precomputed)
            FV = --not defined--
            FE = --not defined--
            FF = --not defined--
            FT = --not defined--
    TV (defined from storage)
    TE (precomputed)
            TF = --not defined--
    TT = TV * TV'
*/
typedef struct TV_from_VTK {
    vtkIdType nPoints, nCells;
    // For some reason, vector of array is experimentally 20x faster than unique_ptr?
    //std::unique_ptr<vtkIdType[]> cells;
    std::vector<std::array<int,4>> cells;

    TV_from_VTK(vtkIdType _points, vtkIdType _cells) {
        nPoints = _points;
        nCells = _cells;
        //cells = std::make_unique<vtkIdType[]>(_points * _cells);
        cells.resize(nCells);
    }
} TV_Data;
struct EdgeData {
    vtkIdType highVert = 0, // edge's higher vertex id
              id = 0;       // edge's id
    EdgeData(vtkIdType hv, vtkIdType i) : highVert{hv}, id{i} {}
};
typedef std::vector<std::array<vtkIdType,6>> TE_Data;
typedef std::vector<std::array<vtkIdType,2>> EV_Data;
typedef std::vector<std::vector<vtkIdType>> ET_Data;
typedef std::vector<std::vector<EdgeData>> VE_Data;

/*
 * Relationship extracting functions
 */

std::unique_ptr<TV_Data> get_TV_from_VTK(const arguments args) {
    // VTK loads the file data
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
        vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(args.fileName.c_str());
    reader->Update();
    vtkUnstructuredGrid *unstructuredGrid = reader->GetOutput();

    // Points
    vtkPoints *points = unstructuredGrid->GetPoints();
    if(!points) {
        std::cerr << "DataSet has uninitialized `vtkPoints`." << std::endl;
    }
    int pointDataType = points->GetDataType();
    if(pointDataType != VTK_FLOAT && pointDataType != VTK_DOUBLE) {
        std::cerr << "Unable to initialize triangulation for point precision "
                     "other than 'float' or 'double'." << std::endl;
    }

    // get information from the input point set
    vtkIdType nPoints = points->GetNumberOfPoints();
    //void *pointDataArray = points->GetVoidPointer(0); // this gets the base pointer for the array of XYZ coordinate values

    // check if cell types are simplices
    int cellTypeStatus = checkCellTypes(unstructuredGrid);
    if(cellTypeStatus == -1) {
        std::cout << "Inhomogeneous cell dimensions detected." << std::endl;
    }
    else if(cellTypeStatus == -2) {
        std::cout << "Cells are not simplices." << std::endl << "Consider "
                     "using `vtkTetrahedralize` in pre-processing." <<
                     std::endl;
    }

    // Cells
    vtkCellArray *cells = unstructuredGrid->GetCells();
    if(!cells) {
        std::cerr << "DataSet has uninitialized `vtkCellArray`." << std::endl;
    }
    vtkIdType nCells = cells->GetNumberOfCells();
    if(nCells < 0) {
        std::cerr << "No cells detected in the dataset." << std::endl;
    }
    else {
        std::cout << "Dataset loaded with " << nCells << " tetrahedra and " <<
                     nPoints << " vertices" << std::endl;
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
    std::unique_ptr<TV_Data> data = std::make_unique<TV_Data>(nPoints, nCells);
    #pragma omp parallel for num_threads(args.threadNumber)
    for (vtkIdType cellIndex = 0; cellIndex < nCells; cellIndex++) {
        for (int vertexIndex = 0; vertexIndex < 4; vertexIndex++) {
            data->cells[cellIndex][vertexIndex] = connectivity[offsets[cellIndex] + vertexIndex];
            //data->cells[(4*cellIndex)+vertexIndex] = connectivity[offsets[cellIndex] + vertexIndex];
        }
    }

    return data;
}

vtkIdType make_TE_and_VE(const TV_Data tv_relationship,
                         TE_Data cellEdgeList,
                         VE_Data edgeTable
                         ) {
    const int nbVertsInCell = 4; // tetrahedron; verifiable via VTK offsets[1]-offsets[0]
    vtkIdType edgeCount = 0;
    // SIMULTANEOUSLY define TE and VE
    for(vtkIdType cid = 0; cid < tv_relationship.nCells; cid++) {
        // id of edge in cell
        vtkIdType ecid = 0;
        // TE case: {0-1}, {0-2}, {0-3}, {1-2}, {1-3}, {2-3}
        for(int j = 0; j <= nbVertsInCell - 2; j++) {
            for(int k = j + 1; k <= nbVertsInCell - 1; k++) {
                // edge processing
                // enforce lower vertex id as first element
                vtkIdType v0 = tv_relationship.cells[cid][j], //[(4*cid)+j],
                          v1 = tv_relationship.cells[cid][k]; //[(4*cid)+k];
                if(v0 > v1) std::swap(v0, v1);
                // We will store edges incident to lower vertex here
                std::vector<EdgeData> &vec = edgeTable[v0];
                // Scan all edges currently stored incident to this basis vertex
                const auto pos = std::find_if(vec.begin(), vec.end(),
                                 [&](const EdgeData &a) { return a.highVert == v1; });
                if(pos == vec.end()) {
                    // not found in edgeTable: new edge for VE
                    vec.emplace_back(EdgeData(v1, edgeCount));
                    cellEdgeList[cid][ecid] = edgeCount;
                    edgeCount++;
                }
                // found an existing edge, but mark it for TE
                else cellEdgeList[cid][ecid] = pos->id;
                ecid++;
            }
        }
    }
    return edgeCount;
}
std::unique_ptr<EV_Data> make_EV(const TV_Data tv_relationship,
                                 const VE_Data edgeTable,
                                 const vtkIdType n_edges,
                                 const arguments args
                                 ) {
    std::unique_ptr<EV_Data> edgeList = std::make_unique<EV_Data>(n_edges);
    #pragma omp parallel for num_threads(args.threadNumber)
    for(vtkIdType i = 0; i < tv_relationship.nPoints; ++i) {
        for(const EdgeData &data : edgeTable[i]) {
            (*edgeList)[data.id] = {i, data.highVert};
        }
    }
    return edgeList;
}

std::unique_ptr<ET_Data> make_ET(const TE_Data cellEdgeList,
                                 const vtkIdType n_edges,
                                 const arguments args
                                 ) {
    std::unique_ptr<ET_Data> edgeStars = std::make_unique<ET_Data>(n_edges);
    #pragma omp parallel for num_threads(args.threadNumber)
    for(vtkIdType i = 0; i < cellEdgeList.size(); ++i) { // for each tetrahedron
        // std::array<vtkIdType,6> list of edges for tetra
        for(const vtkIdType eid : cellEdgeList[i]) { // for each edge
            (*edgeStars)[eid].emplace_back(i); // edge :: tetra
        }
    }
    return edgeStars;
}

int main(int argc, char *argv[]) {
    arguments args;
    parse(argc, argv, args);

    std::cout << "Parsing vtu file: " << args.fileName << std::endl;
    // Should utilize VTK API and then de-allocate all of its heap
    std::unique_ptr<TV_Data> tv_relationship = get_TV_from_VTK(args);

    // Example of building edge list from the cell (tetra) array
    // This should be the VE relation, but we can simultaneously determine TE since we are creating edges arbitrarily based on TV
    // Adapted from TTK Explicit Triangulation
    std::cout << "Building edges..." << std::endl;
    std::unique_ptr<TE_Data> cellEdgeList = std::make_unique<TE_Data>(tv_relationship->nCells); // TE
    std::unique_ptr<VE_Data> edgeTable = std::make_unique<VE_Data>(tv_relationship->nPoints); // VE
    vtkIdType edgeCount = make_TE_and_VE(*tv_relationship,
                                         *(cellEdgeList.get()),
                                         *(edgeTable.get()));

    // allocate & fill edgeList in parallel (EV)
    std::unique_ptr<EV_Data> EV = make_EV(*tv_relationship,
                                          *(edgeTable.get()),
                                          edgeCount,
                                          args);

    // we can also get edgeStars from cellEdgeList (ET)
    std::unique_ptr<ET_Data> ET = make_ET(*(cellEdgeList.get()),
                                          edgeCount,
                                          args);

    // before the extraction, we have TV relation
    // after the extraction, we now have TE, VE, EV and ET relations
    std::cout << "Built " << edgeCount << " edges." << std::endl;

    return 0;
}

