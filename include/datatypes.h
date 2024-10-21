#ifndef TETRA_DATA_TYPES
#define TETRA_DATA_TYPES

#include <vtkCellTypes.h> // VTK Id Types etc

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
 *
 * TF relationship: TF_Data (map cell ID to its 4 face IDs)
 * VF relationship: VF_Data (map low vertex ID to face data including other vertices)
 */
/*
    From here we can dynamically compute:
    VV = TV' * TV
    VE (precomputed to define IDs)
    VF (precomputed to define IDs)
    VT = TV'
    EV (precomputed)
    EE = EV * VE
    EF = EV * VF
    ET (precomputed)
    FV = VF'
    FE = VF' * VE
    FF = TF' * TF
    FT = TF'
    TV (defined from storage)
    TE (precomputed to define IDs)
    TF (precomputed to define IDs)
    TT = TV * TV'
*/

// Tetrahedron geometry constants
const int nbFacesInCell = 4;
const int nbEdgesInCell = 6;
const int nbVertsInEdge = 2;
const int nbVertsInFace = 3;
const int nbVertsInCell = 4; // verifiable at VTK load via VTK offsets[1]-offsets[0]

struct EdgeData {
    vtkIdType highVert = 0, // edge's higher vertex id
              id = 0;       // edge's id
    EdgeData(vtkIdType hv, vtkIdType i) : highVert{hv}, id{i} {}
};
struct FaceData {
    vtkIdType middleVert = 0, // face's second vertex id
              highVert = 0,   // faces highest vertex id
              id = 0;         // face's id
    FaceData(vtkIdType mv, vtkIdType hv, vtkIdType i) : middleVert{mv}, highVert{hv}, id{i} {}
};
struct TV_Data {
    vtkIdType nPoints, nCells;
    // For some reason, vector of array is experimentally 20x faster than unique_ptr?
    //std::unique_ptr<vtkIdType[]> cells;
    std::vector<std::array<vtkIdType,nbVertsInCell>> cells;

    TV_Data(vtkIdType _points, vtkIdType _cells) {
        nPoints = _points;
        nCells = _cells;
        //cells = std::make_unique<vtkIdType[]>(_points * _cells);
        cells.resize(nCells);
    }

    std::vector<std::array<vtkIdType,nbVertsInCell>>::iterator begin() {
        return cells.begin();
    }
    std::vector<std::array<vtkIdType,nbVertsInCell>>::const_iterator begin() const {
        return cells.begin();
    }
    std::vector<std::array<vtkIdType,nbVertsInCell>>::iterator end() {
        return cells.end();
    }
    std::vector<std::array<vtkIdType,nbVertsInCell>>::const_iterator end() const {
        return cells.end();
    }
};

// Required precomputes for IDs with inherited spatial locality from TV relationship
typedef std::vector<std::array<vtkIdType,nbEdgesInCell>> TE_Data;
typedef std::vector<std::vector<EdgeData>> VE_Data;
typedef std::vector<std::array<vtkIdType,nbFacesInCell>> TF_Data;
typedef std::vector<std::vector<FaceData>> VF_Data;

// Elective precomputes
typedef std::vector<std::array<vtkIdType,nbVertsInEdge>> EV_Data;
typedef std::vector<std::vector<vtkIdType>> ET_Data;

// Argument values to be stored here, with defaults if they are optional
typedef struct argstruct {
    std::string fileName;
    int threadNumber = 1;
} arguments;

#endif
