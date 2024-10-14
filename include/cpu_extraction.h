#ifndef TETRA_CPU_RELATIONS
#define TETRA_CPU_RELATIONS

#include <algorithm> // std::sort
#include "vtk_load.h" // TV_Data type

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
// Required precomputes for IDs with inherited spatial locality from TV relationship
typedef std::vector<std::array<vtkIdType,nbEdgesInCell>> TE_Data;
typedef std::vector<std::vector<EdgeData>> VE_Data;
typedef std::vector<std::array<vtkIdType,nbFacesInCell>> TF_Data;
typedef std::vector<std::vector<FaceData>> VF_Data;

// Elective precomputes
typedef std::vector<std::array<vtkIdType,nbVertsInEdge>> EV_Data;
typedef std::vector<std::vector<vtkIdType>> ET_Data;

vtkIdType make_TE_and_VE(const TV_Data & tv_relationship,
                         TE_Data & cellEdgeList,
                         VE_Data & edgeTable
                         );
std::unique_ptr<EV_Data> elective_make_EV(const TV_Data & tv_relationship,
                                          const VE_Data & edgeTable,
                                          const vtkIdType n_edges,
                                          const arguments args
                                         );
std::unique_ptr<ET_Data> elective_make_ET(const TE_Data & cellEdgeList,
                                          const vtkIdType n_edges,
                                          const arguments args
                                         );
std::unique_ptr<ET_Data> elective_make_ET(const TE_Data & cellEdgeList,
                                          const vtkIdType n_edges,
                                          const arguments args
                                         );
vtkIdType make_TF_and_VF(const TV_Data & tv_relationship,
                         TF_Data & cellFaceList,
                         VF_Data & faceTable
                        );
#endif

