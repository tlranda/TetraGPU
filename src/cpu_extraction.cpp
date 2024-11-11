#include "cpu_extraction.h"

/*
 * Relationship extracting functions
 */

vtkIdType make_TE_and_VE(const TV_Data & tv_relationship,
                         TE_Data & cellEdgeList,
                         VE_Data & edgeTable
                         ) {
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
                    cellEdgeList[cid][ecid] = edgeCount++;
                }
                // found an existing edge, but mark it for TE
                else cellEdgeList[cid][ecid] = pos->id;
                ecid++;
            }
        }
    }
    return edgeCount;
}
vtkIdType make_VE(const TV_Data & tv_relationship, VE_Data & edgeTable) {
    vtkIdType edgeCount = 0;
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
                    edgeCount++;
                }
                ecid++;
            }
        }
    }
    return edgeCount;
}

// EV = VE'
std::unique_ptr<EV_Data> elective_make_EV(const VE_Data & edgeTable,
                                          const vtkIdType n_points,
                                          const vtkIdType n_edges,
                                          const arguments args
                                         ) {
    std::unique_ptr<EV_Data> edgeList = std::make_unique<EV_Data>(n_edges);
    #pragma omp parallel for num_threads(args.threadNumber)
    for(vtkIdType i = 0; i < n_points; ++i) {
        for(const EdgeData &data : edgeTable[i]) {
            (*edgeList)[data.id] = {i, data.highVert};
        }
    }
    return edgeList;
}

std::unique_ptr<ET_Data> elective_make_ET(const TE_Data & cellEdgeList,
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

vtkIdType make_TF_and_VF(const TV_Data & tv_relationship,
                         TF_Data & cellFaceList,
                         VF_Data & faceTable
                        ) {
    vtkIdType faceCount = 0;
    // SIMULTANEOUSLY define TF and VF
    for (vtkIdType cid = 0; cid < tv_relationship.nCells; cid++) {
        // Use copy constructor to permit function-local mutation
        std::array<vtkIdType,nbVertsInCell> cellVertices(tv_relationship.cells[cid]);
        std::sort(cellVertices.begin(), cellVertices.end());
        // face IDs can be given based on ascending vertex sums
        // given a SORTED list of vertex IDs, ascending order of vertex sums is:
        // v[[0,1,2]], v[[0,1,3]], v[[0,2,3]], v[[1,2,3]]
        // These sums are unique given that vertex IDs are unique (Proof by contradiction)
        //      a + x + y == x + y + z
        //      a == z                  ! contradiction

        // Based on sorting, 3 faces will use the lowest FaceID of the Cell
        // for their basis in faceTable
        for (int skip_id = 3; skip_id >= 0; skip_id--) {
            // Modified loop iteration order to match GPU processing order (no sorting for validation needed)
            // 3 : lo = 0, mid = 1, hi = 2
            // 0 : lo = 1, mid = 2, hi = 3
            // 1 : lo = 0, mid = 2, hi = 3
            // 2 : lo = 0, mid = 1, hi = 3
            int low_vertex = (skip_id > 0) ? 0 : 1,
                middle_vertex = (skip_id > 1) ? 1 : 2,
                high_vertex = (skip_id == 3) ? 2 : 3;

            std::vector<FaceData> &vec = faceTable[cellVertices[low_vertex]];
            const auto pos = std::find_if(vec.begin(), vec.end(),
                    [&](const FaceData &f) {
                        return f.middleVert == cellVertices[middle_vertex] &&
                               f.highVert == cellVertices[high_vertex];
                    });
            if (pos == vec.end()) {
                // Not found in faceTable: new face for VF
                vec.emplace_back(FaceData(cellVertices[middle_vertex],
                                          cellVertices[high_vertex],
                                          faceCount));
                cellFaceList[cid][3-skip_id] = faceCount;
                faceCount++;
            }
            // Found an existing face, still mark in TF
            else cellFaceList[cid][3-skip_id] = pos->id;
        }
    }
    return faceCount;
}

vtkIdType make_VF(const TV_Data & tv_relationship,
                  VF_Data & faceTable) {
    vtkIdType faceCount = 0;
    for (vtkIdType cid = 0; cid < tv_relationship.nCells; cid++) {
        // Use copy constructor to permit function-local mutation
        std::array<vtkIdType,nbVertsInCell> cellVertices(tv_relationship.cells[cid]);
        std::sort(cellVertices.begin(), cellVertices.end());
        // face IDs can be given based on ascending vertex sums
        // given a SORTED list of vertex IDs, ascending order of vertex sums is:
        // v[[0,1,2]], v[[0,1,3]], v[[0,2,3]], v[[1,2,3]]
        // These sums are unique given that vertex IDs are unique (Proof by contradiction)
        //      a + x + y == x + y + z
        //      a == z                  ! contradiction

        // Based on sorting, 3 faces will use the lowest FaceID of the Cell
        // for their basis in faceTable
        for (int skip_id = 3; skip_id >= 0; skip_id--) {
            // Modified loop iteration order to match GPU processing order (no sorting for validation needed)
            // 3 : lo = 0, mid = 1, hi = 2
            // 0 : lo = 1, mid = 2, hi = 3
            // 1 : lo = 0, mid = 2, hi = 3
            // 2 : lo = 0, mid = 1, hi = 3
            int low_vertex = (skip_id > 0) ? 0 : 1,
                middle_vertex = (skip_id > 1) ? 1 : 2,
                high_vertex = (skip_id == 3) ? 2 : 3;

            std::vector<FaceData> &vec = faceTable[cellVertices[low_vertex]];
            const auto pos = std::find_if(vec.begin(), vec.end(),
                    [&](const FaceData &f) {
                        return f.middleVert == cellVertices[middle_vertex] &&
                               f.highVert == cellVertices[high_vertex];
                    });
            if (pos == vec.end()) {
                // Not found in faceTable: new face for VF
                vec.emplace_back(FaceData(cellVertices[middle_vertex],
                                          cellVertices[high_vertex],
                                          faceCount));
                faceCount++;
            }
        }
    }
    return faceCount;
}

