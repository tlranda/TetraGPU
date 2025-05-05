#include "cpu_extraction.h"

/*
 * Relationship extracting functions using CPU only.
 * For functions named "make_X_and_Y", we're intentionally defining two
 * relationships at the same time (usually because one defines indexing IDs
 * and the other is related to TV, so we may as well map both together at the
 * same time).
 *
 * The vtkIdDType return values inform the caller of how many unique IDs were
 * minted during relationship creation (ie: make_TE_and_VE informs the caller
 * of the number of unique edges via its return value).
 *
 * The "elective_make_X" functions create the unique pointer WITHIN the
 * function's stack, so you need to receive this memory in the caller's frame
 * or it will be deleted. These functions do not need to be called in final
 * versions of the program as the GPU should take over their creation, but they
 * are provided as CPU-sane versions of the relationship definition to help
 * check the validity of GPU results.
 *
 * These functions do not make any particular considerations about the mesh for
 * subsequent GPU performance, but if your TV relationship is set up such that
 * minimal vertex ID difference means nearby vertices, you're likely setting
 * the GPU up to get as much benefit from the ID/mapping and memory access
 * pattern as we would reasonably be able to guarantee.
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
                                          const vtkIdType n_edges
                                         ) {
    std::unique_ptr<EV_Data> edgeList = std::make_unique<EV_Data>(n_edges);
    //#pragma omp parallel for num_threads(args.threadNumber)
    for(vtkIdType i = 0; i < n_points; ++i) {
        for(const EdgeData &data : edgeTable[i]) {
            (*edgeList)[data.id] = {i, data.highVert};
        }
    }
    return edgeList;
}

std::unique_ptr<ET_Data> elective_make_ET(const TE_Data & cellEdgeList,
                                          const vtkIdType n_edges) {
    std::unique_ptr<ET_Data> edgeStars = std::make_unique<ET_Data>(n_edges);
    //#pragma omp parallel for num_threads(args.threadNumber)
    for(unsigned long long i = 0; i < cellEdgeList.size(); ++i) { // for each tetrahedron
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

// FV = VF'
std::unique_ptr<FV_Data> elective_make_FV(const VF_Data & VF,
                                          const vtkIdType n_faces) {
    std::unique_ptr<FV_Data> vertexList = std::make_unique<FV_Data>(n_faces);
    // Vector (low-vertex-id) of vectors (faces)
    vtkIdType i = 0;
    // Each iteration at this level is a new low-face-id (remarked as "id")
    // Middle- and high- vertex are unchanged, but should be located in the
    // output vector at the face ID's location
    std::for_each(VF.begin(), VF.end(),
            [&](const std::vector<FaceData>& fdVec) {
                for (const FaceData& face : fdVec) {
                    (*vertexList)[face.id] = FaceData(face.middleVert,
                                                      face.highVert,
                                                      i);
                }
                i++;
            });
    return vertexList;
}

std::unique_ptr<FE_Data> elective_make_FE(const VF_Data & VF,
                                          const VE_Data & VE,
                                          const vtkIdType n_points,
                                          const vtkIdType n_edges,
                                          const vtkIdType n_faces,
                                          const arguments args) {
    std::unique_ptr<FE_Data> edgeList = std::make_unique<FE_Data>(n_faces);
    // Implementation needed
    std::cerr << "NOT IMPLEMENTED YET!" << std::endl;
    return edgeList;
}

std::unique_ptr<VV_Data> elective_make_VV(const TV_Data & TV,
                                          const vtkIdType n_points) {
    std::unique_ptr<VV_Data> adjacencies = std::make_unique<VV_Data>(n_points);
    // Vertices are adjacent to all other vertices in a cell they appear in
    // This is a once-over pass, just don't double-enter the data (common edge
    // or common face would lead to duplicates)
    std::for_each(TV.begin(), TV.end(),
            [&](const std::array<vtkIdType,nbVertsInCell>& cell) {
                // One-loop to make adjacencies for each vertex in the cell
                for (const vtkIdType src_vertex : cell) {
                    // Double-loop to get adjacent to src vertex within THIS cell
                    for (const vtkIdType v : cell) {
                        if (v == src_vertex) continue;
                        auto index = std::find(begin((*adjacencies)[src_vertex]),
                                               end((*adjacencies)[src_vertex]),
                                               v);
                        if (index == std::end((*adjacencies)[src_vertex])) {
                            // New adjacency
                            (*adjacencies)[src_vertex].emplace_back(v);
                        }
                        // else old adjacency from another cell; skip
                    }
                }
            });
    // Possible global information: longest adjacency?
    unsigned long long longest_adjacency = 0;
    for (const std::vector<vtkIdType>& list : (*adjacencies)) {
        if (list.size() > longest_adjacency) longest_adjacency = list.size();
    }
    std::cerr << INFO_EMOJI << "Longest vertex adjacency: " << longest_adjacency
              << std::endl;
    return adjacencies;
}

std::unique_ptr<VT_Data> elective_make_VT(const TV_Data & tv) {
    std::unique_ptr<VV_Data> vt = std::make_unique<VT_Data>(tv.nPoints);
    // Accumulate inverse mapping with unique and sorted TIDs
    // VTK load should ensure this processing IS sorted and evidently it will
    // be unique as well (no known counterexamples yet)
    vtkIdType tid = 0;
    std::for_each(tv.begin(), tv.end(),
        [&](const std::array<vtkIdType,nbVertsInCell> cell_contents) {
        std::for_each(cell_contents.begin(), cell_contents.end(),
                [&](const vtkIdType vertex) {
                    // Uniqueness check
                    //const auto pos = std::find((*vt)[vertex].begin(), (*vt)[vertex].end(), tid);
                    //if (pos == (*vt)[vertex].end()) {
                    (*vt)[vertex].emplace_back(tid);
                    //}
                });
        tid++;
        });
    // If sorting fails / we want to be paranoid, you would:
    /*
    std::for_each(vt->begin(), vt->end(),
        [&](std::vector<vtkIdType> tids) {
            std::sort(tids.begin(), tids.end());
        });
    // This code helps to debug a problem vs solution with(out) the sort above
    // Uncomment it to read the values from VT, grep for "^Vertex" and you
    // will have an easier time comparing before/after.
    for (vtkIdType i = 0; i < tv.nPoints; i++) {
        std::vector<vtkIdType> & tetra_list = (*vt)[i];
        std::cout << "Vertex " << i << ", Tetra: ";
        bool first = true;
        for (const vtkIdType tetra : tetra_list) {
            if (first) {
                first = false;
            }
            else {
                std::cout << ", ";
            }
            std::cout << tetra;
        }
        std::cout << std::endl;
    }
    */
    return vt;
}
