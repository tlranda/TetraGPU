#include "cuda_extraction.h"

// vector of array of vertices in an edge
                                     // vector of array of vertices in a tetra
std::unique_ptr<EV_Data> make_EV_GPU(const TV_Data & tv_relationship,
                                     // vector of vector of EdgeData
                                     const VE_Data & edgeTable,
                                     const vtkIdType n_edges,
                                     const arguments args) {
    std::unique_ptr<EV_Data> edgeList = std::make_unique<EV_Data>(n_edges);
    /*
    // CPU Implementation
    #pragma omp parallel for num_threads(args.threadNumber)
    for(vtkIdType i = 0; i < tv_relationship.nPoints; ++i) {
        for(const EdgeData &data : edgeTable[i]) {
            (*edgeList)[data.id] = {i, data.highVert};
        }
    }
    */
    return edgeList;
}

