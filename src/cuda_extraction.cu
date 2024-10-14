#include "cuda_extraction.h"

                                     // vector of array of vertices in a tetra
vtkIdType * make_TV_for_GPU(const TV_Data & tv_relationship) {
    size_t allocation_size = sizeof(vtkIdType) * tv_relationship.nCells * nbVertsInCell;
    vtkIdType * device_ptr = nullptr;
    CUDA_WARN(cudaMalloc((void**)&device_ptr, allocation_size));
    vtkIdType * host_ptr = nullptr;
    CUDA_WARN(cudaMallocHost((void**)&host_ptr, allocation_size));
    vtkIdType index = 0;
    for (const auto & VertList : tv_relationship)
        for (const vtkIdType vertex : VertList)
            host_ptr[index++] = vertex;
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before copy completes
    CUDA_WARN(cudaMemcpy(device_ptr, host_ptr, allocation_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaFreeHost(host_ptr));
    return device_ptr;
}

                                     // vector of vectors of edge IDs
vtkIdType * make_VE_for_GPU(const VE_Data & ve_relationship) {
    size_t allocation_size = sizeof(vtkIdType) * ve_relationship. * nbVertsInCell;
    vtkIdType * device_ptr = nullptr;
    CUDA_WARN(cudaMalloc((void**)&device_ptr, allocation_size));
    vtkIdType * host_ptr = nullptr;
    CUDA_WARN(cudaMallocHost((void**)&host_ptr, allocation_size));
    vtkIdType index = 0;
    for (const auto & VertList : tv_relationship)
        for (const vtkIdType vertex : VertList)
            host_ptr[index++] = vertex;
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before copy completes
    CUDA_WARN(cudaMemcpy(device_ptr, host_ptr, allocation_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaFreeHost(host_ptr));
    return device_ptr;
}

// vector of array of vertices in an edge
std::unique_ptr<EV_Data> make_EV_GPU(const TV_Data & tv_relationship,
                                     // vector of vector of EdgeData
                                     const VE_Data & edgeTable,
                                     const vtkIdType n_edges,
                                     const arguments args) {
    std::unique_ptr<EV_Data> edgeList = std::make_unique<EV_Data>(n_edges);

    // Marshall data to GPU
    vtkIdType * tv_device = make_TV_for_GPU(tv_relationship);
    vtkIdType * ve_device = make_VE_for_GPU(edgeTable);
    CUDA_WARN(cudaFree(tv_device));
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

