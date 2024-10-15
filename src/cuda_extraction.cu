#include "cuda_extraction.h"

void make_TV_for_GPU(vtkIdType * device_tv,
                           // vector of array of vertices in a tetra
                     const TV_Data & tv_relationship) {
    // Size determination
    size_t tv_flat_size = sizeof(vtkIdType) * tv_relationship.nCells * nbVertsInCell;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)&device_tv, tv_flat_size));
    vtkIdType * host_flat_tv = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_flat_tv, tv_flat_size));

    // Set contiguous data in host memory
    vtkIdType index = 0;
    for (const auto & VertList : tv_relationship)
        for (const vtkIdType vertex : VertList)
            host_flat_tv[index++] = vertex;

    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before copy completes
    CUDA_WARN(cudaMemcpy(device_tv, host_flat_tv,
                         tv_flat_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaFreeHost(host_flat_tv));
}

void make_VE_for_GPU(vtkIdType * device_ve,
                     vtkIdType * device_ve_offset,
                           // vector of vectors of edge IDs
                     const VE_Data & ve_relationship,
                     const vtkIdType n_edges,
                     const vtkIdType n_verts) {
    // Size determinations
    size_t ve_flat_size = sizeof(vtkIdType) * n_edges * nbVertsInEdge;
    size_t ve_offset_size = sizeof(vtkIdType) * n_verts;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)&device_ve, ve_flat_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_ve_offset, ve_offset_size));
    vtkIdType * host_ve_flat = nullptr,
              * host_ve_offset = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_ve_flat,
                               ve_flat_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_ve_offset, ve_offset_size));

    // Set contiguous data in host memory
    vtkIdType vertex_id = 0,
              vertex_offset = 0;
    for (const auto & EdgeList : ve_relationship) {
        host_ve_offset[vertex_id++] = vertex_offset;
        for (const EdgeData & edge : EdgeList) {
            host_ve_flat[vertex_offset++] = edge.id;
            host_ve_flat[vertex_offset++] = edge.highVert;
        }
    }

    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before copy completes
    CUDA_WARN(cudaMemcpy(device_ve, host_ve_flat,
                         ve_flat_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(device_ve_offset, host_ve_offset,
                         ve_offset_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaFreeHost(host_ve_flat));
    CUDA_WARN(cudaFreeHost(host_ve_offset));
}

// vector of array of vertices in an edge
std::unique_ptr<EV_Data> make_EV_GPU(const TV_Data & tv_relationship,
                                     // vector of vector of EdgeData
                                     const VE_Data & edgeTable,
                                     const vtkIdType n_edges,
                                     const arguments args) {
    std::unique_ptr<EV_Data> edgeList = std::make_unique<EV_Data>(n_edges);

    // Marshall data to GPU
    vtkIdType * tv_device = nullptr;
    make_TV_for_GPU(tv_device, tv_relationship);
    vtkIdType * ve_device = nullptr,
              * ve_offsets = nullptr;
    make_VE_for_GPU(ve_device,
                    ve_offsets,
                    edgeTable,
                    n_edges,
                    tv_relationship.nPoints);
    // Compute the relationship
    // Copy back to host and set in edgeList
    // Free device memory
    if (tv_device != nullptr)
        CUDA_WARN(cudaFree(tv_device));
    if (ve_device != nullptr)
        CUDA_WARN(cudaFree(ve_device));
    if (ve_offsets != nullptr)
        CUDA_WARN(cudaFree(ve_offsets));
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

