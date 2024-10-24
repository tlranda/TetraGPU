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

void make_VE_for_GPU(vtkIdType ** device_vertices,
                     vtkIdType ** device_edges,
                           // vector of vectors of edge IDs
                     const VE_Data & ve_relationship,
                     const vtkIdType n_verts,
                     const vtkIdType n_edges
                     ) {
    // Size determinations
    size_t vertices_size = sizeof(vtkIdType) * n_edges * nbVertsInEdge,
           // Can technically be half-sized, but duplicate for now
           edges_size = sizeof(vtkIdType) * n_edges * nbVertsInEdge;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)device_vertices, vertices_size));
    CUDA_ASSERT(cudaMalloc((void**)device_edges, edges_size));
    vtkIdType * host_vertices = nullptr,
              * host_edges = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vertices, vertices_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_edges, edges_size));

    // Set contiguous data in host memory
    for (vtkIdType vertex_id = 0, index = 0; vertex_id < n_verts; vertex_id++) {
        for (const EdgeData & edge : ve_relationship[vertex_id]) {
            // Pack low edge / ID
            host_edges[index] = edge.id;
            host_vertices[index++] = vertex_id;
            // Pack high edge / ID
            host_edges[index] = edge.id;
            host_vertices[index++] = edge.highVert;
        }
    }

    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before copy completes
    CUDA_WARN(cudaMemcpy(*device_vertices, host_vertices,
                         vertices_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(*device_edges, host_edges,
                         edges_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaFreeHost(host_vertices));
    CUDA_WARN(cudaFreeHost(host_edges));
}

__global__ void EV_kernel(vtkIdType * __restrict__ vertices,
                          vtkIdType * __restrict__ edges,
                          const vtkIdType n_edges,
                          vtkIdType * __restrict__ ev) {
    vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x,
              hi_vert = (tid % 2);
    if (tid >= (n_edges * nbVertsInEdge)) return;
    ev[(edges[tid] * nbVertsInEdge) + hi_vert] = vertices[tid];
}

// vector of array of vertices in an edge
                                     // vector of vector of EdgeData
std::unique_ptr<EV_Data> make_EV_GPU(const VE_Data & edgeTable,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const arguments args) {
    std::unique_ptr<EV_Data> edgeList = std::make_unique<EV_Data>();
    edgeList->reserve(n_edges);

    // Marshall data to GPU
    vtkIdType * vertices_device = nullptr,
              * edges_device = nullptr;
    make_VE_for_GPU(&vertices_device,
                    &edges_device,
                    edgeTable,
                    n_points,
                    n_edges
                    );
    // Compute the relationship
    size_t ev_size = sizeof(vtkIdType) * n_edges * nbVertsInEdge;
    vtkIdType * ev_computed = nullptr,
              * ev_host = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&ev_computed, ev_size));
    CUDA_ASSERT(cudaMallocHost((void**)&ev_host, ev_size));
    vtkIdType n_to_compute = n_edges * nbVertsInEdge;
    dim3 thread_block_size = 1024,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    std::cout << "Kernel launch configuration is " << grid_size.x << " grid blocks "
              << "with " << thread_block_size.x << " threads per block" << std::endl;
    std::cout << "The mesh has " << n_points << " points and " << n_edges
              << " edges" << std::endl;
    std::cout << "Tids >= " << n_edges * nbVertsInEdge << " should auto-exit ("
              << (thread_block_size.x * grid_size.x) - n_to_compute << ")" << std::endl;
   KERNEL_WARN(EV_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                            thread_block_size>>>(vertices_device,
                                edges_device,
                                n_edges,
                                ev_computed));
    /*
    EV_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                            thread_block_size>>>(vertices_device,
                                edges_device,
                                n_edges,
                                ev_computed);
    */
    CUDA_WARN(cudaDeviceSynchronize());
    // Copy back to host and set in edgeList
    CUDA_WARN(cudaMemcpy(ev_host, ev_computed, ev_size, cudaMemcpyDeviceToHost));
    // Reconfigure into edgeList for comparison
    #pragma omp parallel for num_threads(args.threadNumber)
    for (vtkIdType e = 0; e < n_edges; ++e) {
        edgeList->emplace_back(std::array<vtkIdType,nbVertsInEdge>{ev_host[(2*e)],ev_host[(2*e)+1]});
    }
    // Free device memory
    if (vertices_device != nullptr)
        CUDA_WARN(cudaFree(vertices_device));
    if (edges_device != nullptr)
        CUDA_WARN(cudaFree(edges_device));
    if (ev_computed != nullptr)
        CUDA_WARN(cudaFree(ev_computed));

    return edgeList;
}

    // Not used yet, but I defined it anyways
    //vtkIdType * tv_device = nullptr;
    //make_TV_for_GPU(tv_device, tv_relationship);
    //if (tv_device != nullptr)
    //    CUDA_WARN(cudaFree(tv_device));

