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

    Timer ve_translation;
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
    ve_translation.tick();

    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before copy completes
    ve_translation.tick();
    CUDA_WARN(cudaMemcpy(*device_vertices, host_vertices,
                         vertices_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(*device_edges, host_edges,
                         edges_size, cudaMemcpyHostToDevice));
    ve_translation.tick();
    ve_translation.label_interval(0, "VE Host->GPU Translation");
    ve_translation.label_interval(1, "VE Host->GPU Data Transfer");
    CUDA_WARN(cudaFreeHost(host_vertices));
    CUDA_WARN(cudaFreeHost(host_edges));
}

void make_VF_for_GPU(vtkIdType * device_vertices,
                     vtkIdType * device_faces,
                     const VF_Data & vf_relationship,
                     const vtkIdType n_verts,
                     const vtkIdType n_faces) {
    // Size determinations
    size_t vertices_size = sizeof(vtkIdType) * n_faces * nbVertsInFace,
           // Can technically be one-third this size, but duplicate for now
           faces_size = sizeof(vtkIdType) * n_faces * nbVertsInFace;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)&device_vertices, vertices_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_faces, faces_size));
    vtkIdType * host_vertices = nullptr,
              * host_faces = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vertices, vertices_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_faces, faces_size));

    Timer vf_translation;
    // Set contiguous data in host memory
    for (vtkIdType vertex_id = 0, index = 0; vertex_id < n_verts; vertex_id++) {
        for (const FaceData & face : vf_relationship[vertex_id]) {
            // Pack lowest face / ID
            host_faces[index] = face.id;
            host_vertices[index++] = vertex_id;
            // Pack middle edge / ID
            host_faces[index] = face.id;
            host_vertices[index++] = face.middleVert;
            // Pack highest edge / ID
            host_faces[index] = face.id;
            host_vertices[index++] = face.highVert;
        }
    }
    vf_translation.tick();

    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before the copy completes
    vf_translation.tick();
    CUDA_WARN(cudaMemcpy(device_vertices, host_vertices, vertices_size,
                         cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(device_faces, host_faces, faces_size,
                         cudaMemcpyHostToDevice));
    vf_translation.tick();
    vf_translation.label_interval(0, "VF Host->GPU Translation");
    vf_translation.label_interval(1, "VF Host->GPU Data Transfer");
    CUDA_WARN(cudaFreeHost(host_vertices));
    CUDA_WARN(cudaFreeHost(host_faces));
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
    std::cout << INFO_EMOJI << "Kernel launch configuration is " << grid_size.x << " grid blocks "
              << "with " << thread_block_size.x << " threads per block" << std::endl;
    std::cout << INFO_EMOJI << "The mesh has " << n_points << " points and " << n_edges
              << " edges" << std::endl;
    std::cout << INFO_EMOJI << "Tids >= " << n_edges * nbVertsInEdge << " should auto-exit ("
              << (thread_block_size.x * grid_size.x) - n_to_compute << ")" << std::endl;
    Timer kernel;
    KERNEL_WARN(EV_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                            thread_block_size>>>(vertices_device,
                                edges_device,
                                n_edges,
                                ev_computed));
    CUDA_WARN(cudaDeviceSynchronize());
    kernel.tick();
    kernel.label_prev_interval("GPU kernel duration");
    // Copy back to host and set in edgeList
    kernel.tick();
    CUDA_WARN(cudaMemcpy(ev_host, ev_computed, ev_size, cudaMemcpyDeviceToHost));
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host transfer");
    kernel.tick();
    // Reconfigure into edgeList for comparison
    #pragma omp parallel for num_threads(args.threadNumber)
    for (vtkIdType e = 0; e < n_edges; ++e) {
        edgeList->emplace_back(std::array<vtkIdType,nbVertsInEdge>{ev_host[(2*e)],ev_host[(2*e)+1]});
    }
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host translation");
    // Free device memory
    if (vertices_device != nullptr) CUDA_WARN(cudaFree(vertices_device));
    if (edges_device != nullptr) CUDA_WARN(cudaFree(edges_device));
    if (ev_computed != nullptr) CUDA_WARN(cudaFree(ev_computed));
    // Free host memory
    if (ev_host != nullptr) CUDA_WARN(cudaFreeHost(ev_host));

    return edgeList;
}

__global__ void TF_kernel(vtkIdType * __restrict__ tv,
                          vtkIdType * __restrict__ vertices,
                          vtkIdType * __restrict__ faces,
                          const vtkIdType n_cells,
                          const vtkIdType n_faces,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ tf) {
    vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x,
              face = (tid % 3);
    if (tid >= (n_cells * nbFacesInCell)) return;
    // Fetch your cell ID based on your vertex-index
    // Fetch your face ID based on your vertex-index
    // Write your face ID into your cell's list at the appropriate spot
    //tf[(tv[vertices[tid]] * nbFacesInCell) + face] = faces[tid];
}

std::unique_ptr<TF_Data> make_TF_GPU(const TV_Data & TV,
                                     const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const vtkIdType n_cells,
                                     const arguments args) {
    std::unique_ptr<TF_Data> TF = std::make_unique<TF_Data>();
    TF->reserve(n_cells);

    // Make data ready for GPU
    vtkIdType * tv_device = nullptr,
              * vertices_device = nullptr,
              * faces_device = nullptr;
    make_TV_for_GPU(tv_device, TV);
    make_VF_for_GPU(vertices_device, faces_device, VF, n_points, n_faces);

    // Compute the relationship
    size_t tf_size = sizeof(vtkIdType) * n_cells * nbFacesInCell;
    vtkIdType * tf_computed = nullptr,
              * tf_host = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&tf_computed, tf_size));
    CUDA_ASSERT(cudaMalloc((void**)&tf_host, tf_size));
    vtkIdType n_to_compute = n_cells * nbFacesInCell;
    dim3 thread_block_size = 1024,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    std::cout << INFO_EMOJI << "Kernel launch configuration is " << grid_size.x
              << " grid blocks with " << thread_block_size.x << " threads per block"
              << std::endl;
    std::cout << INFO_EMOJI << "The mesh has " << n_cells << " cells and "
              << n_faces << " faces" << std::endl;
    std::cout << INFO_EMOJI << "Tids >= " << n_cells * nbFacesInCell << " should auto-exit ("
              << (thread_block_size.x * grid_size .x) - n_to_compute << ")"
              << std::endl;
    Timer kernel;
    KERNEL_WARN(TF_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                            thread_block_size>>>(tv_device,
                                vertices_device,
                                faces_device,
                                n_cells,
                                n_faces,
                                n_points,
                                tf_computed));
    CUDA_WARN(cudaDeviceSynchronize());
    kernel.tick();
    kernel.label_prev_interval("GPU kernel duration");
    // Copy back to host and set in edgeList
    kernel.tick();
    CUDA_WARN(cudaMemcpy(tf_host, tf_computed, tf_size, cudaMemcpyDeviceToHost));
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host transfer");
    kernel.tick();
    // Reconfigure into host-side structure for comparison
    std::cerr << "Not implemented: Device->Host memory transformation" << std::endl;
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host translation");
    // Free device memory
    if (tv_device != nullptr) CUDA_WARN(cudaFree(tv_device));
    if (vertices_device != nullptr) CUDA_WARN(cudaFree(vertices_device));
    if (faces_device != nullptr) CUDA_WARN(cudaFree(faces_device));
    if (tf_computed != nullptr) CUDA_WARN(cudaFree(tf_computed));
    // Free host memory -- segfaults when present, no leaks when absent, IDK why
    //if (tf_host != nullptr) CUDA_WARN(cudaFreeHost(tf_host));
    return TF;
}

