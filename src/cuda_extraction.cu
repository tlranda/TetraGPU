#include "cuda_extraction.h"

void make_TV_for_GPU(vtkIdType * device_tv,
                           // vector of array of vertices in a tetra
                     const TV_Data & tv_relationship) {
    // Size determination
    size_t tv_flat_size = sizeof(vtkIdType) * tv_relationship.nCells * nbVertsInCell;
    std::cout << "Allocating " << tv_relationship.nCells * nbVertsInCell << " vtkIdTypes for TV on GPU" << std::endl;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)&device_tv, tv_flat_size));
    vtkIdType * host_flat_tv = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_flat_tv, tv_flat_size));

    // Set contiguous data in host memory
    vtkIdType index = 0;
    for (const auto & VertList : tv_relationship) {
        /* Prototyping aid: see actual data order */
        //std::cout << "Tetra [" << index / 4 << "] Vertices: [";
        for (const vtkIdType vertex : VertList) {
            host_flat_tv[index++] = vertex;
            //std::cout << vertex << " ";
        }
        //std::cout << "]" << std::endl;
    }
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
                     vtkIdType * device_first_faces,
                     const VF_Data & vf_relationship,
                     const vtkIdType n_verts,
                     const vtkIdType n_faces) {
    // Size determinations
    size_t vertices_size = sizeof(vtkIdType) * n_faces * nbVertsInFace,
           // Can technically be one-third this size, but duplicate for now
           faces_size =    sizeof(vtkIdType) * n_faces * nbVertsInFace,
           // Index into other arrays
           index_face_size = sizeof(vtkIdType) * n_verts;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)&device_vertices, vertices_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_faces, faces_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_first_faces, index_face_size));
    vtkIdType * host_vertices = nullptr,
              * host_faces = nullptr,
              * host_first_faces = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vertices, vertices_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_faces, faces_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_first_faces, index_face_size));

    Timer vf_translation;
    // Set contiguous data in host memory
    // max_real_value = n_faces * nbVertsInFace
    std::fill(host_first_faces, host_first_faces+index_face_size, (n_faces+1) * nbVertsInFace);
    for (vtkIdType vertex_id = 0, index = 0; vertex_id < n_verts; vertex_id++) {
        for (const FaceData & face : vf_relationship[vertex_id]) {
            // Update first-face index if necessary
            if (host_first_faces[vertex_id] > index) host_first_faces[vertex_id] = index;
            // Pack lowest face / ID
            host_faces[index] = face.id;
            host_vertices[index++] = vertex_id;
            // Pack middle edge / ID
            host_faces[index] = face.id;
            host_vertices[index++] = face.middleVert;
            // Pack highest edge / ID
            host_faces[index] = face.id;
            host_vertices[index++] = face.highVert;
            /* Prototyping aid: see actual data order
            std::cout << "Face [" << face.id << "] Vertices [" << vertex_id
                      << ", " << face.middleVert << ", " << face.highVert
                      << "]" << std::endl;
            */
        }
    }
    // We use idx+1 to set a scan limit when looking for the face, though TBH
    // in C++/CUDA unless there's a bug in my logic we don't actually need to
    // set a scan limit as we'll find the face at before that value (and we can
    // always skip anything behind our first_face ID). But anyways, this will
    // ensure that our first_face values are monotonically increasing and if
    // we're able to do anything clever by knowing the scanning range, then you
    // merely look at the next element in the array to know when to stop.
    // We have to reverse-iterate the array to ensure sequences of 0-length
    // vertices are handled correctly, which is highly likely to occur a bunch
    // at the high-end of this data structure
    for (vtkIdType vertex_id = n_verts-1; vertex_id >= 0; vertex_id--) {
        if (host_first_faces[vertex_id] == (n_faces+1) * nbVertsInFace) {
            if (vertex_id == n_verts-1) {
                host_first_faces[vertex_id] = n_faces * nbVertsInFace;
            }
            else {
                host_first_faces[vertex_id] = host_first_faces[vertex_id+1];
            }
        }
    }
    vf_translation.tick();
    /*
    for (vtkIdType vertex_id = 0; vertex_id < n_verts; vertex_id++) {
        std::cout << "First face of vertex " << vertex_id << " = "
                  << host_first_faces[vertex_id] / 3 << std::endl;
    }
    */

    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before the copy completes
    vf_translation.tick();
    CUDA_WARN(cudaMemcpy(device_vertices, host_vertices, vertices_size,
                         cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(device_faces, host_faces, faces_size,
                         cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(device_first_faces, host_first_faces, index_face_size,
                         cudaMemcpyHostToDevice));
    vf_translation.tick();
    vf_translation.label_interval(0, "VF Host->GPU Translation");
    vf_translation.label_interval(1, "VF Host->GPU Data Transfer");
    CUDA_WARN(cudaFreeHost(host_vertices));
    CUDA_WARN(cudaFreeHost(host_faces));
    CUDA_WARN(cudaFreeHost(host_first_faces));
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
                          vtkIdType * __restrict__ first_faces,
                          const vtkIdType n_cells,
                          const vtkIdType n_faces,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ tf) {
    vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x,
              face = (tid % 4);
    if (tid >= 32) return; //(n_cells * nbFacesInCell)) return;

    // Read your TV value -- because there are 4 vertices in a cell, every warp
    // is automatically cell-aligned in memory along 8 cells :)
    vtkIdType cell_vertex = tv[tid], v0, v1, v2, v3;
    // Use register exchanges within the warp to read all other values for your
    // cell
    v0 = __shfl_sync(0xffffffff, cell_vertex, 0, 4);
    v1 = __shfl_sync(0xffffffff, cell_vertex, 1, 4);
    v2 = __shfl_sync(0xffffffff, cell_vertex, 2, 4);
    v3 = __shfl_sync(0xffffffff, cell_vertex, 3, 4);

    // !! Define each TID's represented face -- divergence expected !!
    /*
       f0 = v0 - v1 - v2
       f1 = v1 - v2 - v3
       f2 = v0 - v2 - v3
       f3 = v0 - v1 - v3
    */
    vtkIdType face_low = v0, face_mid = v1, face_high = v3;
    if (face == 0) face_high = v2;
    if (face == 1 || face == 2) {
        face_mid = v2;
        if (face == 1) face_low = v1;
    }

    // !! Scan VF for your face match -- divergence expected !!
    // We do NOT guard against an out-of-bounds check on the condition, as the
    // LOW face explicitly has 2 other vertices higher than it, therefore those
    // vertices always define an upper bound without touching OOM.
    for (vtkIdType i = first_faces[face_low]; i < first_faces[face_low+1]; i += 3) {
        if (vertices[i+1] == face_mid && vertices[i+2] == face_high) {
            tf[tid] = faces[i];
            // !! We can exit the threads that find their face early as a way
            // to cut down on sustained divergence !!
            return;
            // If more processing exists within this kernel later on, then we
            // can break instead
        }
    }
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
              * faces_device = nullptr,
              * first_faces_device = nullptr;
    make_TV_for_GPU(tv_device, TV);
    make_VF_for_GPU(vertices_device, faces_device, first_faces_device, VF,
                    n_points, n_faces);

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
                                first_faces_device,
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
    //#pragma omp parallel for num_threads(args.threadNumber)
    for (vtkIdType c = 0; c < n_cells; ++c) {
        TF->emplace_back(std::array<vtkIdType,nbFacesInCell>{
                tf_host[(nbFacesInCell*c)],
                tf_host[(nbFacesInCell*c)+1],
                tf_host[(nbFacesInCell*c)+2],
                tf_host[(nbFacesInCell*c)+3],
                });
    }
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host translation");
    // Free device memory
    if (tv_device != nullptr) CUDA_WARN(cudaFree(tv_device));
    if (vertices_device != nullptr) CUDA_WARN(cudaFree(vertices_device));
    if (faces_device != nullptr) CUDA_WARN(cudaFree(faces_device));
    if (first_faces_device != nullptr) CUDA_WARN(cudaFree(first_faces_device));
    if (tf_computed != nullptr) CUDA_WARN(cudaFree(tf_computed));
    // Free host memory -- segfaults when present, no leaks when absent, IDK why
    //if (tf_host != nullptr) CUDA_WARN(cudaFreeHost(tf_host));
    return TF;
}

