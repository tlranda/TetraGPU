#include "cuda_extraction.h"

/* There are a few classes of functions here and general design patterns to be
   aware of:
    * "make_X_for_GPU" functions allocate GPU-side memory for X via a reference
        to a pointer on the caller's frame and use the host to populate
        appropriate values. No kernels, just remapping and data transfer.
    * "make_X_GPU" functions use GPU-side memory and kernels to create X using
        the GPU. Currently, these functions remap the memory to host and clean
        up the device allocation before exiting, but real applications may want
        the GPU-side relationship to persist (or to not be remapped in the same
        manner).
    * "X_kernel" functions are the device-side kernels that create new
        relationships using the GPU. They are called by the corresponding X's
        "make_X_GPU" kernel. All of the kernels attempt to utilize the GPU's
        thread scalability to get massive parallelism even if the linear
        algebra approach is very sparse. Our hope is that you can tie these
        functions together (perhaps as inlines) to make a device-contained
        relationship remap fit within the algorithm kernel with great locality.

  As a general note, everything is currently scheduled on the default stream;
  you may want to take care to adjust cudaMemcpy and kernel invokations if
  overlapping is intended down the line.

  So far, we've found the "vectorized" kernel approach to be promising enough
  for relationship precompute. However, the approach is not trivial and
  requires several key strategies which you may want to re-use as you expand
  the relationship coverage.
    * "EV_kernel" shows a basic transpose from VE (split via "make_VE_for_GPU")
        and has coalesced reads with poor write ordering. You'll get one or the
        other to coalesce, and hardware generally optimizes for reads better.
    * "TF_kernel" shows a register-shuffle for exchanging global reads without
        shared memory. This ONLY works if you can do exchanges on powers-of-two
        exactly and get all necessary information. It then has a diverging
        block to handle thread-face assignment and a less-diverging scan lookup
        through the precomputed VF data to locate the correct face ID.
    * "TE_kernel" shows a shared-memory approach for exchanging global reads
        when it's necessary to do so on non-powers-of-two subsets. It also
        unrolls its relationship extraction to make more usage out of the
        requested shared memory. This unrolling pattern can technically be
        looped, but you have to adjust all constants to properly account for
        doing that (there are a bunch!). Pay special attention to definitions
        and comments that detail how these constants should be determined and
        adjusted for various constraints on the register file and occupancy.

  I've also run into a small number of strange behaviors here that are handled
  by workarounds for now; if you find a better way to do it, be vigilant to
  ensure all workarounds are replaced with the better behavior:
    * std::fill for host-side memory in "make_X_for_GPU" should work, but seems
        to segfault on certain sizes that our tests definitely reach and exceed.
        It may have something to do with working on pointers from
        CUDA_MALLOC_HOST, but root-causes were not identified and instead we
        see if the compiler understands the intent and can pitch in
        optimizations on the trivial loop with constant value assignments.
        Feel free to add #pragma unroll etc but I don't think they're present
        yet.
*/

void make_TV_for_GPU(vtkIdType ** device_tv,
                           // vector of array of vertices in a tetra
                     const TV_Data & tv_relationship) {
    // Size determination
    size_t tv_flat_size = sizeof(vtkIdType) * tv_relationship.nCells * nbVertsInCell;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)device_tv, tv_flat_size));
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
    CUDA_WARN(cudaMemcpy(*device_tv, host_flat_tv,
                         tv_flat_size, cudaMemcpyHostToDevice));
    CUDA_WARN(cudaFreeHost(host_flat_tv));
}

void make_VE_for_GPU(vtkIdType ** device_vertices,
                     vtkIdType ** device_edges,
                     vtkIdType ** device_first_vertex,
                           // vector of vectors of edge IDs
                     const VE_Data & ve_relationship,
                     const vtkIdType n_verts,
                     const vtkIdType n_edges
                     ) {
    // Size determinations
    size_t vertices_size = sizeof(vtkIdType) * n_edges * nbVertsInEdge,
           // Can technically be half-sized, but duplicate for now so index in
           // vertices directly maps to edgeID without further manip (revisit
           // later as minor optimization -- would just drop multiplier of
           // nbVertsInEdge and adjust EVERY kernel making use of the edges
           // array to left-shift its index one bit (divide by two))
           edges_size = sizeof(vtkIdType) * n_edges * nbVertsInEdge,
           index_vertex_size = sizeof(vtkIdType) * n_verts;
    // Allocations
    CUDA_ASSERT(cudaMalloc((void**)device_vertices, vertices_size));
    CUDA_ASSERT(cudaMalloc((void**)device_edges, edges_size));
    CUDA_ASSERT(cudaMalloc((void**)device_first_vertex, index_vertex_size));
    vtkIdType * host_vertices = nullptr,
              * host_edges = nullptr,
              * host_first_vertices = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vertices, vertices_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_edges, edges_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_first_vertices, index_vertex_size));

    Timer ve_translation;
    // Set contiguous data in host memory
    // Index defaults to END-OF-LIST to help with scanning
    // while std::fill should work, it can segfault on sizes (see the similar
    // code in make_VF_for_GPU() for further explanation; same bugfix should
    // apply here if one is found
    {
        const vtkIdType val = (n_verts+1)*nbVertsInEdge;
        for (vtkIdType i = 0; i < n_verts; i++) {
            host_first_vertices[i] = val;
        }
    }
    for (vtkIdType vertex_id = 0, index = 0, first = 0; vertex_id < n_verts; vertex_id++) {
        host_first_vertices[first++] = index;
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
    CUDA_WARN(cudaMemcpy(*device_first_vertex, host_first_vertices,
                         index_vertex_size, cudaMemcpyHostToDevice));
    ve_translation.tick();
    ve_translation.label_interval(0, "VE Host->GPU Translation");
    ve_translation.label_interval(1, "VE Host->GPU Data Transfer");
    CUDA_WARN(cudaFreeHost(host_vertices));
    CUDA_WARN(cudaFreeHost(host_edges));
    CUDA_WARN(cudaFreeHost(host_first_vertices));
}

void make_VF_for_GPU(vtkIdType ** device_vertices,
                     vtkIdType ** device_faces,
                     vtkIdType ** device_first_faces,
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
    CUDA_ASSERT(cudaMalloc((void**)device_vertices, vertices_size));
    CUDA_ASSERT(cudaMalloc((void**)device_faces, faces_size));
    CUDA_ASSERT(cudaMalloc((void**)device_first_faces, index_face_size));
    vtkIdType * host_vertices = nullptr,
              * host_faces = nullptr,
              * host_first_faces = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vertices, vertices_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_faces, faces_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_first_faces, index_face_size));

    Timer vf_translation;
    // Set contiguous data in host memory
    // max_real_value = n_faces * nbVertsInFace
    // While std::fill should work, it can segfault on sizes that otherwise work?
    // KNOWN ISSUE: This size only appears to support up to 262,144 bytes allocation
    // in subsequent CUDA_MALLOC_HOST when the value is set to 800,000 bytes (100k vertices)
    // Not sure why -- may readdress later
    //std::fill(host_first_faces, host_first_faces+index_face_size, (n_faces+1) * nbVertsInFace);
    {
        const vtkIdType val = (n_faces+1)*nbVertsInFace;
        for (vtkIdType i = 0; i < n_verts; i++) {
            host_first_faces[i] = val;
        }
    }
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
            if (vertex_id == n_verts-1)
                host_first_faces[vertex_id] = n_faces * nbVertsInFace;
            else
                host_first_faces[vertex_id] = host_first_faces[vertex_id+1];
        }
    }
    vf_translation.tick();

    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before the copy completes
    vf_translation.tick();
    CUDA_WARN(cudaMemcpy(*device_vertices, host_vertices, vertices_size,
                         cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(*device_faces, host_faces, faces_size,
                         cudaMemcpyHostToDevice));
    CUDA_WARN(cudaMemcpy(*device_first_faces, host_first_faces, index_face_size,
                         cudaMemcpyHostToDevice));
    vf_translation.tick();
    vf_translation.label_interval(0, "VF Host->GPU Translation");
    vf_translation.label_interval(1, "VF Host->GPU Data Transfer");
    CUDA_WARN(cudaFreeHost(host_vertices));
    CUDA_WARN(cudaFreeHost(host_faces));
    CUDA_WARN(cudaFreeHost(host_first_faces));
}

__global__ void EV_kernel(const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ edges,
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
              * edges_device = nullptr,
              * index_device = nullptr;
    make_VE_for_GPU(&vertices_device,
                    &edges_device,
                    &index_device,
                    edgeTable,
                    n_points,
                    n_edges
                    );
    // Free index_device as EV does not need it
    if (index_device != nullptr) CUDA_WARN(cudaFree(index_device));
    // Compute the relationship
    size_t ev_size = sizeof(vtkIdType) * n_edges * nbVertsInEdge;
    vtkIdType * ev_computed = nullptr,
              * ev_host = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&ev_computed, ev_size));
    CUDA_ASSERT(cudaMallocHost((void**)&ev_host, ev_size));
    vtkIdType n_to_compute = n_edges * nbVertsInEdge;
    dim3 thread_block_size = 1024,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    std::cout << INFO_EMOJI << "Kernel launch configuration is " << grid_size.x
              << " grid blocks with " << thread_block_size.x
              << " threads per block" << std::endl;
    std::cout << INFO_EMOJI << "The mesh has " << n_points << " points and "
              << n_edges << " edges" << std::endl;
    std::cout << INFO_EMOJI << "Tids >= " << n_edges * nbVertsInEdge
              << " should auto-exit ("
              << (thread_block_size.x * grid_size.x) - n_to_compute << ")"
              << std::endl;
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
    for (vtkIdType e = 0; e < n_edges; ++e)
        edgeList->emplace_back(std::array<vtkIdType,nbVertsInEdge>{
                                    ev_host[(2*e)],ev_host[(2*e)+1]});
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

__global__ void TF_kernel(const vtkIdType * __restrict__ tv,
                          const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ faces,
                          const vtkIdType * __restrict__ first_faces,
                          const vtkIdType n_cells,
                          const vtkIdType n_faces,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ tf) {
    vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x,
              face = (tid % 4);
    if (tid >= (n_cells * nbFacesInCell)) return;

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

    // While syncing may not be strictly required, empirically it seems to make
    // the kernel faster on average
    __syncthreads();

    // !! Scan VF for your face match -- divergence expected !!
    // We do NOT guard against an out-of-bounds check on the condition, as the
    // LOW face explicitly has 2 other vertices higher than it, therefore those
    // vertices always define an upper bound without touching OOM (worst-case
    // those vertices indicate to go to the end of the vertices array)
    for (vtkIdType i = first_faces[face_low]; i < first_faces[face_low+1]; i += 3) {
        if (vertices[i+1] == face_mid && vertices[i+2] == face_high) {
            tf[tid] = faces[i];
            break;
        }
        __syncthreads();
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
    make_TV_for_GPU(&tv_device, TV);
    make_VF_for_GPU(&vertices_device, &faces_device, &first_faces_device, VF,
                    n_points, n_faces);

    // Compute the relationship
    size_t tf_size = sizeof(vtkIdType) * n_cells * nbFacesInCell;
    vtkIdType * tf_computed = nullptr,
              * tf_host = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&tf_computed, tf_size));
    CUDA_ASSERT(cudaMallocHost((void**)&tf_host, tf_size));
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
    if (tf_host != nullptr) CUDA_WARN(cudaFreeHost(tf_host));
    return TF;
}

__device__ __inline__ void te_combine(vtkIdType quad0, vtkIdType quad1,
                                      vtkIdType quad2, vtkIdType quad3,
                                      const vtkIdType laneID,
                                      vtkIdType * __restrict__ te,
                                      const vtkIdType * __restrict__ vertices,
                                      const vtkIdType * __restrict__ edges,
                                      const vtkIdType n_points,
                                      const vtkIdType * __restrict__ index) {
    // Within each sub-group, assign unique combination of vertex pairs from quad
    // Then look up the edge ID in VE and assign it to TE
    /* Pattern:
       0: q0 - q1
       1: q1 - q2
       2: q2 - q3
       3: q0 - q2
       4: q1 - q3
       5: q0 - q3
    */
    vtkIdType left_vertex = quad0, // 0, 3, 5
              right_vertex = quad3; // 2, 4, 5
    if (laneID == 1 || laneID == 2 || laneID == 4) {
        if (laneID == 2) left_vertex = quad2;
        else /* 1, 4 */ left_vertex = quad1;
    }
    if (laneID == 0 || laneID == 1 || laneID == 3) {
        if (laneID == 0) right_vertex = quad1;
        else /* 1, 3 */ right_vertex = quad2;
    }
    // Ensure lowest index is the left one
    if (left_vertex > right_vertex) {
        vtkIdType swap = left_vertex;
        left_vertex = right_vertex;
        right_vertex = swap;
    }

    __syncthreads();
    // !! Scan VE for first edge match -- divergence expected !!
    // There is no OOB guard on the for-loop condition as the LOWER index is
    // explicitly less than the HIGHER index, ergo index[left_vertex+1] is
    // definitely in-bounds
    for (vtkIdType i = index[left_vertex]; i < index[left_vertex+1]; i+= 2) {
        // vertices = [low-edge, high-edge] x n-Edges
        // edges =    [edge id , edge id  ] x n-Edges
        if (vertices[i+1] == right_vertex) {
            // TE is already shifted for every thread, so just write to your
            // laneID and that should mark the edge
            te[laneID] = edges[i];
            break;
        }
    }
    __syncthreads();
}

#define TE_CELLS_PER_BLOCK 195
__global__ void TE_kernel(const vtkIdType * __restrict__ tv,
                          const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ edges,
                          const vtkIdType * __restrict__ first_index,
                          const vtkIdType n_cells,
                          const vtkIdType n_edges,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ te) {
    // LAUNCH WITH 6 THREADS PER CELL, LOSE 2 THREADS PER WARP (32) WHICH
    // REQUIRES OVERSUBSCRIPTION IMMEDIATELY
    // ALSO MUST ALLOCATE ENOUGH SHARED MEMORY FOR KERNEL.

    // TAKE CARE THAT CONSTANTS ARE WRITTEN FOR UNROLLING 3 LOOP ITERATIONS,
    // IF UNROLLING MORE OR LESS, THESE CONSTANTS MUST BE UPDATED
    extern __shared__ vtkIdType sh_scratch[];

    vtkIdType tid = (blockIdx.x * blockDim.x) + threadIdx.x,
              warpID = (threadIdx.x % 32),
              laneID = warpID % 6,
              laneDepth = 3*(((tid / 32)*5) + (warpID / 6)),
              /* shLaneDepth MUST ALWAYS BE MODULO THE CELLS_PER_BLOCK VALUE */
              shLaneDepth = laneDepth % TE_CELLS_PER_BLOCK,
              edge = (tid % nbEdgesInCell);
    // Early-exit threads reading beyond #cells at base index AND 2 straggler threads per warp
    if (laneDepth >= n_cells || warpID > 29) return;

    // Push output pointer TE per-thread to its writing position
    te += (laneDepth * 6);

    // Read FIRST value from global memory --> shared
    // laneDepth *= 4 to use vector-addressing; not set permanently as the
    // cellID is nice to hold onto for later
    vtkIdType read_indicator = n_cells-laneDepth-1;
    if (read_indicator >= 1 || (read_indicator == 0 && laneID < 4)) {
        sh_scratch[(shLaneDepth*6)+laneID] = tv[(laneDepth*4)+laneID];
    }
    __syncthreads();

    // UNROLL 1: First quadruplet is guaranteed to be useful due to early-exit threads no longer being present
    vtkIdType quad0 = sh_scratch[(shLaneDepth*6)  ],
              quad1 = sh_scratch[(shLaneDepth*6)+1],
              quad2 = sh_scratch[(shLaneDepth*6)+2],
              quad3 = sh_scratch[(shLaneDepth*6)+3];
    // All 6 combinations of values need to be made to get the TE relationship,
    // but the edgeID has to be looked up in VE relationship
    te_combine(quad0,quad1,quad2,quad3, laneID, te, vertices, edges, n_points,
               first_index);

    // UNROLL 2: Second quadruplet is half-read already; exit if NOT useful
    if (read_indicator == 0) return;
    // Adjust pointers to not overwrite previous iteration's data
    te += 6;
    quad0 = sh_scratch[(shLaneDepth*6)+4];
    quad1 = sh_scratch[(shLaneDepth*6)+5];
    __syncthreads();
    // Continue reading for unrolls 2 & 3
    if (read_indicator > 1 || (read_indicator == 1 & laneID < 2)) {
        sh_scratch[(shLaneDepth*6)+laneID] = tv[(laneDepth*4)+laneID+6];
    }
    __syncthreads();
    quad2 = sh_scratch[(shLaneDepth*6)  ];
    quad3 = sh_scratch[(shLaneDepth*6)+1];
    te_combine(quad0,quad1,quad2,quad3, laneID, te, vertices, edges, n_points,
               first_index);

    // UNROLL 3: Third quadruplet is read; early exit if NOT useful
    if (read_indicator == 1) return;
    // Adjust pointers to not overwrite previous iteration's data
    te += 6;
    quad0 = sh_scratch[(shLaneDepth*6)+2];
    quad1 = sh_scratch[(shLaneDepth*6)+3];
    quad2 = sh_scratch[(shLaneDepth*6)+4];
    quad3 = sh_scratch[(shLaneDepth*6)+5];
    te_combine(quad0,quad1,quad2,quad3, laneID, te, vertices, edges, n_points,
               first_index);
}
// TE = TV x VE
std::unique_ptr<TE_Data> make_TE_GPU(const TV_Data & TV,
                                     const VE_Data & VE,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const vtkIdType n_cells,
                                     const arguments args) {
    std::unique_ptr<TE_Data> TE = std::make_unique<TE_Data>();
    TE->reserve(n_cells);

    // Make ready for GPU
    vtkIdType * tv_device = nullptr,
              * vertices_device = nullptr,
              * edges_device = nullptr,
              * index_device = nullptr;
    make_TV_for_GPU(&tv_device, TV);
    make_VE_for_GPU(&vertices_device,
                    &edges_device,
                    &index_device,
                    VE,
                    n_points,
                    n_edges
                    );

    // Compute relationship
    vtkIdType n_to_compute = n_cells * nbEdgesInCell;
    size_t te_size = sizeof(vtkIdType) * n_to_compute;
    vtkIdType * te_computed = nullptr,
              * te_host = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&te_computed, te_size));
    CUDA_ASSERT(cudaMallocHost((void**)&te_host, te_size));

    // Set up launch configuration for the kernel
    const vtkIdType N_THREADS = 416,
                    /*
                       6 edges required per cell (1 edge : 1 thread)
                       Up to 3 cells unrolled in each group of threads
                       -2 threads per warp of 32 threads for warp alignment on factor of 6
                       480 cells comes from:
                       6*((480+2)//3) == 6 * 160 = 960 work with unrolling
                       ((960+29)//30)*32 == 32*32 = 1024 threads in block

                       Max 1024 threads per block in hardware, increasing to 481 threads requires a new block for the warp

                       -- however, register usage can pose an even greater problem for us --

                       1024 threads * 78 registers (current HW) = 79,872 / 65,536 registers demanded
                       Our early-exits cost us in that the CUDA launch API has
                       no clue that we're going to honor that

                       At 78 registers, we can use up to 840 threads in a block
                       Round this down to 832 == 26*32 (fullwarp alignment)
                       Each warp has 5 groups (26*5 == 130 single-cells), with
                       3 unrolled for 390 cells per block after unrolling

                       The above isn't working on this hardware, idk let's cut
                       it in half. 416 threads -> 13 full warps AKA 65 groups
                       unrolling to 195 cells in a block

                       The value ALSO needs to be set within the kernel, so
                       update the TE_CELLS_PER_BLOCK preprocessor definition
                       above the TE_kernel() function if you need to change it
                    */
                    cells_per_block = TE_CELLS_PER_BLOCK,
                    SHARED_PER_BLOCK = cells_per_block * 6 * sizeof(vtkIdType);
    vtkIdType N_BLOCKS = (n_cells+cells_per_block-1)/cells_per_block;

    std::cout << INFO_EMOJI << "Kernel launch configuration is " << N_BLOCKS
              << " grid blocks with " << N_THREADS << " threads per block"
              << " and " << SHARED_PER_BLOCK << " bytes shmem per block"
              << std::endl;
    std::cout << INFO_EMOJI << "The mesh has " << n_cells << " cells and "
              << n_edges << " edges" << std::endl;
    if (cudaFuncSetAttribute(TE_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             49152/*SHARED_PER_BLOCK*/) != cudaSuccess) {
        std::cerr << WARN_EMOJI << "Could not set max dynamic shared memory size to "
                  << SHARED_PER_BLOCK << " bytes" << std::endl;
    }
    Timer kernel;
    KERNEL_WARN(TE_kernel<<<N_BLOCKS KERNEL_LAUNCH_SEPARATOR
                            N_THREADS KERNEL_LAUNCH_SEPARATOR
                            SHARED_PER_BLOCK>>>(tv_device,
                                vertices_device,
                                edges_device,
                                index_device,
                                n_cells,
                                n_edges,
                                n_points,
                                te_computed));
    CUDA_WARN(cudaDeviceSynchronize());
    kernel.tick();
    kernel.label_prev_interval("GPU kernel duration");

    // Copy back to host with transformation
    kernel.tick();
    CUDA_WARN(cudaMemcpy(te_host, te_computed, te_size, cudaMemcpyDeviceToHost));
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host transfer");
    kernel.tick();
    // Reconfigure for host
    for (vtkIdType c = 0; c < n_cells; ++c) {
        TE->emplace_back(std::array<vtkIdType,nbEdgesInCell>{
                te_host[(6*c)  ], te_host[(6*c)+1],
                te_host[(6*c)+2], te_host[(6*c)+3],
                te_host[(6*c)+4], te_host[(6*c)+5]
                });
    }
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host translation");
    // Free device memory
    if (tv_device != nullptr) CUDA_WARN(cudaFree(tv_device));
    if (vertices_device != nullptr) CUDA_WARN(cudaFree(vertices_device));
    if (edges_device != nullptr) CUDA_WARN(cudaFree(edges_device));
    if (te_computed != nullptr) CUDA_WARN(cudaFree(te_computed));
    if (te_host != nullptr) CUDA_WARN(cudaFreeHost(te_host));
    return TE;
}

__global__ void FV_kernel(const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ faces,
                          const vtkIdType n_faces,
                          vtkIdType * __restrict__ fv) {
    vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x,
              vert_idx = (tid % 3);
    if (tid >= (n_faces * nbVertsInFace)) return;
    fv[(faces[tid] * nbVertsInFace) + vert_idx] = vertices[tid];
}

std::unique_ptr<FV_Data> make_FV_GPU(const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const arguments args) {
    // FV_data = std::vector<FaceData{middleVert,highVert,id}>
    std::unique_ptr<FV_Data> vertexList = std::make_unique<FV_Data>();
    vertexList->reserve(n_faces);

    // Marshall data to GPU
    vtkIdType * vertices_device = nullptr,
              * faces_device = nullptr,
              * index_device = nullptr;
    make_VF_for_GPU(&vertices_device,
                    &faces_device,
                    &index_device,
                    VF,
                    n_points,
                    n_faces
                    );
    // Free index device as FV does not need it
    if (index_device != nullptr) CUDA_WARN(cudaFree(index_device));
    // Compute the relationship
    size_t fv_size = sizeof(vtkIdType) * n_faces * nbVertsInFace;
    vtkIdType * fv_computed = nullptr,
              * fv_host = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&fv_computed, fv_size));
    CUDA_ASSERT(cudaMallocHost((void**)&fv_host, fv_size));
    vtkIdType n_to_compute = n_faces * nbVertsInFace;
    dim3 thread_block_size = 1024,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    std::cout << INFO_EMOJI << "Kernel launch configuration is " << grid_size.x
              << " grid blocks with " << thread_block_size.x
              << " threads per block" << std::endl;
    std::cout << INFO_EMOJI << "The mesh has " << n_points << " points and "
              << n_faces << " faces" << std::endl;
    std::cout << INFO_EMOJI << "Tids >= " << n_faces * nbVertsInFace
              << " should auto-exit ("
              << (thread_block_size.x * grid_size.x) - n_to_compute << ")"
              << std::endl;
    Timer kernel;
    KERNEL_WARN(FV_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                            thread_block_size>>>(vertices_device,
                                faces_device,
                                n_faces,
                                fv_computed));
    CUDA_WARN(cudaDeviceSynchronize());
    kernel.tick();
    kernel.label_prev_interval("GPU kernel duration");
    // Copy back to host and set in vertexList
    kernel.tick();
    CUDA_WARN(cudaMemcpy(fv_host, fv_computed, fv_size, cudaMemcpyDeviceToHost));
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host transfer");
    kernel.tick();
    // Reconfigure for host comparison
    //#pragma omp parallel for num_threads(args.threadNumber)
    for (vtkIdType f = 0; f < n_faces; ++f) {
        vertexList->emplace_back(FaceData(fv_host[(3*f)], fv_host[(3*f)+1],
                                          fv_host[(3*f)+2]));
    }
    kernel.tick();
    kernel.label_prev_interval("GPU Device->Host translation");
    // Free device memory
    if (vertices_device != nullptr) CUDA_WARN(cudaFree(vertices_device));
    if (faces_device != nullptr) CUDA_WARN(cudaFree(faces_device));
    if (fv_computed != nullptr) CUDA_WARN(cudaFree(fv_computed));
    // Free host memory
    if (fv_host != nullptr) CUDA_WARN(cudaFreeHost(fv_host));

    return vertexList;
}

