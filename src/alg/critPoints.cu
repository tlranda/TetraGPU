#include "alg/critPoints.h"

/* Drives a critical points algorithm on an unstructured tetrahedral mesh. This
   can generalize to higher-order meshes if they are preprocessed to be divided
   into tetrahedra (not included).

   Mesh data is loaded via the VTK unstructured format (.vtu). Only scalar data
   is supported on the mesh at this time (not vectors on the mesh). We only
   classify points as regular, maximum, minimum, or saddles (we do not inspect
   sub-classes of saddles).
*/

// This kernel helps to ensure consistent and accurate timing of device-side
// events
__global__ void dummy_kernel(void) {
    //int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
}

#define MAXIMUM_CLASS 1
#define MINIMUM_CLASS 2
#define REGULAR_CLASS 3
#define SADDLE_CLASS  4

#define TID_SELECTION (blockDim.x * blockIdx.x) + threadIdx.x
#define PRINT_ON 0
#define CONSTRAIN_BLOCK 0
/*
// Debugging specific block executions
#define FORCED_BLOCK_IDX 9904
#define TID_SELECTION (blockDim.x * FORCED_BLOCK_IDX) + threadIdx.x
#define PRINT_ON 1
#define CONSTRAIN_BLOCK 1
*/


__global__ void critPoints(const int * __restrict__ VV,
                           const unsigned int * __restrict__ VV_index,
                           int * __restrict__ valences,
                           const int points,
                           const int max_VV_guess,
                           const double * __restrict__ scalar_values,
                           unsigned int * __restrict__ classes) {
    extern __shared__ unsigned int block_shared[];

    unsigned int *component_edits = &block_shared[0],
                 *upper_edits = &block_shared[0],
                 *lower_edits = &block_shared[1],
                 *neighborhood = &block_shared[2];;
    const int tid = TID_SELECTION;
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    const int my_1d = tid / max_VV_guess,
              my_2d = VV[tid];
    // No work for this point's valence
    if (VV_index[my_1d] <= 0 || my_2d < 0) return;
    // Prefix scan as anti-duplication
    for (int i = my_1d * max_VV_guess; i < tid; i++) {
        if (VV[i] == my_2d) return;
    }

    // BEYOND THIS POINT, YOU ARE AN ACTUAL WORKER THREAD ON THE PROBLEM

    /*
        2) Read the scalar value used for point classification and classify
           yourself relative to your primary-dimension scalar value as upper or
           lower neighbor
    */
    // Classify yourself as an upper or lower valence neighbor to your 1d point
    // Upper = -1, Lower = 1
    // For parity with Paraview, a tie needs to be broken by lower vertex ID being "lower" than the other one
    const int my_class = 1 - (
                          (scalar_values[my_2d] == scalar_values[my_1d] ?
                                my_2d < my_1d :
                                scalar_values[my_2d] < scalar_values[my_1d])
                           << 1);
    valences[tid] = my_class;
    const int min_my_1d = (my_1d * max_VV_guess),
              max_my_1d = min_my_1d + VV_index[my_1d];
    neighborhood[threadIdx.x] = my_2d; // Initialize neighborhood with YOURSELF
    __syncthreads();
    #if PRINT_ON
    printf("Block %d Thread %02d B init on my_1d %d and my_2d %d with valence %d\n", blockIdx.x, threadIdx.x, my_1d, my_2d, my_class);
    #endif
    /*
        3) For all other threads sharing your neighborhood classification, scan
           their connectivity in VV and update your "lowest classifying point"
           to be the lowest IDX (including yourself). Repeat until convergence
           where no edits are made, then if you have your own IDX at your
           location in memory, you log +1 component of your type.
    */
    bool upper_converge = false, lower_converge = false;
    // Repeat until both component directions converge
    while (!(upper_converge && lower_converge)) {
        // Sanity: Guarantee zero-init
        if (threadIdx.x == 0) {
            #if PRINT_ON
            printf("Block %d Iteration %d: upper_edits %u lower_edits %u Neighborhood %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                    blockIdx.x, inspect_step, *upper_edits, *lower_edits,
                    neighborhood[0],
                    neighborhood[1],
                    neighborhood[2],
                    neighborhood[3],
                    neighborhood[4],
                    neighborhood[5],
                    neighborhood[6],
                    neighborhood[7],
                    neighborhood[8],
                    neighborhood[9],
                    neighborhood[10],
                    neighborhood[11],
                    neighborhood[12],
                    neighborhood[13],
                    neighborhood[14],
                    neighborhood[15],
                    neighborhood[16],
                    neighborhood[17],
                    neighborhood[18],
                    neighborhood[19],
                    neighborhood[20],
                    neighborhood[21],
                    neighborhood[22],
                    neighborhood[23],
                    neighborhood[24],
                    neighborhood[25],
                    neighborhood[26],
                    neighborhood[27],
                    neighborhood[28],
                    neighborhood[29],
                    neighborhood[30],
                    neighborhood[31]);
            #endif
            *upper_edits = 0;
            *lower_edits = 0;
        }
        __syncthreads();
        for(int i = min_my_1d; i < max_my_1d; i++) {
            // Found same valence
            const int candidate_component_2d = VV[i];
            if (i != tid && valences[i] == my_class) {
                // Find yourself in their adjacency to become a shared component and release burden
                const int start_2d = candidate_component_2d*max_VV_guess,
                          stop_2d  = start_2d + VV_index[candidate_component_2d];
                for(int j = start_2d; j < stop_2d; j++) {
                    /* Do you see your 2d in their 2d? If so, update your
                       neighborhood with the minimum of your two points
                    */
                    const int theirNeighborhood = neighborhood[i-min_my_1d];
                    if (VV[j] == my_2d && neighborhood[threadIdx.x] > theirNeighborhood) {
                        neighborhood[threadIdx.x] = theirNeighborhood;
                        atomicAdd(component_edits+((my_class+1)/2), 1);
                        /*
                        printf("Block %d Thread %02d Inspect %02d Shares component at VV[%d][%d] (%s)\n",
                                blockIdx.x, threadIdx.x, inspect_step, candidate_component_2d, j-(candidate_component_2d*max_VV_guess),
                                my_class == -1 ? "Upper" : "Lower");
                        */
                    }
                }
            }
        }
        __syncthreads();
        upper_converge = *upper_edits == 0;
        lower_converge = *lower_edits == 0;
    }
    if (neighborhood[threadIdx.x] == my_2d) { // You are the root of a component!
        const int memoffset = ((my_1d*3)+((my_class+1)/2));
        const unsigned int old = atomicAdd(classes+memoffset,1);
        #if PRINT_ON
        printf("Block %d Thread %02d Writes %s component @ mem offset %d (old: %u)\n", blockIdx.x, threadIdx.x, my_class == -1 ? "Upper" : "Lower", memoffset, old);
        #endif
    }
    __syncthreads();
    /*
        4) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
    */
    // Limit classification to lowest-ranked thread for single write
    if (threadIdx.x == 0) {
        const int my_classes = my_1d*3;
        const unsigned int upper = classes[my_classes],
                           lower = classes[my_classes+1];
        #if PRINT_ON
        printf("Block %d Thread %02d Verdict: my_1d (%d) has %u upper and %u lower\n", blockIdx.x, threadIdx.x, my_1d, upper, lower);
        #endif
        if (upper >= 1 && lower == 0) classes[my_classes+2] = MINIMUM_CLASS;
        else if (upper == 0 && lower >= 1) classes[my_classes+2] = MAXIMUM_CLASS;
        else if (/* upper >= 1 and upper == lower / **/ upper == 1 && lower == 1 /**/) classes[my_classes+2] = REGULAR_CLASS;
        else classes[my_classes+2] = SADDLE_CLASS;
    }
}

void unify_mesh_classes(const int * points_per_segment,
			const int max_VV_guess,
			const int ** VV_segments,
			const int ** VV_indices,
			unsigned int * unified_classes,
			#if CONSTRAIN_BLOCK
			#else
			vtkIdType n_classes,
			#endif
			arguments & args) {
	/* Combine multiple kernel results into a single class list for use in export_classes()
	   Mostly expecting to need for duplicate answers on common boundaries between devices
	   to need to be addressed, otherwise it's a simple merge. */
}

void export_classes(unsigned int * classes,
                    #if CONSTRAIN_BLOCK
                    #else
                    vtkIdType n_classes,
                    #endif
                    arguments & args) {
    std::ofstream output_fstream; // Used for file handle to indicated name
    std::streambuf * output_buffer; // Buffer may point to stdout or file handle
    if (args.export_ == "") {
        // No export provided by user, write to stdout
        output_buffer = std::cout.rdbuf();
        std::cerr << WARN_EMOJI << YELLOW_COLOR << "No export file; outputting "
                  "classes to stdout" << RESET_COLOR << std::endl;
    }
    else {
        // Put results in the indicated file
        output_fstream.open(args.export_);
        output_buffer = output_fstream.rdbuf();
        std::cerr << INFO_EMOJI << "Outputting classes to " << args.export_
                                << std::endl;
    }
    // Used for actual file handling
    std::ostream out(output_buffer);
    std::string class_names[] = {"NULL", "min", "max", "regular", "saddle"};
    vtkIdType n_insane = 0;
    #if CONSTRAIN_BLOCK
    for (vtkIdType i = FORCED_BLOCK_IDX; i < FORCED_BLOCK_IDX+1; i++) {
    #else
    for (vtkIdType i = 0; i < n_classes; i++) {
    #endif
        // The classification information is provided, then the class:
        // {# upper, # lower, class}
        // CLASSES = {'minimum': 1, 'maximum': 2, 'regular': 3, 'saddle': 4}
        unsigned int n_upper  = classes[(i*3)],
                     n_lower  = classes[(i*3)+1],
                     my_class = classes[(i*3)+2];
        // Misclassification sanity checks
        if ((n_upper >= 1 && n_lower == 0 && my_class != MINIMUM_CLASS) ||
            (n_upper == 0 && n_lower >= 1 && my_class != MAXIMUM_CLASS) ||
            (n_upper == 1 && n_lower == 1 && my_class != REGULAR_CLASS) ||
            ((n_upper > 1 && n_lower > 1) && my_class != SADDLE_CLASS)) {
            out << "INSANITY DETECTED (" << n_upper << ", " << n_lower << ") FOR POINT " << i << std::endl;
            n_insane++;
        }
        /*
        out << "A Class " << i << " = " << my_class << std::endl;
        */
        out << "A Class " << i << " = " << class_names[my_class] << "(Upper: "
            << n_upper << ", Lower: " << n_lower << ")" << std::endl;
    }
    if (n_insane > 0) {
        std::cerr << WARN_EMOJI << RED_COLOR << "Insanity detected; "
                     "GPU did not agree on its own answers for " << n_insane
                  << " points." << RESET_COLOR << std::endl;
    }
    #ifdef VALIDATE_GPU
    else {
        std::cerr << OK_EMOJI << "No insanity detected in GPU self-agreement "
                     "when classifying points." << std::endl;
    }
    #endif
}

/*
void * critPointsDriver(void *arg) {
   int id = *(int *)arg;
   // cudaStream_t thread_stream;
   //CHECK_CUDA_ERROR(cudaStreamCreate(&thread_stream));
   //CHECK_CUDA_ERROR(cudaStreamSynchronize(thread_stream));
   //CHECK_CUDA_ERROR(cudaMemcpyAsync(device,host,size,direction,thread_stream));
   //CHECK_CUDA_ERROR(cudaStreamSynchronize(thread_stream));
   //KERNEL<<<grid,block,shmem,thread_stream>>>(ARGS);
   int *tids = new int[n_threads];
   pthread_t *threads = new pthread_t[n_threads];
   for (int i = 0; i < n_threads; i++) {
    tids[i] = i;
    pthread_create(&threads[i], NULL, critPointsDriver, &tids[i]);
   }
   for (int i = 0; i < n_threads; i++) {
    pthread_join(threads[i], NULL);
   }
}
*/

int main(int argc, char *argv[]) {
    Timer timer(false, "Main");
    arguments args;
    parse(argc, argv, args);
    timer.tick();
    timer.interval("Argument parsing");

    // Always create a context
    timer.label_next_interval(RED_COLOR "Create CUDA Contexts on all GPUs" RESET_COLOR);
    timer.tick();
    int N_GPUS = 0;
    cudaGetDeviceCount(&N_GPUS);
    if (N_GPUS == 0) {
        std::cout << ERROR_EMOJI << "No GPUs detected!" << std::endl;
        return 1;
    }
    for (int i = 0; i < N_GPUS; i++) {
        cudaSetDevice(i);
        (void)cudaFree(0);
    }
    timer.tick_announce();
    // GPU initialization
    if (! args.validate()) {
        timer.label_next_interval("GPU context creation with dummy kernel");
        timer.tick();
        for (int i = 0; i < N_GPUS; i++) {
	    cudaSetDevice(i);
            KERNEL_WARN(dummy_kernel<<<1 KERNEL_LAUNCH_SEPARATOR 1>>>());
            CUDA_ASSERT(cudaDeviceSynchronize());
        }
        timer.tick_announce();
        timer.label_next_interval("GPU trivial kernel launch");
        timer.tick();
	for (int i = 0; i < N_GPUS; i++) {
	    cudaSetDevice(i);
            KERNEL_WARN(dummy_kernel<<<1 KERNEL_LAUNCH_SEPARATOR 1>>>());
            CUDA_ASSERT(cudaDeviceSynchronize());
	}
        timer.tick_announce();
    }

    // MANDATORY: TV (green) [from storage]
    std::cout << PUSHPIN_EMOJI << "Parsing vtu file: " << args.fileName
              << std::endl;
    timer.label_next_interval(GREEN_COLOR "TV" RESET_COLOR " from VTK");
    timer.tick();
    // Should utilize VTK API and then de-allocate all of its heap
    // Also loads the vertex attributes (host-side) and sets them in
    // TV->vertexAttributes (one scalar per vertex)
    // TODO: Add N_GPUS argument and split data into a list of unique_ptrs to be allocated per device!
    std::unique_ptr<TV_Data> TV = get_TV_from_VTK(args); // args.filename
    timer.tick_announce();


    Timer all_critical_points(false, "ALL Critical Points");
    // Have to make a max VV guess
    timer.label_next_interval("Approximate max VV");
    timer.tick();
    int max_VV_guess = args.max_VV;
    // TODO: Parallelize this across unique_ptrs and get a different(?) max VV or unify max VV for easier coding elsewhere
    if (max_VV_guess < 0) max_VV_guess = get_approx_max_VV(*TV, TV->nPoints);
    timer.tick_announce();

    timer.label_next_interval("Host-async size determination + allocations");
    timer.tick();
    // Size determinations
    // TODO: Need to be made per-device
    size_t tv_flat_size = sizeof(int) * TV->nCells * nbVertsInCell,
           vv_size = sizeof(int) * TV->nPoints * max_VV_guess,
           vv_index_size = sizeof(unsigned int) * TV->nPoints,
           // #Upper, #Lower, Classification
           classes_size = sizeof(unsigned int) * TV->nPoints * 3,
           // Upper/lower per adjacency
           valences_size = sizeof(int) * TV->nPoints * max_VV_guess,
           scalars_size = sizeof(double) * TV->nPoints;
    int n_to_compute = TV->nCells * nbVertsInCell;
    dim3 thread_block_size = 1024,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    const int shared_mem_size = sizeof(unsigned int) * (2+max_VV_guess);

    // Announce expected sizes before possibly running OOM
    std::cout << INFO_EMOJI << "Estimated memory footprint: " << (tv_flat_size+
            classes_size+scalars_size+vv_size+vv_index_size+valences_size) / static_cast<float>(1024*1024*1024) <<
        " GiB" << std::endl;
    std::cout << INFO_EMOJI << "Estimated H->D memory demand: " << (tv_flat_size+
            scalars_size) / static_cast<float>(1024*1024*1024) <<
        " GiB" << std::endl;


    // Allocations
    // in cuda_extraction.cu: int * device_TV = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&device_TV, tv_flat_size));
    int * host_flat_tv = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_flat_tv, tv_flat_size));
    //host_flat_tv = (int*) malloc(tv_flat_size);
    int * vv_computed = nullptr;
    unsigned int * vv_index = nullptr;
    CUDA_ASSERT(cudaMalloc((void**)&vv_computed, vv_size));
    CUDA_ASSERT(cudaMalloc((void**)&vv_index, vv_index_size));
    // CPC = actual critical points classifications
    // valences = adjacency upper/lower classification PRIOR to point classification
    unsigned int *host_CPC = nullptr,
                 *device_CPC = nullptr;
    int *device_valences = nullptr;
    double    *scalar_values = nullptr,
              *device_scalar_values = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_CPC, classes_size));
    //host_CPC = (unsigned int*) malloc(classes_size);
    CUDA_ASSERT(cudaMalloc((void**)&device_CPC, classes_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_valences, valences_size));
    CUDA_ASSERT(cudaMallocHost((void**)&scalar_values, scalars_size));
    //scalar_values = (double*) malloc(scalars_size);
    CUDA_ASSERT(cudaMalloc((void**)&device_scalar_values, scalars_size));

    // These used to be ephemeral in a context
    timer.tick_announce();

    // Usually VE and VF are also mandatory, but CritPoints does not require
    // these relationships! Skip them!

    // OPTIONAL: VV (yellow) [TV' x TV]
    // REQUIRED for CritPoints
    std::cout << PUSHPIN_EMOJI << "Using GPU to compute " YELLOW_COLOR "VV" RESET_COLOR << std::endl;
    timer.label_next_interval(YELLOW_COLOR "VV" RESET_COLOR " [GPU]");
    timer.tick();

    // Set contiguous data in host memory
    int index = 0;
    for (const auto & VertList : (*TV))
        for (const int vertex : VertList)
            host_flat_tv[index++] = vertex;
    // Device copy and host free
    // BLOCKING -- provide barrier if made asynchronous to avoid free of host
    // memory before copy completes
    CUDA_WARN(cudaMemcpyAsync(device_TV, host_flat_tv, tv_flat_size,
                              cudaMemcpyHostToDevice));
    // Pre-populate vv!
    CUDA_WARN(cudaMemset(vv_computed, -1, vv_size));

    // Compute the relationship
    std::cout << INFO_EMOJI << "Kernel launch configuration is " << grid_size.x
              << " grid blocks with " << thread_block_size.x << " threads per block"
              << std::endl;
    std::cout << INFO_EMOJI << "The mesh has " << TV->nCells << " cells and "
              << TV->nPoints << " vertices" << std::endl;
    std::cout << INFO_EMOJI << "Tids >= " << TV->nCells * nbVertsInCell
              << " should auto-exit (" << (thread_block_size.x * grid_size.x) - n_to_compute
              << ")" << std::endl;
    Timer kernel(false, "VV_Kernel");
    KERNEL_WARN(VV_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                            thread_block_size>>>(device_TV,
                                TV->nCells,
                                TV->nPoints,
                                max_VV_guess,
                                vv_index,
                                vv_computed));
    // Things for Critical points to overlap with VV above
    // Pre-populate scalar values
    {
        // Scalar values from VTK
        for(int i = 0; i < TV->nPoints; i++) {
            scalar_values[i] = TV->vertexAttributes[i];
            //std::cerr << "TV value for point " << i << ": " << TV->vertexAttributes[i] << std::endl;
            //std::cout << "A Scalar value for point " << i << ": " << scalar_values[i] << std::endl;
        }
        CUDA_WARN(cudaMemcpyAsync(device_scalar_values, scalar_values, scalars_size, cudaMemcpyHostToDevice));
    }
    CUDA_WARN(cudaDeviceSynchronize());
    kernel.tick();
    kernel.label_prev_interval("GPU kernel duration");
    CUDA_WARN(cudaFree(device_TV));
    // Pack data and return
    device_VV * dvv = new device_VV{vv_computed, vv_index};
    timer.tick_announce();

    // Critical Points
    timer.label_next_interval("Run " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    // Set kernel launch parameters here
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    timer.tick();
    n_to_compute = TV->nPoints * max_VV_guess;
    thread_block_size.x = max_VV_guess;
    grid_size.x = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    #if CONSTRAIN_BLOCK
    grid_size.x = 1;
    std::cerr << EXCLAIM_EMOJI << "DEBUG! Setting kernel size to 1 block (as block " << FORCED_BLOCK_IDX << ")" << std::endl;
    #endif
    KERNEL_WARN(critPoints<<<grid_size KERNEL_LAUNCH_SEPARATOR
                             thread_block_size KERNEL_LAUNCH_SEPARATOR
                             shared_mem_size>>>(dvv->computed,
                                                  dvv->index,
                                                  device_valences,
                                                  TV->nPoints,
                                                  max_VV_guess,
                                                  device_scalar_values,
                                                  device_CPC));
    CUDA_WARN(cudaDeviceSynchronize()); // Make algorithm timing accurate
    timer.tick_announce();
    all_critical_points.tick_announce();
    timer.label_next_interval("Retrieve results from GPU");
    timer.tick();
    CUDA_WARN(cudaMemcpy(host_CPC, device_CPC, classes_size, cudaMemcpyDeviceToHost));
    timer.tick();
    timer.label_next_interval("Export results from " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    timer.tick();
    if (args.export_ == "/dev/null") {
        std::cerr << WARN_EMOJI << "Output to /dev/null omits function call entirely"
                  << std::endl;
    }
    else {
        export_classes(host_CPC,
                       #if CONSTRAIN_BLOCK
                       #else
                       TV->nPoints,
                       #endif
                       args);
    }
    timer.tick_announce();
    timer.label_next_interval("Free memory");
    timer.tick();
    if (host_CPC != nullptr) CUDA_WARN(cudaFreeHost(host_CPC));
    //if (host_CPC != nullptr) free(host_CPC);
    if (device_CPC != nullptr) CUDA_WARN(cudaFree(device_CPC));
    if (device_valences != nullptr) CUDA_WARN(cudaFree(device_valences));
    if (scalar_values != nullptr) CUDA_WARN(cudaFreeHost(scalar_values));
    //if (scalar_values != nullptr) free(scalar_values);
    if (device_scalar_values != nullptr) CUDA_WARN(cudaFree(device_scalar_values));
    if (dvv != nullptr) free(dvv);
    CUDA_WARN(cudaFreeHost(host_flat_tv));
    //free(host_flat_tv);
    timer.tick_announce();
}

