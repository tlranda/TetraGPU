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

/* CriticalPoints kernel should:
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
        2) Read the scalar value used for point classification and classify
           yourself relative to your primary-dimension scalar value as upper or
           lower neighbor
        -- VV-PARALLEL SYNC REQUIRED --
        3) For all other threads sharing your neighborhood classification, scan
           their connectivity in VV. If you connect to at least one, then you
           share a component with that neighbor -- the lowest-ranked neighbor
           will log +1 component of this type and all others exit. If you fail
           to locate any connections to others in your class, then you have 2+
           components and are immediately a saddle -- increment your component
           counter and exit. It does not matter if this "over-counts" the
           number of components!
        -- VV-PARALLEL SYNC REQUIRED --
        4) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
*/

// For parity with Paraview, a tie needs to be broken by lower vertex ID being "lower" than the other one
#define CLASSIFY const vtkIdType my_class = 1 - (\
        (scalar_values[my_2d] == scalar_values[my_1d] ? \
         my_2d < my_1d : \
         scalar_values[my_2d] < scalar_values[my_1d]) \
        << 1);
#define MAXIMUM_CLASS 1
#define MINIMUM_CLASS 2
#define REGULAR_CLASS 3
#define SADDLE_CLASS  4

#define FORCED_BLOCK_IDX 1

__global__ void critPointsA(const vtkIdType * __restrict__ VV,
                           const unsigned long long * __restrict__ VV_index,
                           vtkIdType * __restrict__ valences,
                           const vtkIdType points,
                           const vtkIdType max_VV_guess,
                           const double * __restrict__ scalar_values,
                           unsigned int * __restrict__ classes) {
    const vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    //const vtkIdType tid = (blockDim.x * FORCED_BLOCK_IDX) + threadIdx.x;
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    const vtkIdType my_1d = tid / max_VV_guess,
                    my_2d = VV[tid];
    // No work for this point's valence
    if (VV_index[my_1d] <= 0 || my_2d < 0) return;
    // Prefix scan as anti-duplication
    for (vtkIdType i = my_1d * max_VV_guess; i < tid; i++) {
        if (VV[i] == my_2d) return;
    }

    // BEYOND THIS POINT, YOU ARE AN ACTUAL WORKER THREAD ON THE PROBLEM

    /*
        2) Read the scalar value used for point classification and classify
           yourself relative to your primary-dimension scalar value as upper or
           lower neighbor
        -- VV-PARALLEL SYNC REQUIRED --
    */
    // Classify yourself as an upper or lower valence neighbor to your 1d point
    // Upper = -1, Lower = 1
    //vtkIdType my_class = 1 - ((scalar_values[my_2d] (>= / <) scalar_values[my_1d])<<1);
    CLASSIFY
    valences[tid] = my_class;
    //printf("Block %d Thread %02d A valence (%lld,%lld) is %lld\n", blockIdx.x, threadIdx.x, my_1d, my_2d, my_class);
    //__syncthreads();
}

__global__ void critPointsB(const vtkIdType * __restrict__ VV,
                           const unsigned long long * __restrict__ VV_index,
                           vtkIdType * __restrict__ valences,
                           const vtkIdType points,
                           const vtkIdType max_VV_guess,
                           const double * __restrict__ scalar_values,
                           unsigned int * __restrict__ classes) {
    const vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    //bool printing = blockIdx.x == FORCED_BLOCK_IDX;
    //const vtkIdType tid = (blockDim.x * FORCED_BLOCK_IDX) + threadIdx.x;
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    const vtkIdType my_1d = tid / max_VV_guess,
              my_2d = VV[tid];
    // No work for this point's valence
    if (VV_index[my_1d] <= 0 || my_2d < 0) return;
    //{ printf("Block %d Thread %02d early exit: %s\n", blockIdx.x, threadIdx.x, "No VV_index work"); return; }
    // Prefix scan as anti-duplication
    for (vtkIdType i = my_1d * max_VV_guess; i < tid; i++) {
        if (VV[i] == my_2d) return;
        //{ printf("Block %d Thread %02d early exit: %s %d\n", blockIdx.x, threadIdx.x, "Prefix duplicate found @ index", i-(my_1d*max_VV_guess)); return; }
    }

    // BEYOND THIS POINT, YOU ARE AN ACTUAL WORKER THREAD ON THE PROBLEM
    // Which way should the tiebreaker go here?
    const vtkIdType my_class = valences[tid];
    /*
    if (printing)
    printf("Block %d Thread %02d B init on my_1d %lld and my_2d %lld with valence %lld\n", blockIdx.x, threadIdx.x, my_1d, my_2d, my_class);
    */
    /*
        3) For all other threads sharing your neighborhood classification, scan
           their connectivity in VV. If you connect to at least one, then you
           share a component with that neighbor -- the lowest-ranked neighbor
           will log +1 component of this type and all others exit. If you fail
           to locate any connections to others in your class, then you have 2+
           components and are immediately a saddle -- increment your component
           counter and exit. It does not matter if this "over-counts" the
           number of components!
        -- VV-PARALLEL SYNC REQUIRED --
    */
    const vtkIdType max_my_1d = (my_1d * max_VV_guess) + VV_index[my_1d];
    /*
    if (printing)
    printf("Block %d Thread %02d has max 1d range %lld\n", blockIdx.x, threadIdx.x, max_my_1d-(my_1d*max_VV_guess));
    */
    bool done = false, burdened = true;
    int inspect_step = 0;
    // MISSING: Burden of lowest check! If you are the lowest to have your valence class, you ALWAYS log yourself as a component
    // PRACTICAL: Should only need to scan elements sharing your 1D up until yourself
    for(vtkIdType i = my_1d * max_VV_guess; !done && (i < max_my_1d); i++) {
        // Found same valence connected to 1D at a point with a lower index than you
        const vtkIdType candidate_component_2d = VV[i]; //[(my_1d*max_VV_guess)+(i-(my_1d*max_VV_guess))];
        if (valences[i] == my_class && candidate_component_2d < my_2d) {
            /*
            if (printing)
            printf("Block %d Thread %02d Inspect %02d Possible shared component with VV[%lld][%lld] (%lld)\n", blockIdx.x, threadIdx.x, inspect_step++, my_1d, i-(my_1d*max_VV_guess), candidate_component_2d);
            */
            // Find yourself in their adjacency to become a shared component and release burden
            const vtkIdType start_2d = candidate_component_2d*max_VV_guess,
                            stop_2d  = start_2d + VV_index[candidate_component_2d];
            for(vtkIdType j = start_2d; !done && (j < stop_2d); j++) {
                // Do you see your 2d in their 2d?
                if (VV[j] == my_2d) {
                    // Shared component!
                    burdened = (candidate_component_2d > my_2d); // lower one writes!
                    //done = burdened; // EXPERIMENTAL: Only done if you're burdened
                    /*
                    if (printing)
                    printf("Block %d Thread %02d Inspect %02d Shares component at VV[%lld][%lld] (%s, %s)\n",
                            blockIdx.x, threadIdx.x, inspect_step, candidate_component_2d, j-(candidate_component_2d*max_VV_guess),
                            my_class == -1 ? "Upper" : "Lower", burdened ? "Burdened to write" : "Not writing");
                    */
                }
                inspect_step++;
            }
        }
    }
    __syncthreads();
    if (burdened) {
        const vtkIdType memoffset = ((my_1d*3)+((my_class+1)/2));
        const unsigned int old = atomicAdd(classes+memoffset,1);
        /*
        if (printing)
        printf("Block %d Thread %02d Writes %s component @ mem offset %lld (old: %u)\n", blockIdx.x, threadIdx.x, my_class == -1 ? "Upper" : "Lower", memoffset, old);
        */
    }
    //__syncthreads();
}

__global__ void critPointsC(const vtkIdType * __restrict__ VV,
                           const unsigned long long * __restrict__ VV_index,
                           vtkIdType * __restrict__ valences,
                           const vtkIdType points,
                           const vtkIdType max_VV_guess,
                           const double * __restrict__ scalar_values,
                           unsigned int * __restrict__ classes) {
    const vtkIdType tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    //bool printing = blockIdx.x == FORCED_BLOCK_IDX;
    //const vtkIdType tid = (blockDim.x * FORCED_BLOCK_IDX) + threadIdx.x;
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    const vtkIdType my_1d = tid / max_VV_guess;
                    //my_2d = VV[tid];
    /*
        4) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
    */
    // Limit classification to lowest-ranked thread for single write
    if (my_1d * max_VV_guess == tid) {
        const vtkIdType my_classes = my_1d*3;
        const unsigned int upper = classes[my_classes],
                           lower = classes[my_classes+1];
        /*
        if (printing)
        printf("Block %d Thread %02d Verdict: my_1d (%lld) has %u upper and %u lower\n", blockIdx.x, threadIdx.x, my_1d, upper, lower);
        */
        if (upper >= 1 && lower == 0) classes[my_classes+2] = MINIMUM_CLASS;
        else if (upper == 0 && lower >= 1) classes[my_classes+2] = MAXIMUM_CLASS;
        else if (/* upper >= 1 and upper == lower /**/ upper == 1 && lower == 1 /**/) classes[my_classes+2] = REGULAR_CLASS;
        else classes[my_classes+2] = SADDLE_CLASS;
    }
}

void export_classes(unsigned int * classes, vtkIdType n_classes, arguments & args) {
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
    for (vtkIdType i = 0; i < n_classes; i++) {
    //for (vtkIdType i = FORCED_BLOCK_IDX; i < FORCED_BLOCK_IDX+1; i++) {
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

int main(int argc, char *argv[]) {
    Timer timer(false, "Main");
    arguments args;
    parse(argc, argv, args);
    timer.tick();
    timer.interval("Argument parsing");

    // GPU initialization
    if (! args.validate()) {
        timer.label_next_interval("GPU context creation with dummy kernel");
        timer.tick();
        KERNEL_WARN(dummy_kernel<<<1 KERNEL_LAUNCH_SEPARATOR 1>>>());
        CUDA_ASSERT(cudaDeviceSynchronize());
        timer.tick_announce();
        timer.label_next_interval("GPU trivial kernel launch");
        timer.tick();
        KERNEL_WARN(dummy_kernel<<<1 KERNEL_LAUNCH_SEPARATOR 1>>>());
        CUDA_ASSERT(cudaDeviceSynchronize());
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
    std::unique_ptr<TV_Data> TV = get_TV_from_VTK(args); // args.filename
    timer.tick_announce();

    // Usually VE and VF are also mandatory, but CritPoints does not require
    // these relationships! Skip them!

    // OPTIONAL: VV (yellow) [TV' x TV]
    // REQUIRED for CritPoints
    std::cout << PUSHPIN_EMOJI << "Using GPU to compute " YELLOW_COLOR "VV" RESET_COLOR << std::endl;
    timer.label_next_interval(YELLOW_COLOR "VV" RESET_COLOR " [GPU]");
    timer.tick();
    // Have to make a max VV guess
    vtkIdType max_VV_guess = get_approx_max_VV(*TV, TV->nPoints);
    device_VV * dvv = make_VV_GPU_return(*TV, TV->nCells, TV->nPoints,
                                         max_VV_guess, true, args); // Args not used, actually
    timer.tick_announce();

    // Critical Points
    timer.label_next_interval("Allocate " CYAN_COLOR "Critical Points" RESET_COLOR " memory");
    timer.tick();
    // CPC = actual critical points classifications
    // valences = adjacency upper/lower classification PRIOR to point classification
    unsigned int *host_CPC = nullptr,
                 *device_CPC = nullptr;
    vtkIdType *device_valences = nullptr;
    double    *scalar_values = nullptr,
              *device_scalar_values = nullptr;
    // #Upper, #Lower, Classification
    size_t classes_size = sizeof(unsigned int) * TV->nPoints * 3,
           // Upper/lower per adjacency
           valences_size = sizeof(vtkIdType) * TV->nPoints * max_VV_guess,
           scalars_size = sizeof(double) * TV->nPoints;
    CUDA_ASSERT(cudaMallocHost((void**)&host_CPC, classes_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_CPC, classes_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_valences, valences_size));
    CUDA_ASSERT(cudaMallocHost((void**)&scalar_values, scalars_size));
    CUDA_ASSERT(cudaMalloc((void**)&device_scalar_values, scalars_size));
    // Pre-populate valences as zeros and populate scalar values
    vtkIdType * valences = nullptr;
    {
        CUDA_ASSERT(cudaMallocHost((void**)&valences, valences_size));
        for(vtkIdType i = 0; i < valences_size / sizeof(vtkIdType); i++) {
            valences[i] = 0;
        }
        CUDA_WARN(cudaMemcpy(device_valences, valences, valences_size, cudaMemcpyHostToDevice));

        // Scalar values from VTK
        for(vtkIdType i = 0; i < TV->nPoints; i++) {
            scalar_values[i] = TV->vertexAttributes[i];
            //std::cerr << "TV value for point " << i << ": " << TV->vertexAttributes[i] << std::endl;
            std::cout << "A Scalar value for point " << i << ": " << scalar_values[i] << std::endl;
        }
        CUDA_WARN(cudaMemcpy(device_scalar_values, scalar_values, scalars_size, cudaMemcpyHostToDevice));

        // Class values
        for (vtkIdType i = 0; i < TV->nPoints * 3; i++) {
            host_CPC[i] = 0;
        }
        CUDA_WARN(cudaMemcpy(device_CPC, host_CPC, classes_size, cudaMemcpyHostToDevice));
    }
    timer.tick_announce();
    timer.label_next_interval("Run " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    // Set kernel launch parameters here
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    const vtkIdType n_to_compute = TV->nPoints * max_VV_guess;
    dim3 thread_block_size = max_VV_guess,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    // DEBUG: Set grid size to one to only inspect a single block
    //grid_size.x = 1;
    //std::cerr << EXCLAIM_EMOJI << "DEBUG! Setting kernel size to 1 block (as block " << FORCED_BLOCK_IDX << ")" << std::endl;
    timer.tick();
    KERNEL_WARN(critPointsA<<<grid_size KERNEL_LAUNCH_SEPARATOR
                             thread_block_size>>>(dvv->computed,
                                                  dvv->index,
                                                  device_valences,
                                                  TV->nPoints,
                                                  max_VV_guess,
                                                  device_scalar_values,
                                                  device_CPC));
    KERNEL_WARN(critPointsB<<<grid_size KERNEL_LAUNCH_SEPARATOR
                             thread_block_size>>>(dvv->computed,
                                                  dvv->index,
                                                  device_valences,
                                                  TV->nPoints,
                                                  max_VV_guess,
                                                  device_scalar_values,
                                                  device_CPC));
    KERNEL_WARN(critPointsC<<<grid_size KERNEL_LAUNCH_SEPARATOR
                             thread_block_size>>>(dvv->computed,
                                                  dvv->index,
                                                  device_valences,
                                                  TV->nPoints,
                                                  max_VV_guess,
                                                  device_scalar_values,
                                                  device_CPC));
    timer.tick_announce();
    timer.label_next_interval("Retrieve results from GPU");
    timer.tick();
    CUDA_WARN(cudaMemcpy(host_CPC, device_CPC, classes_size, cudaMemcpyDeviceToHost));
    CUDA_WARN(cudaMemcpy(valences, device_valences, valences_size, cudaMemcpyDeviceToHost));
    timer.tick();
    timer.label_next_interval("Export results from " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    /* DEBUG CHECK VALENCES
    for (int i = 0; i < TV->nPoints; i++) {
        for (int j = 0; j < max_VV_guess; j++) {
            std::cerr << "Valence point " << i << " entry " << j << ": "
                      << valences[(i*max_VV_guess)+j] << std::endl;
        }
    }
    */
    timer.tick();
    export_classes(host_CPC, TV->nPoints, args);
    timer.tick_announce();
    if (host_CPC != nullptr) CUDA_WARN(cudaFreeHost(host_CPC));
    if (device_CPC != nullptr) CUDA_WARN(cudaFree(device_CPC));
    if (valences != nullptr) CUDA_WARN(cudaFreeHost(valences));
    if (device_valences != nullptr) CUDA_WARN(cudaFree(device_valences));
    if (scalar_values != nullptr) CUDA_WARN(cudaFreeHost(scalar_values));
    if (device_scalar_values != nullptr) CUDA_WARN(cudaFree(device_scalar_values));
    if (dvv != nullptr) free(dvv);
}

