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

#define TID_SELECTION (blockDim.x * blockIdx.x) + threadIdx.x
#define PRINT_ON 0
#define CONSTRAIN_BLOCK 0
// DEBUG: Set grid size to one to only inspect a single block
// BAD (regular -> 1-saddle): 667, 9361, 13566
// BAD (regular -> 2-saddle): 9904, 10748, 10773, 10930, 10984, 10998, 11013, 11076, 11081, 11150, 11158, 11514, 11692, 13013, 13323, 14383
#define FORCED_BLOCK_IDX 9904
/*
#define TID_SELECTION (blockDim.x * FORCED_BLOCK_IDX) + threadIdx.x
#define PRINT_ON 1
#define CONSTRAIN_BLOCK 1
*/


__global__ void critPointsA(const vtkIdType * __restrict__ VV,
                           const unsigned long long * __restrict__ VV_index,
                           vtkIdType * __restrict__ valences,
                           const vtkIdType points,
                           const vtkIdType max_VV_guess,
                           const double * __restrict__ scalar_values,
                           unsigned int * __restrict__ classes) {
    extern __shared__ unsigned long long block_shared[];

    unsigned long long *component_edits = &block_shared[0],
                       *upper_edits = &block_shared[0],
                       *lower_edits = &block_shared[1],
                       *neighborhood = &block_shared[2];;
    const vtkIdType tid = TID_SELECTION;
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
    */
    // Classify yourself as an upper or lower valence neighbor to your 1d point
    // Upper = -1, Lower = 1
    CLASSIFY
    valences[tid] = my_class;
    //printf("Block %d Thread %02d A valence (%lld,%lld) is %lld\n", blockIdx.x, threadIdx.x, my_1d, my_2d, my_class);
/*
}

__global__ void critPointsB(const vtkIdType * __restrict__ VV,
                           const unsigned long long * __restrict__ VV_index,
                           vtkIdType * __restrict__ valences,
                           const vtkIdType points,
                           const vtkIdType max_VV_guess,
                           const double * __restrict__ scalar_values,
                           unsigned int * __restrict__ classes) {
*/
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    const vtkIdType /*tid = TID_SELECTION,
                    my_1d = tid / max_VV_guess,
                    my_2d = VV[tid],
                    */
                    min_my_1d = (my_1d * max_VV_guess),
                    max_my_1d = min_my_1d + VV_index[my_1d];
    /*
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
    */
    neighborhood[threadIdx.x] = my_2d; // Initialize neighborhood with YOURSELF
    __syncthreads();
    #if PRINT_ON
    printf("Block %d Thread %02d B init on my_1d %lld and my_2d %lld with valence %lld\n", blockIdx.x, threadIdx.x, my_1d, my_2d, my_class);
    #endif
    /*
        3) For all other threads sharing your neighborhood classification, scan
           their connectivity in VV and update your "lowest classifying point"
           to be the lowest IDX (including yourself). Repeat until convergence
           where no edits are made, then if you have your own IDX at your
           location in memory, you log +1 component of your type.
    */
    bool upper_converge = false, lower_converge = false;
    int inspect_step = 0;
    // Repeat until both component directions converge
    while (!(upper_converge && lower_converge)) {
        inspect_step++;
        // Sanity: Guarantee zero-init
        if (threadIdx.x == 0) {
            #if PRINT_ON
            printf("Block %d Iteration %d: upper_edits %llu lower_edits %llu Neighborhood %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld\n",
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
        for(vtkIdType i = min_my_1d; i < max_my_1d; i++) {
            // Found same valence
            const vtkIdType candidate_component_2d = VV[i];
            if (i != tid && valences[i] == my_class) {
                /*
                printf("Block %d Thread %02d Inspect %02d Possible shared component with VV[%lld][%lld] (%lld)\n", blockIdx.x, threadIdx.x, inspect_step++, my_1d, i-(my_1d*max_VV_guess), candidate_component_2d);
                */
                // Find yourself in their adjacency to become a shared component and release burden
                const vtkIdType start_2d = candidate_component_2d*max_VV_guess,
                                stop_2d  = start_2d + VV_index[candidate_component_2d];
                for(vtkIdType j = start_2d; j < stop_2d; j++) {
                    /* Do you see your 2d in their 2d? If so, update your
                       neighborhood with the minimum of your two points
                    */
                    const vtkIdType theirNeighborhood = neighborhood[i-min_my_1d];
                    if (VV[j] == my_2d && neighborhood[threadIdx.x] > theirNeighborhood) {
                        neighborhood[threadIdx.x] = theirNeighborhood;
                        atomicAdd(component_edits+((my_class+1)/2), 1);
                        /*
                        printf("Block %d Thread %02d Inspect %02d Shares component at VV[%lld][%lld] (%s)\n",
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
        const vtkIdType memoffset = ((my_1d*3)+((my_class+1)/2));
        const unsigned int old = atomicAdd(classes+memoffset,1);
        #if PRINT_ON
        printf("Block %d Thread %02d Writes %s component @ mem offset %lld (old: %u)\n", blockIdx.x, threadIdx.x, my_class == -1 ? "Upper" : "Lower", memoffset, old);
        #endif
    }
    __syncthreads();
    /*
}

__global__ void critPointsC(const vtkIdType * __restrict__ VV,
                           const unsigned long long * __restrict__ VV_index,
                           vtkIdType * __restrict__ valences,
                           const vtkIdType points,
                           const vtkIdType max_VV_guess,
                           const double * __restrict__ scalar_values,
                           unsigned int * __restrict__ classes) {
    const vtkIdType tid = TID_SELECTION;
    / *
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    * /
    const vtkIdType my_1d = tid / max_VV_guess;
                    //my_2d = VV[tid];
    */
    /*
        4) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
    */
    // Limit classification to lowest-ranked thread for single write
    if (threadIdx.x == 0) {
        const vtkIdType my_classes = my_1d*3;
        const unsigned int upper = classes[my_classes],
                           lower = classes[my_classes+1];
        #if PRINT_ON
        printf("Block %d Thread %02d Verdict: my_1d (%lld) has %u upper and %u lower\n", blockIdx.x, threadIdx.x, my_1d, upper, lower);
        #endif
        if (upper >= 1 && lower == 0) classes[my_classes+2] = MINIMUM_CLASS;
        else if (upper == 0 && lower >= 1) classes[my_classes+2] = MAXIMUM_CLASS;
        else if (/* upper >= 1 and upper == lower / **/ upper == 1 && lower == 1 /**/) classes[my_classes+2] = REGULAR_CLASS;
        else classes[my_classes+2] = SADDLE_CLASS;
    }
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
                                         max_VV_guess, true);
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
        for(unsigned long long i = 0; i < valences_size / sizeof(vtkIdType); i++) {
            valences[i] = 0;
        }
        CUDA_WARN(cudaMemcpy(device_valences, valences, valences_size, cudaMemcpyHostToDevice));

        // Scalar values from VTK
        for(vtkIdType i = 0; i < TV->nPoints; i++) {
            scalar_values[i] = TV->vertexAttributes[i];
            //std::cerr << "TV value for point " << i << ": " << TV->vertexAttributes[i] << std::endl;
            //std::cout << "A Scalar value for point " << i << ": " << scalar_values[i] << std::endl;
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
    const vtkIdType n_to_compute = TV->nPoints * max_VV_guess,
                    shared_mem_size = sizeof(unsigned long long) * (2+max_VV_guess);
    dim3 thread_block_size = max_VV_guess,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    #if CONSTRAIN_BLOCK
    grid_size.x = 1;
    std::cerr << EXCLAIM_EMOJI << "DEBUG! Setting kernel size to 1 block (as block " << FORCED_BLOCK_IDX << ")" << std::endl;
    #endif
    timer.tick();
    KERNEL_WARN(critPointsA<<<grid_size KERNEL_LAUNCH_SEPARATOR
                             thread_block_size KERNEL_LAUNCH_SEPARATOR
                             shared_mem_size>>>(dvv->computed,
                                                  dvv->index,
                                                  device_valences,
                                                  TV->nPoints,
                                                  max_VV_guess,
                                                  device_scalar_values,
                                                  device_CPC));
    /*
    KERNEL_WARN(critPointsB<<<grid_size KERNEL_LAUNCH_SEPARATOR
                             thread_block_size KERNEL_LAUNCH_SEPARATOR
                             shared_mem_size>>>(dvv->computed,
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
    */
    CUDA_WARN(cudaDeviceSynchronize()); // Make algorithm timing accurate
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
    export_classes(host_CPC,
                   #if CONSTRAIN_BLOCK
                   #else
                   TV->nPoints,
                   #endif
                   args);
    timer.tick_announce();
    if (host_CPC != nullptr) CUDA_WARN(cudaFreeHost(host_CPC));
    if (device_CPC != nullptr) CUDA_WARN(cudaFree(device_CPC));
    if (valences != nullptr) CUDA_WARN(cudaFreeHost(valences));
    if (device_valences != nullptr) CUDA_WARN(cudaFree(device_valences));
    if (scalar_values != nullptr) CUDA_WARN(cudaFreeHost(scalar_values));
    if (device_scalar_values != nullptr) CUDA_WARN(cudaFree(device_scalar_values));
    if (dvv != nullptr) free(dvv);
}

