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
// Debugging specific block executions
/*
#define FORCED_BLOCK_IDX 13
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
                           const unsigned int * __restrict__ partition,
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
    // No work for this point's valence or out of partition
    if (VV_index[my_1d] <= 0 || my_2d < 0 || partition[my_1d] == 0) return;
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
    printf("Block %d Thread %02d B init on my_1d %d and my_2d %d with valence %d based on %lf (block) vs %lf (thread)\n", blockIdx.x, threadIdx.x, my_1d, my_2d, my_class, scalar_values[my_1d], scalar_values[my_2d]);
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
    int to_converge = 0;
    while (!(upper_converge && lower_converge)) {
        // Sanity: Guarantee zero-init
        if (threadIdx.x == 0) {
            #if PRINT_ON
            printf("Block %d Iteration %d: upper_edits %u lower_edits %u Neighborhood %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                    blockIdx.x, /*inspect_step*/0, *upper_edits, *lower_edits,
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
        to_converge++;
        __syncthreads();

        // Union-Find iteration
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
    // End Union-Find

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
        if (blockIdx.x < 4) {
            printf("Block %02d took %d loops to converge\n", blockIdx.x, to_converge);
        }
        printf("Block %d Thread %02d Verdict: my_1d (%d) has %u upper and %u lower\n", blockIdx.x, threadIdx.x, my_1d, upper, lower);
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
                    runtime_arguments & args) {
    // VOIDS can be ignored during aggregation, should all be eliminated after all GPU kernels have returned!
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
    vtkIdType n_insane = 0, n_min = 0, n_max = 0, n_saddle = 0, n_void = 0;
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
            out << "INSANITY DETECTED (" << n_upper << ", " << n_lower << ", " << my_class << ") FOR POINT " << i << " (Given class "
                << (my_class == 1 ? "Maximum" : (my_class == 2 ? "Minimum" : (my_class == 3 ? "Regular" : "Saddle")))
                << ")" << std::endl;
            n_insane++;
        }
        /*
        out << "A Class " << i << " = " << my_class << std::endl;
        out << "A Class " << i << " = " << class_names[my_class] << "(Upper: "
            << n_upper << ", Lower: " << n_lower << ")" << std::endl;
        */
        if (my_class == MAXIMUM_CLASS) {
            n_max++;
            //std::cerr << "GREPME Maximum point: " << i << std::endl;
        }
        else if (my_class == MINIMUM_CLASS) {
            n_min++;
        }
        else if (my_class == SADDLE_CLASS) {
            n_saddle++;
        }
        else if (my_class == 0) {
            n_void++;
        }
    }
    if (n_insane > 0) {
        std::cerr << WARN_EMOJI << RED_COLOR << "Insanity detected; "
                     "GPU did not agree on its own answers for " << n_insane
                  << " points." << RESET_COLOR << std::endl;
    }
    std::cout << "Number of minima: " << n_min << std::endl;
    std::cout << "Number of maxima: " << n_max << std::endl;
    std::cout << "Number of saddles: " << n_saddle << std::endl;
    std::cout << "Number of voids: " << n_void << std::endl;
    #ifdef VALIDATE_GPU
    else {
        std::cerr << OK_EMOJI << "No insanity detected in GPU self-agreement "
                     "when classifying points." << std::endl;
    }
    #endif
}

struct thread_arguments {
    int gpu_id;
    std::unique_ptr<TV_Data> TV;
    int * host_flat_tv;
    int * device_tv;
    size_t tv_flat_size;
    int * vv_computed;
    unsigned int * vv_index;
    size_t vv_size;
    size_t vv_index_size;
    double * scalar_values;
    double * device_scalar_values;
    size_t scalars_size;
    int * device_valences;
    size_t valences_size;
    unsigned int * partition_ids;
    size_t partition_ids_size;
    unsigned int * device_CPCs;
    unsigned int * host_CPCs;
    size_t classes_size;
    int n_to_compute;
    int max_VV_guess;
    dim3 thread_block_size;
    dim3 grid_size;
    int shared_mem_size;
    runtime_arguments args;

    thread_arguments(void) { }
    thread_arguments(
        int gpu_id1,
        std::unique_ptr<TV_Data> TV1,
        int * host_flat_tv1,
        int * device_tv1,
        size_t tv_flat_size1,
        int * vv_computed1,
        unsigned int * vv_index1,
        size_t vv_size1,
        size_t vv_index_size1,
        double * scalar_values1,
        double * device_scalar_values1,
        size_t scalars_size1,
        int * device_valences1,
        size_t valences_size1,
        unsigned int * partition_ids1,
        size_t partition_ids_size1,
        unsigned int * device_CPCs1,
        unsigned int * host_CPCs1,
        size_t classes_size1,
        int n_to_compute1,
        int max_VV_guess1,
        dim3 thread_block_size1,
        dim3 grid_size1,
        int shared_mem_size1,
        runtime_arguments args1) {
            gpu_id = gpu_id1;
            TV = std::move(TV1);
            host_flat_tv = host_flat_tv1;
            device_tv = device_tv1;
            tv_flat_size = tv_flat_size1;
            vv_computed = vv_computed1;
            vv_index = vv_index1;
            vv_size = vv_size1;
            vv_index_size = vv_index_size1;
            scalar_values = scalar_values1;
            device_scalar_values = device_scalar_values1;
            scalars_size = scalars_size1;
            device_valences = device_valences1;
            valences_size = valences_size1;
            partition_ids = partition_ids1;
            partition_ids_size = partition_ids_size1;
            device_CPCs = device_CPCs1;
            host_CPCs = host_CPCs1;
            classes_size = classes_size1;
            n_to_compute = n_to_compute1;
            max_VV_guess = max_VV_guess1;
            thread_block_size = thread_block_size1;
            grid_size = grid_size1;
            shared_mem_size = shared_mem_size1;
            args = args1;
    }
};
void * parallel_work(void *parallel_arguments) {
    // Unpacking
    thread_arguments *thread_args = (thread_arguments *)parallel_arguments;
    int gpu_id = thread_args->gpu_id;
    const std::unique_ptr<TV_Data> TV = std::move(thread_args->TV);
    int * host_flat_tv = thread_args->host_flat_tv;
    int * device_tv = thread_args->device_tv;
    const size_t tv_flat_size = thread_args->tv_flat_size;
    int * vv_computed = thread_args->vv_computed;
    unsigned int * vv_index = thread_args->vv_index;
    const size_t vv_size = thread_args->vv_size;
    const size_t vv_index_size = thread_args->vv_index_size;
    double * scalar_values = thread_args->scalar_values;
    double * device_scalar_values = thread_args->device_scalar_values;
    const size_t scalars_size = thread_args->scalars_size;
    int * device_valences = thread_args->device_valences;
    const size_t valences_size = thread_args->valences_size;
    unsigned int * partition_ids = thread_args->partition_ids;
    const size_t partition_ids_size = thread_args->partition_ids_size;
    unsigned int * device_CPCs = thread_args->device_CPCs;
    // host_CPCs is the return value! (or part of it, depending on implementation)
    unsigned int * host_CPCs = thread_args->host_CPCs;
    const size_t classes_size = thread_args->classes_size;
    int n_to_compute = thread_args->n_to_compute;
    const int max_VV_guess = thread_args->max_VV_guess;
    const int shared_mem_size = thread_args->shared_mem_size;
    runtime_arguments _my_args = thread_args->args;
    dim3 thread_block_size = thread_args->thread_block_size;
    dim3 grid_size = thread_args->grid_size;

    char timername[32];
    sprintf(timername, "Parallel worker %02d", gpu_id);
    Timer timer(false, timername);
    // TEMPORARY: Need to know actual GPUs if oversubscribing
    int actual_gpus, vgpu_id;
    CUDA_WARN(cudaGetDeviceCount(&actual_gpus));
    vgpu_id = (gpu_id % actual_gpus);
    cudaSetDevice(vgpu_id);
    // OPTIONAL: VV (yellow) [TV' x TV]
    // REQUIRED for CritPoints
    std::cout << PUSHPIN_EMOJI << "Using GPU " << gpu_id << " (Actual GPU ID: "
              << vgpu_id << ") to compute " YELLOW_COLOR "VV" RESET_COLOR
              << std::endl;
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
    CUDA_WARN(cudaMemcpyAsync(device_tv, host_flat_tv, tv_flat_size,
                              cudaMemcpyHostToDevice));
    // Pre-populate vv!
    CUDA_WARN(cudaMemset(vv_computed, -1, vv_size));

    // Compute the relationship
    std::cout << INFO_EMOJI << timername << " Kernel launch configuration is "
              << grid_size.x << " grid blocks with " << thread_block_size.x
              << " threads per block" << std::endl;
    std::cout << INFO_EMOJI << timername << " The mesh has " << TV->nCells
              << " cells and " << TV->nPoints << " vertices" << std::endl;
    std::cout << INFO_EMOJI << timername << " Tids >= " << TV->nCells * nbVertsInCell
              << " should auto-exit (" << (thread_block_size.x * grid_size.x) - n_to_compute
              << ")" << std::endl;
    char kerneltimername[32];
    sprintf(kerneltimername, "Parallel kernel %02d", gpu_id);
    Timer kernel(false, kerneltimername);
    KERNEL_WARN(VV_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                            thread_block_size>>>(device_tv,
                                TV->nCells,
                                TV->nPoints,
                                max_VV_guess,
                                vv_index,
                                vv_computed));
    // Things for Critical points to overlap with VV above
    // Pre-populate scalar values
    {
        // Scalar values from VTK
        for(int j = 0; j < TV->nPoints; j++) {
            scalar_values[j] = TV->vertexAttributes[j];
            //std::cerr << "TV value for point " << j << ": " << TV->vertexAttributes[j] << std::endl;
            //std::cout << "A Scalar value for point " << j << ": " << scalar_values[j] << std::endl;
        }
        CUDA_WARN(cudaMemcpyAsync(device_scalar_values, scalar_values, scalars_size, cudaMemcpyHostToDevice));
    }
    CUDA_WARN(cudaDeviceSynchronize());
    kernel.tick();
    kernel.label_prev_interval("GPU kernel duration");
    CUDA_WARN(cudaFree(device_tv));
    // Pack data and return
    device_VV * dvv = new device_VV{vv_computed, vv_index};
    timer.tick_announce();

    // DEBUG FOR GALE INTEGRATION: Check VV ON HOST A BIT
    /*
    int * host_vv = nullptr;
    unsigned int * host_vv_index = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv, vv_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv_index, vv_index_size));
    CUDA_WARN(cudaMemcpy(host_vv, vv_computed, vv_size, cudaMemcpyDeviceToHost));
    CUDA_WARN(cudaMemcpy(host_vv_index, vv_index, vv_index_size, cudaMemcpyDeviceToHost));
    int MAX_PRINT = 100;
    for (int i = 0; i < 4; i++) {
        std::cout << "Sanity check VV[" << i << "] with size " << host_vv_index[i] << std::endl;
        int consecutive_minus_1 = 0;
        for (int j = 0; j < max_VV_guess && MAX_PRINT > 0; j++) {
            if (host_vv[(i*max_VV_guess)+j] == -1) {
                consecutive_minus_1++;
                continue;
            }
            else if (consecutive_minus_1 > 0) {
                std::cout << "\t" << i << ": -1 (" << consecutive_minus_1 << " times)" << std::endl;
            }
            std::cout << "\t" << i << ": " << host_vv[(i*max_VV_guess)+j] << std::endl;
            MAX_PRINT--;
        }
        if (consecutive_minus_1 > 0) {
            std::cout << "\t" << i << ": (possibly not complete: " << consecutive_minus_1 << " consecutive leftover -1's)" << std::endl;
        }
    }
    CUDA_ASSERT(cudaFreeHost(host_vv));
    CUDA_ASSERT(cudaFreeHost(host_vv_index));
    */
    // DEBUG: Check lengths for threats to correctness / efficiency
    timer.label_next_interval("Check for VV duplication / error");
    timer.tick();
    unsigned int * host_vv_index = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv_index, vv_index_size));
    CUDA_WARN(cudaMemcpy(host_vv_index, vv_index, vv_index_size, cudaMemcpyDeviceToHost));
    unsigned int duplicates = 0, minmax_size = 0, actual_size = 0;
    std::vector<unsigned int> overflow = std::vector<unsigned int>();
    int * host_vv = nullptr;
    // Determine the real size of de-duplicated VV
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv, vv_size));
    CUDA_WARN(cudaMemcpy(host_vv, vv_computed, vv_size, cudaMemcpyDeviceToHost));
    std::vector<unsigned int> deduped_length(TV->nPoints);
    std::vector<int> known_points(max_VV_guess);
    unsigned int max_maybe_duped_degree = 0, max_actual_degree = 0;
    for (int j = 0; j < TV->nPoints; j++) {
        known_points.clear();
        actual_size += host_vv_index[j];
        if (host_vv_index[j] > static_cast<unsigned int>(max_VV_guess)) {
            overflow.emplace_back(host_vv_index[j]-max_VV_guess);
        }
        if (host_vv_index[j] > max_maybe_duped_degree) {
            max_maybe_duped_degree = host_vv_index[j];
        }
        for (unsigned int k = 0; k < host_vv_index[j]; k++) {
            if (std::find(known_points.begin(), known_points.end(), host_vv[(j*max_VV_guess+k)]) == known_points.end()) {
                known_points.emplace_back(host_vv[(j*max_VV_guess)+k]);
            }
            else {
                duplicates++;
            }
        }
        minmax_size += known_points.size();
        if (known_points.size() > max_actual_degree) {
            max_actual_degree = known_points.size();
        }
    }
    if (overflow.size() > 0) {
        std::cerr << WARN_EMOJI << "Detected " << overflow.size() << " points in VV that overflow the maximum adjacency limit!" << std::endl;
        std::cerr << WARN_EMOJI << "Overflow amounts: ";
        for (auto j : overflow) {
            std::cerr << j << ", ";
        }
        std::cerr << std::endl;
    }
    else {
        std::cerr << OK_EMOJI << "No overflow of VV detected" << std::endl;
    }
    if (duplicates > 0) {
        std::cerr << WARN_EMOJI << "Detected " << duplicates << " VV duplicates (" << actual_size << " total entries, minimum count: " << minmax_size << " -- " << (actual_size/static_cast<float>(minmax_size))-1.0 << " proportionate extra)" << std::endl;
    }
    else {
        std::cerr << OK_EMOJI << "No VV duplication detected. Minimum count: " << minmax_size << std::endl;
    }
    std::cerr << INFO_EMOJI << "Max VV was set to " << max_VV_guess << ". ACTUAL max VV is " << max_actual_degree << ", max utilized VV is " << max_maybe_duped_degree << std::endl;
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
    std::cout << timername << " Launch critPoints kernel with " << n_to_compute << " units of work (" << grid_size.x << " blocks, " << thread_block_size.x << " threads per block)" << std::endl;
    KERNEL_WARN(critPoints<<<grid_size KERNEL_LAUNCH_SEPARATOR
                             thread_block_size KERNEL_LAUNCH_SEPARATOR
                             shared_mem_size>>>(dvv->computed,
                                                  dvv->index,
                                                  device_valences,
                                                  TV->nPoints,
                                                  max_VV_guess,
                                                  device_scalar_values,
                                                  partition_ids,
                                                  device_CPCs));
    CUDA_WARN(cudaDeviceSynchronize()); // Make algorithm timing accurate
    timer.tick_announce();
    timer.label_next_interval("Retrieve results from GPU");
    timer.tick();
    CUDA_WARN(cudaMemcpy(host_CPCs, device_CPCs, classes_size, cudaMemcpyDeviceToHost));
    timer.tick();
    /*
    timer.label_next_interval("Export results from " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    timer.tick();
    if (_my_args.export_ == "/dev/null") {
        std::cerr << WARN_EMOJI << "Output to /dev/null omits function call entirely"
                  << std::endl;
    }
    else {
        export_classes(host_CPCs,
                       #if CONSTRAIN_BLOCK
                       #else
                       TV->nPoints,
                       #endif
                       _my_args);
    }
    timer.tick_announce();
    */
    timer.label_next_interval("Free memory");
    timer.tick();
    if (dvv != nullptr) free(dvv);
    if (partition_ids != nullptr) CUDA_WARN(cudaFree(partition_ids));
    if (device_CPCs != nullptr) CUDA_WARN(cudaFree(device_CPCs));
    if (device_valences != nullptr) CUDA_WARN(cudaFree(device_valences));
    if (scalar_values != nullptr) CUDA_WARN(cudaFreeHost(scalar_values));
    if (device_scalar_values != nullptr) CUDA_WARN(cudaFree(device_scalar_values));
    //if (host_CPCs != nullptr) CUDA_WARN(cudaFreeHost(host_CPCs));
    CUDA_WARN(cudaFreeHost(host_flat_tv));
    //TVs[i].release();
    timer.tick_announce();

    return (void*)host_CPCs;
}

int main(int argc, char *argv[]) {
    Timer timer(false, "Main");
    runtime_arguments args;
    parse(argc, argv, args);
    timer.tick();
    timer.interval("Argument parsing");

    // Always create a context
    timer.label_next_interval(RED_COLOR "Create CUDA Contexts on all GPUs" RESET_COLOR);
    timer.tick();
    for (int i = 0; i < args.n_GPUS; i++) {
        cudaSetDevice(i);
        (void)cudaFree(0);
    }
    timer.tick_announce();
    // GPU initialization
    if (! args.validate()) {
        timer.label_next_interval("GPU context creation with dummy kernel");
        timer.tick();
        for (int i = 0; i < args.n_GPUS; i++) {
            cudaSetDevice(i);
            KERNEL_WARN(dummy_kernel<<<1 KERNEL_LAUNCH_SEPARATOR 1>>>());
            CUDA_ASSERT(cudaDeviceSynchronize());
        }
        timer.tick_announce();
        timer.label_next_interval("GPU trivial kernel launch");
        timer.tick();
        for (int i = 0; i < args.n_GPUS; i++) {
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
    // PROTOTYPE: Reloads a single VTK mesh ONCE PER GPU rather than partitioning a large mesh or loading precomputed partitions -- just demonstrate scaling on similar data volume!
    std::vector<std::unique_ptr<TV_Data>> TVs;
    for (int i = 0; i < args.n_GPUS; i++) {
        std::cout << INFO_EMOJI << "Load for GPU #" << i << std::endl;
        /*
        std::unique_ptr<TV_Data> TV = get_TV_from_VTK(args); // Uses: args.filename
        TVs.push_back(std::move(TV));
        */
        TVs.emplace_back(get_TV_from_VTK(args)); // Uses: args.filename
    }
    timer.tick_announce();


    Timer all_critical_points(false, "ALL Critical Points");
    // Have to make a max VV guess
    timer.label_next_interval("Approximate max VV");
    timer.tick();
    int max_VV_guess = args.max_VV;
    // TODO: Parallelize this across unique_ptrs and possibly make max_VV independent between GPUs
    if (max_VV_guess < 0) {
        for (int i = 0; i < args.n_GPUS; i++) {
            int new_VV_guess = get_approx_max_VV(*TVs[i], TVs[i]->nPoints);
            if (max_VV_guess < new_VV_guess) max_VV_guess = new_VV_guess;
        }
    }
    timer.tick_announce();

    timer.label_next_interval("Host-async size determination + allocations");
    timer.tick();
    // Size determinations
    // TODO: Need to be made per-device
    size_t tv_flat_size = sizeof(int) * TVs[0]->nCells * nbVertsInCell,
           vv_size = sizeof(int) * TVs[0]->nPoints * max_VV_guess,
           vv_index_size = sizeof(unsigned int) * TVs[0]->nPoints,
           // #Upper, #Lower, Classification
           partition_ids_size = sizeof(unsigned int) * TVs[0]->nPoints,
           classes_size = sizeof(unsigned int) * TVs[0]->nPoints * 3,
           // Upper/lower per adjacency
           valences_size = sizeof(int) * TVs[0]->nPoints * max_VV_guess,
           scalars_size = sizeof(double) * TVs[0]->nPoints;
    int n_to_compute = TVs[0]->nCells * nbVertsInCell;
    dim3 thread_block_size = 1024,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    const int shared_mem_size = sizeof(unsigned int) * (2+max_VV_guess);

    // Announce expected sizes before possibly running OOM
    std::cout << INFO_EMOJI << "Estimated memory footprint:   " << (tv_flat_size+
            partition_ids_size+classes_size+scalars_size+vv_size+vv_index_size+valences_size) / static_cast<float>(1024*1024*1024) <<
        " GiB" << std::endl;
    std::cout << INFO_EMOJI << "Estimated H->D memory demand: " << (tv_flat_size+
            scalars_size) / static_cast<float>(1024*1024*1024) <<
        " GiB" << std::endl;


    // Allocations
    // in cuda_extraction.cu: int * device_TV = nullptr;
    std::vector<int *> device_TVs(args.n_GPUS),
                       host_flat_tvs(args.n_GPUS),
                       vv_computeds(args.n_GPUS);
    std::vector<unsigned int *> vv_indices(args.n_GPUS),
                                partition_ids(args.n_GPUS),
                                host_CPCs(args.n_GPUS),
                                device_CPCs(args.n_GPUS);
    std::vector<int *> device_valences(args.n_GPUS);
    std::vector<double *> scalar_values(args.n_GPUS),
                          device_scalar_values(args.n_GPUS);
    for (int i = 0; i < args.n_GPUS; i++) {
        cudaSetDevice(i); // SHOULD BE 'i'
        CUDA_ASSERT(cudaMalloc((void**)&device_TVs[i], tv_flat_size));
        CUDA_ASSERT(cudaMallocHost((void**)&host_flat_tvs[i], tv_flat_size));
        CUDA_ASSERT(cudaMalloc((void**)&vv_computeds[i], vv_size));
        CUDA_ASSERT(cudaMalloc((void**)&vv_indices[i], vv_index_size));
        CUDA_ASSERT(cudaMalloc((void**)&partition_ids[i], partition_ids_size));
        CUDA_ASSERT(cudaMallocHost((void**)&host_CPCs[i], classes_size));
        CUDA_ASSERT(cudaMalloc((void**)&device_CPCs[i], classes_size));
        CUDA_ASSERT(cudaMalloc((void**)&device_valences[i], valences_size));
        CUDA_ASSERT(cudaMallocHost((void**)&scalar_values[i], scalars_size));
        CUDA_ASSERT(cudaMalloc((void**)&device_scalar_values[i], scalars_size));
    }
    // BUILDING: Partition IDs based on gpu ID for now (naive round robin)
    for (int i = 0; i < args.n_GPUS; i++) {
        unsigned int * partition_id_settings = new unsigned int[TVs[0]->nPoints];
        for (int j = 0; j < TVs[0]->nPoints; j++) {
            partition_id_settings[j] = (j % args.n_GPUS) == i;
        }
        CUDA_WARN(cudaMemcpy(partition_ids[i], partition_id_settings,
                             partition_ids_size, cudaMemcpyHostToDevice));
        delete partition_id_settings;
    }
    // END BUILDING -- this will be deprecated for real behavior in the future!
    timer.tick_announce();

    // Usually VE and VF are also mandatory, but CritPoints does not require
    // these relationships! Skip them!


    std::cout << WARN_EMOJI << "BEEG LOOP" << std::endl;
    pthread_t threads[args.n_GPUS];
    thread_arguments thread_args[args.n_GPUS];
    const int n_points = TVs[0]->nPoints;
    for (int i = 0; i < args.n_GPUS; i++) {
        std::cout << INFO_EMOJI << "Make thread GPU " << i << " ready" << std::endl;
        thread_args[i] = thread_arguments(i, std::move(TVs[i]), host_flat_tvs[i], device_TVs[i], tv_flat_size,
                      vv_computeds[i], vv_indices[i], vv_size, vv_index_size,
                      scalar_values[i], device_scalar_values[i], scalars_size,
                      device_valences[i], valences_size,
                      partition_ids[i], partition_ids_size,
                      device_CPCs[i], host_CPCs[i], classes_size, n_to_compute,
                      max_VV_guess, thread_block_size, grid_size, shared_mem_size,
                      args);
        pthread_create(&threads[i], NULL, parallel_work, (void*)&thread_args[i]);
    }
    // Merge-able structure
    unsigned int * returned_values = new unsigned int[(3*n_points)];
    bzero(returned_values, 3*n_points);
    for (int i = 0; i < args.n_GPUS; i++) {
        unsigned int *return_val;
        pthread_join(threads[i], (void**)&return_val);
        timer.label_next_interval("Export results from " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
        timer.tick();
        if (args.export_ == "/dev/null") {
            std::cerr << WARN_EMOJI << "Output to /dev/null omits function call entirely"
                      << std::endl;
        }
        else {
            export_classes(return_val,
                           #if CONSTRAIN_BLOCK
                           #else
                           n_points,
                           #endif
                           args);
        }
        timer.tick_announce();
        // Merge results
        for (int j = 0; j < (3*n_points); j++) {
            if (return_val[j] != 0) {
                unsigned int localized = j/3;
                if (localized == 5031) {
                    std::cout << "GPU " << i << " makes a write for point "
                              << localized << "(sub-index " << j % 3
                              << " with value=" << return_val[j] << ")"
                              << std::endl;
                }
                returned_values[j] = return_val[j];
            }
        }
    }
    timer.label_next_interval("Full results");
    timer.tick();
    if (args.export_ == "/dev/null") {
        std::cerr << WARN_EMOJI << "Output to /dev/null omits function call entirely"
                  << std::endl;
    }
    else {
        export_classes(returned_values,
                       #if CONSTRAIN_BLOCK
                       #else
                       n_points,
                       #endif
                       args);
    }
    delete[] returned_values;
    timer.tick_announce();
    all_critical_points.tick_announce();
    // TODO: Merge results between threads here, perhaps have parallel_work return mesh-able arrays to re-call a similar function to export_classes() upon
}

