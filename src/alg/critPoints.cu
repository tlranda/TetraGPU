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
#define FORCED_BLOCK_IDX 3183
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
                           const int * __restrict__ partition,
                           unsigned int * __restrict__ classes) {
    extern __shared__ unsigned int block_shared[];

    unsigned int *component_edits = &block_shared[0],
                 *upper_edits = &block_shared[0],
                 *lower_edits = &block_shared[1],
                 *neighborhood = &block_shared[2];
    const int tid = TID_SELECTION;
    /*
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
    */
    const int my_1d = tid / max_VV_guess,
              my_2d = VV[tid];
    // No work for this point's valence or out of partition
    if (VV_index[my_1d] <= 0 || my_2d < 0 || partition[my_1d] == 0) {
        #if PRINT_ON
        /*
        printf("Block %d Thread %02d exits due to partitioning or OOB data\n",
               blockIdx.x, threadIdx.x);
        */
        #endif
        return;
    }
    // Prefix scan as anti-duplication
    for (int i = my_1d * max_VV_guess; i < tid; i++) {
        if (VV[i] == my_2d) {
            #if PRINT_ON
            /*
            printf("Block %d Thread %02d exits from prefix scan\n",
                   blockIdx.x, threadIdx.x);
            */
            #endif
            return;
        }
    }

    // BEYOND THIS POINT, YOU ARE AN ACTUAL WORKER THREAD ON THE PROBLEM
    #if PRINT_ON
    // Not every thread needs to print this out, but every thread should have the same value
    printf("Block %d Thread %02d remains to participate in computation (VV_index=%d)\n",
           blockIdx.x, threadIdx.x, VV_index[my_1d]);
    #endif

    /*
        2) Read the scalar value used for point classification and classify
           yourself relative to your primary-dimension scalar value as upper or
           lower neighbor
    */
    // Classify yourself as an upper or lower valence neighbor to your 1d point
    // my_2d is {Upper = -1, Lower = 1} relative to my_1d
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
    printf("Block %d Thread %02d B init on my_1d %d and my_2d %d with valence %d based on %lf (block) vs %lf (thread)\n",
           blockIdx.x, threadIdx.x, my_1d, my_2d, my_class,
           scalar_values[my_1d], scalar_values[my_2d]);
    #endif
    /*
        3) For all other threads sharing your neighborhood classification, scan
           their connectivity in VV and update your "lowest classifying point"
           to be the lowest IDX (including yourself). Repeat until convergence
           where no edits are made, then if you have your own IDX at your
           location in memory, you log +1 component of your type.
    */
    bool upper_converge = false, lower_converge = false;
    /*
       Repeat Union-Find until both component directions converge

       Technically to_converge should NEVER be a termination condition, but
       it is probably not going to cost much in terms of overhead for the kernel
       and guarantees loop termination (eventually) on bad inputs / buggy versions
    */
    int to_converge = 0;
    while (!(upper_converge && lower_converge && to_converge < max_VV_guess)) {
        // Sanity: Guarantee zero-init
        if (threadIdx.x == 0) {
            #if PRINT_ON
            /*
            printf("Block %d Iteration %d: upper_edits %u lower_edits %u "
                   "Neighborhood %d %d %d %d %d %d %d %d "
                                "%d %d %d %d %d %d %d %d "
                                "%d %d %d %d %d %d %d %d "
                                "%d %d %d %d %d %d %d %d\n",
                    blockIdx.x, to_converge, *upper_edits, *lower_edits,
                    neighborhood[0],neighborhood[1],neighborhood[2],neighborhood[3],
                    neighborhood[4],neighborhood[5],neighborhood[6],neighborhood[7],
                    neighborhood[8],neighborhood[9],neighborhood[10],neighborhood[11],
                    neighborhood[12],neighborhood[13],neighborhood[14],neighborhood[15],
                    neighborhood[16],neighborhood[17],neighborhood[18],neighborhood[19],
                    neighborhood[20],neighborhood[21],neighborhood[22],neighborhood[23],
                    neighborhood[24],neighborhood[25],neighborhood[26],neighborhood[27],
                    neighborhood[28],neighborhood[29],neighborhood[30],neighborhood[31]);
            */
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
                // Find yourself in their adjacency to become a shared component and release shmem write burden
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
                        #ifdef PRINT_ON
                        /*
                        printf("Block %d Thread %02d Inspect %02d Shares component at VV[%d][%d] (%s)\n",
                                blockIdx.x, threadIdx.x, inspect_step, candidate_component_2d,
                                j-(candidate_component_2d*max_VV_guess),
                                my_class == -1 ? "Upper" : "Lower");
                        */
                        #endif
                    }
                }
            }
        }
        __syncthreads();
        upper_converge = *upper_edits == 0;
        lower_converge = *lower_edits == 0;
    }
    // End all Union-Finds

    if (neighborhood[threadIdx.x] == my_2d) { // You are the root of a component!
        const int memoffset = ((my_1d*3)+((my_class+1)/2));
        const unsigned int old = atomicAdd(classes+memoffset,1);
        #if PRINT_ON
        printf("Block %d Thread %02d Writes %s component @ mem offset %d (old: %u)\n",
               blockIdx.x, threadIdx.x,
               (my_class == -1 ? "Upper" : "Lower"), memoffset, old);
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
        if (to_converge > max_VV_guess / 2) {
            printf("Block %02d took %d loops to converge\n", blockIdx.x, to_converge);
        }
        printf("Block %d Thread %02d Verdict: my_1d (%d) has %u upper and %u lower\n",
               blockIdx.x, threadIdx.x, my_1d, upper, lower);
        #endif

        // Set classification
        if (upper >= 1 && lower == 0) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n", blockIdx.x,
                   threadIdx.x, "MINIMUM");
            #endif
            classes[my_classes+2] = MINIMUM_CLASS;
        }
        else if (upper == 0 && lower >= 1) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n", blockIdx.x,
                   threadIdx.x, "MAXIMUM");
            #endif
            classes[my_classes+2] = MAXIMUM_CLASS;
        }
        else if (upper == 1 && lower == 1) {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n", blockIdx.x,
                   threadIdx.x, "REGULAR");
            #endif
            classes[my_classes+2] = REGULAR_CLASS;
        }
        else {
            #if PRINT_ON
            printf("Block %d Thread %02d set classification as %s\n", blockIdx.x,
                   threadIdx.x, "SADDLE");
            #endif
            classes[my_classes+2] = SADDLE_CLASS;
        }
    }
    #if PRINT_ON
    printf("Block %d Thread %02d exits normally at end of computation\n",
           blockIdx.x, threadIdx.x);
    #endif
}

void export_classes(unsigned int * classes,
                    TV_Data & TV,
                    runtime_arguments & args) {
    vtkIdType n_classes = TV.nPoints;
    // VOIDS can be ignored during aggregation, but should not exist after all GPU kernels have returned!
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
        std::cout << INFO_EMOJI << "Outputting classes to " << args.export_
                                << std::endl;
    }
    // Used for actual file handling
    std::ostream out(output_buffer);
    std::string class_names[] = {"NULL", "min", "max", "regular", "saddle"};
    std::vector<vtkIdType> n_insane(TV.n_partitions,0),
                           n_min(TV.n_partitions,0),
                           n_max(TV.n_partitions,0),
                           n_saddle(TV.n_partitions,0),
                           n_regular(TV.n_partitions,0),
                           n_void(TV.n_partitions,0);
    for (vtkIdType i = 0; i < n_classes; i++)
    {
        const int partition_id = args.no_partitioning ? 0 : TV.partitionIDs[i];
        // The classification information is provided, then the class:
        // {# upper, # lower, class}
        // CLASSES = {'minimum': 1, 'maximum': 2, 'regular': 3, 'saddle': 4}
        const unsigned int n_upper  = classes[(i*3)],
                           n_lower  = classes[(i*3)+1],
                           my_class = classes[(i*3)+2];
        // Misclassification sanity checks
        if ((n_upper >= 1 && n_lower == 0 && my_class != MINIMUM_CLASS) ||
            (n_upper == 0 && n_lower >= 1 && my_class != MAXIMUM_CLASS) ||
            (n_upper == 1 && n_lower == 1 && my_class != REGULAR_CLASS) ||
            ((n_upper > 1 && n_lower > 1) && my_class != SADDLE_CLASS)) {
            /*
            out << "INSANITY DETECTED (" << n_upper << ", " << n_lower << ", "
                << my_class << ") FOR POINT " << i << " (Given class "
                << (my_class == 1 ? "Maximum" :
                    (my_class == 2 ? "Minimum" :
                     (my_class == 3 ? "Regular" :
                      (my_class == 4 ? "Saddle" :
                       (my_class == 0 ? "VOID" : "Non-mapped value")))))
                << ")" << std::endl;
            */
            n_insane[partition_id]++;
        }
        // Output formats
        if (args.debug > NO_DEBUG)
            out << "A Class " << i << " = " << my_class << std::endl;
        /*
        out << "A Class " << i << " = " << class_names[my_class] << "(Upper: "
            << n_upper << ", Lower: " << n_lower << ")" << std::endl;
        */
        if (my_class == MAXIMUM_CLASS) {
            n_max[partition_id]++;
            //out << "Point " << i << " in partition " << partition_id << " is a MAXIMUM" << std::endl;
        }
        else if (my_class == MINIMUM_CLASS) {
            n_min[partition_id]++;
            //out << "Point " << i << " in partition " << partition_id << " is a MINIMUM" << std::endl;
        }
        else if (my_class == SADDLE_CLASS) {
            n_saddle[partition_id]++;
            //out << "Point " << i << " in partition " << partition_id << " is a SADDLE" << std::endl;
        }
        else if (my_class == REGULAR_CLASS) {
            n_regular[partition_id]++;
        }
        else if (my_class == 0) {
            n_void[partition_id]++;
        }
    }
    const vtkIdType total_insane = std::accumulate(n_insane.begin(), n_insane.end(), 0);
    if (total_insane > 0) {
        std::cerr << WARN_EMOJI << RED_COLOR << "Insanity detected; "
                     "GPU did not agree on its own answers for " << total_insane
                  << " points." << RESET_COLOR << std::endl;
    }
    #ifdef VALIDATE_GPU
    else {
        std::cerr << OK_EMOJI << "No insanity detected in GPU self-agreement "
                     "when classifying points." << std::endl;
    }
    #endif
    out << "Total number of minima:  " << std::accumulate(n_min.begin(), n_min.end(), 0) << std::endl
        << "Total number of maxima:  " << std::accumulate(n_max.begin(), n_max.end(), 0) << std::endl
        << "Total number of saddles: " << std::accumulate(n_saddle.begin(), n_saddle.end(), 0) << std::endl
        << "Total number of regular: " << std::accumulate(n_regular.begin(), n_regular.end(), 0) << std::endl
        << "Total number of voids:   " << std::accumulate(n_void.begin(), n_void.end(), 0) << std::endl;
    if (args.debug == DEBUG_MAX) {
        for (int i = 0; i < TV.n_partitions; i++) {
            out << "Partition " << i << " number of minima:  " << n_min[i] << std::endl
                << "Partition " << i << " number of maxima:  " << n_max[i] << std::endl
                << "Partition " << i << " number of saddles: " << n_saddle[i] << std::endl
                << "Partition " << i << " number of regular: " << n_regular[i] << std::endl
                << "Partition " << i << " number of voids:   " << n_void[i] << std::endl;
        }
    }
}

void full_check_VV_Host(const size_t vv_size, const size_t vv_index_size, const size_t tv_size,
                        const int max_VV_local, const vtkIdType points, const vtkIdType cells,
                        const int * device_vv, const unsigned int * device_vv_index,
                        const int * device_localized_tv, cudaStream_t stream) {
    // Host copy of TV-flat
    int * host_flat_tv = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_flat_tv, tv_size));
    CUDA_WARN(cudaMemcpyAsync(host_flat_tv, device_localized_tv, tv_size, cudaMemcpyDeviceToHost, stream));
    // Host copy of VV data
    int * host_vv = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv, vv_size));
    CUDA_WARN(cudaMemcpyAsync(host_vv, device_vv, vv_size, cudaMemcpyDeviceToHost, stream));
    unsigned int * host_vv_index = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv_index, vv_index_size));
    CUDA_WARN(cudaMemcpyAsync(host_vv_index, device_vv_index, vv_index_size, cudaMemcpyDeviceToHost, stream));
    CUDA_WARN(cudaStreamSynchronize(stream));

    // Rigorously test to find all combinations using CPU
    for (vtkIdType i = 0; i < cells; i++) {
        int tv_cell[4] = { host_flat_tv[(i*4)+0],
                           host_flat_tv[(i*4)+1],
                           host_flat_tv[(i*4)+2],
                           host_flat_tv[(i*4)+3] };
        for (int idx = 0; idx < 4; idx++) {
            int look_match = tv_cell[idx];
            unsigned int max_index = host_vv_index[look_match],
                         base_index = look_match*max_VV_local;

            for (int idx2 = 0; idx2 < 4; idx2++) {
                if (idx2 == idx) continue;

                int look_for = tv_cell[idx2];
                bool found = false;
                for (unsigned int vv_index = 0; vv_index < max_index; vv_index++) {
                    if (host_vv[base_index+vv_index] == look_for) {
                        found = true;
                    }
                }
                if (!found) {
                    std::cerr << EXCLAIM_EMOJI <<  "Failed to locate edge for cell "
                              << i << ": (" << look_match << ", " << look_for
                              << ")" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    std::cerr << OK_EMOJI << "All VV entries found for TV" << std::endl;

    // Test outside of boundaries
    for (vtkIdType i = 0; i < points; i++) {
        unsigned int base_index = i * max_VV_local,
                     max_index = (i+1) * max_VV_local,
                     fill_depth = host_vv_index[i];
        for (unsigned int idx = base_index+fill_depth; idx < max_index; idx++) {
            if (host_vv[idx] != -1) {
                std::cerr << EXCLAIM_EMOJI << "VV has junk in point " << i
                          << " entries, starting as soon as index " << idx-base_index
                          << " / " << fill_depth << "(max-to-be-filled: "
                          << max_VV_local << ")" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    std::cerr << OK_EMOJI << "No junk detected OOB for VV" << std::endl;

    // No memory leaks! Deallocate!
    CUDA_WARN(cudaFreeHost(host_flat_tv));
    CUDA_WARN(cudaFreeHost(host_vv));
    CUDA_WARN(cudaFreeHost(host_vv_index));
}

void gale_check_VV_Host(size_t vv_size, size_t vv_index_size, int max_VV_local,
                        int * device_vv, unsigned int * device_vv_index, cudaStream_t stream) {
    int * host_vv = nullptr;
    unsigned int * host_vv_index = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv, vv_size));
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv_index, vv_index_size));
    CUDA_WARN(cudaMemcpyAsync(host_vv, device_vv, vv_size, cudaMemcpyDeviceToHost, stream));
    CUDA_WARN(cudaMemcpyAsync(host_vv_index, device_vv_index, vv_index_size, cudaMemcpyDeviceToHost, stream));
    CUDA_WARN(cudaStreamSynchronize(stream));
    int MAX_PRINT = 100;
    for (int i = 0; i < 4; i++) {
        std::cerr << "Sanity check VV[" << i << "] with size " << host_vv_index[i] << std::endl;
        int consecutive_minus_1 = 0;
        for (int j = 0; j < max_VV_local && MAX_PRINT > 0; j++) {
            if (host_vv[(i*max_VV_local)+j] == -1) {
                consecutive_minus_1++;
                continue;
            }
            else if (consecutive_minus_1 > 0) {
                std::cerr << "\t" << i << ": -1 (" << consecutive_minus_1 << " times)" << std::endl;
            }
            std::cerr << "\t" << i << ": " << host_vv[(i*max_VV_local)+j] << std::endl;
            MAX_PRINT--;
        }
        if (consecutive_minus_1 > 0) {
            std::cerr << "\t" << i << ": (possibly not complete: " << consecutive_minus_1 << " consecutive leftover -1's)" << std::endl;
        }
    }
    CUDA_ASSERT(cudaFreeHost(host_vv));
    CUDA_ASSERT(cudaFreeHost(host_vv_index));
}

void check_VV_errors(size_t vv_size, size_t vv_index_size,
                     int * vv_computed, unsigned int * vv_index,
                     TV_Data * TV, int max_VV_guess, cudaStream_t stream) {
    unsigned int * host_vv_index = nullptr;
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv_index, vv_index_size));
    CUDA_WARN(cudaMemcpyAsync(host_vv_index, vv_index, vv_index_size, cudaMemcpyDeviceToHost, stream));
    unsigned int duplicates = 0, minmax_size = 0, actual_size = 0;
    std::vector<unsigned int> overflow = std::vector<unsigned int>();
    int * host_vv = nullptr;
    // Determine the real size of de-duplicated VV
    CUDA_ASSERT(cudaMallocHost((void**)&host_vv, vv_size));
    CUDA_WARN(cudaMemcpyAsync(host_vv, vv_computed, vv_size, cudaMemcpyDeviceToHost, stream));
    CUDA_WARN(cudaStreamSynchronize(stream));
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
}

struct thread_arguments {
    int gpu_id, n_parallel;
    std::shared_ptr<TV_Data> TV;
    int max_VV;
    runtime_arguments args;

    thread_arguments(void) { }
    thread_arguments(
        int gpu_id1,
        int n_parallel1,
        std::shared_ptr<TV_Data> TV1,
        int max_VV1,
        runtime_arguments args1) {
            gpu_id = gpu_id1;
            n_parallel = n_parallel1;
            TV = TV1;
            max_VV = max_VV1;
            args = args1;
    }
};
void * parallel_work(void *parallel_arguments) {
    // Unpacking
    thread_arguments *thread_args = (thread_arguments *)parallel_arguments;
    int gpu_id = thread_args->gpu_id,
        max_VV_override = thread_args->max_VV,
        n_parallel = thread_args->n_parallel;
    const std::shared_ptr<TV_Data> TV = thread_args->TV;
    runtime_arguments _my_args = thread_args->args;

    std::streambuf * output_buffer;
    std::ofstream output_fstream;
    if (_my_args.threadNumber == 1) {
        output_buffer = std::cout.rdbuf();
    }
    else {
        char logname[32];
        sprintf(logname, "thread_%02d.log", gpu_id);
        output_fstream.open(logname);
        output_buffer = output_fstream.rdbuf();
    }
    std::ostream out(output_buffer);

    // Set up parallel timer
    char timername[32], intervalname[128];
    sprintf(timername, "Parallel worker %02d", gpu_id);
    Timer timer(true, timername, _my_args.debug == NO_DEBUG);

    // Establish GPU to utilize and allocate a stream for its operations
    int actual_gpus, vgpu_id, my_partition_id = gpu_id;
    CUDA_WARN(cudaGetDeviceCount(&actual_gpus));
    vgpu_id = (gpu_id % actual_gpus);
    cudaSetDevice(vgpu_id);
    /* MAYBE: Retain primary device context
    CUcontext cuContex;
    CUdevice cuDevice;
    // Port CUDA_SAFE_CALL() to include/cuda_safety.h if using this portion of API, as it is annoyingly different
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, vgpu_id);
    CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
    */
    // Set device stream
    cudaStream_t thread_stream;
    CUDA_WARN(cudaStreamCreate(&thread_stream));

    // Returnable memory has to be in dense format!
    unsigned int * dense_CPCs;
    size_t dense_cpc_size = (3*TV->nPoints)*sizeof(unsigned int);
    CUDA_ASSERT(cudaMallocHost((void**)&dense_CPCs, dense_cpc_size));
    bzero(dense_CPCs, 3*TV->nPoints);

    if (_my_args.no_partitioning) {
        my_partition_id = 0; // Guarantee single iteration
    }
    /*
    // DEBUG: Only loop certain partitions
    unsigned int approved_partition_ids[] = {9,};
    for (unsigned int my_partition_id : approved_partition_ids)
    */
    for (; my_partition_id < TV->n_partitions; my_partition_id += n_parallel)
    {
        // TV-localization
        sprintf(intervalname, "%s TV localization for partition %d", timername, my_partition_id);
        timer.label_next_interval(intervalname);
        timer.tick();
        TV_Data * TV_local;
        int max_VV_local;
        // Have to preserve relative vertex ID ordering for semantic consistency between --no_partition and utilizing partitions
        std::set<vtkIdType> included_points;
        std::vector<vtkIdType> included_cells, inverse_partition_mapping;
        // Shortcut available when no partitioning used
        if (_my_args.no_partitioning) {
            TV_local = &(*TV);
            max_VV_local = get_approx_max_VV(*TV_local, TV_local->nPoints, _my_args.debug);
            // Ensure partitioning data is set correctly
            TV_local->n_partitions = 1;
            std::vector<int> local_partitionIDs(TV_local->nPoints, 1);
            TV_local->partitionIDs = std::move(local_partitionIDs);
        }
        else {
            // Determine the number of points and cells
            vtkIdType nth_tetra = 0, n_points = 0, n_cells = 0;
            std::vector<std::array<vtkIdType, nbVertsInCell>> cells;
            std::vector<int> partition_IDs;
            std::vector<double> vertexAttributes;
            // First order inclusion: All points of any tetra with one or more vertices included in partition
            for (const auto & VertList : (*TV)) {
                for (const vtkIdType vertex : VertList) {
                    if (TV->partitionIDs[vertex] == my_partition_id) {
                        included_cells.emplace_back(nth_tetra);

                        // Include ALL vertices in this tetra in the partition
                        for (const vtkIdType incl_vertex : VertList) {
                            // Set object will not insert duplicates
                            included_points.insert(incl_vertex);
                        }
                        break; // No need to continue iterating this tetra
                    }
                }
                nth_tetra++;
            }
            // Now all included points and cells are known -- make stable remapping
            std::map<vtkIdType, vtkIdType> partition_remapping;
            std::vector<vtkIdType> stable_vertices(included_points.begin(), included_points.end());
            std::sort(stable_vertices.begin(), stable_vertices.end());
            vtkIdType nth_point = 0;
            for (const vtkIdType point : stable_vertices) {
                partition_remapping.insert({point, nth_point});
                nth_point++;
            }
            // This is correct, save an operation or two
            inverse_partition_mapping = std::move(stable_vertices);
            // Write the cells using localized indices
            for (vtkIdType tetra_id : included_cells) {
                cells.emplace_back(std::array<vtkIdType,nbVertsInCell>{
                        partition_remapping.at(TV->cells[tetra_id][0]),
                        partition_remapping.at(TV->cells[tetra_id][1]),
                        partition_remapping.at(TV->cells[tetra_id][2]),
                        partition_remapping.at(TV->cells[tetra_id][3]),
                        });
            }
            // Now fetch data for partition inclusion and scalars using inv map
            for (vtkIdType vertex : inverse_partition_mapping) {
                // Partition == 0 will skip classification in SFCP kernel
                partition_IDs.emplace_back(TV->partitionIDs[vertex] == my_partition_id);
                vertexAttributes.emplace_back(TV->vertexAttributes[vertex]);
            }
            n_points = included_points.size();
            n_cells = cells.size();

            // Set TV_local using localized data
            TV_local = new TV_Data(n_points, n_cells);
            TV_local->n_partitions = 2; // You're in-partition or out-of-partition
            TV_local->cells = std::move(cells);
            TV_local->partitionIDs = std::move(partition_IDs);
            TV_local->vertexAttributes = std::move(vertexAttributes);
            max_VV_local = get_approx_max_VV(*TV_local, n_points, _my_args.debug);
            vtkIdType n_in_partition = 0;
            for (vtkIdType i = 0; i < n_points; i++) {
                if (TV_local->partitionIDs[i] != 0) {
                    n_in_partition++;
                }
            }
            if (_my_args.debug > NO_DEBUG) {
                out << INFO_EMOJI << timername << " works on partition "
                    << my_partition_id << " with " << n_points << " points and "
                    << n_cells << " cells (" << n_in_partition
                    << " points are within partition)" << std::endl;
            }
        }
        if (max_VV_override != -1) {
            if (_my_args.debug > NO_DEBUG) {
                out << INFO_EMOJI << timername << " detected max_VV for partition "
                    << max_VV_local << ", but will be overridden by max_VV"
                    << max_VV_override << std::endl;
            }
            max_VV_local = max_VV_override;
        }
        timer.tick_announce();

        // Data Sizes & Allocations for VV
        sprintf(intervalname, "%s VV setup", timername);
        timer.label_next_interval(intervalname);
        timer.tick();
        int * host_flat_tv,
            * device_tv,
            * partition,
            * vv_computed;
        unsigned int * vv_index;
        size_t tv_flat_size = sizeof(int) * TV_local->nCells * nbVertsInCell,
               partition_size = sizeof(int) * TV_local->nPoints,
               vv_size = sizeof(int) * TV_local->nPoints * max_VV_local,
               vv_index_size = sizeof(unsigned int) * TV_local->nPoints;
        if (_my_args.debug > DEBUG_MIN) {
            out << PUSHPIN_EMOJI << timername << " will use GPU " << gpu_id
                << " (Actual GPU ID: " << vgpu_id << ") to compute "
                YELLOW_COLOR "VV" RESET_COLOR << std::endl
        // Announce expected sizes before possibly running OOM
                << INFO_EMOJI << timername << "'s estimated VV memory footprint: "
                << (tv_flat_size+partition_size+vv_size+vv_index_size) / static_cast<float>(1024*1024*1024)
                << " GiB" << std::endl;
        }
        // Allocated separately because we can release it earlier
        CUDA_ASSERT(cudaMalloc((void**)&device_tv, tv_flat_size));
        // TODO: Allocate together because we can release them simultaneously
        CUDA_ASSERT(cudaMalloc((void**)&partition, partition_size));
        CUDA_ASSERT(cudaMalloc((void**)&vv_computed, vv_size));
        CUDA_ASSERT(cudaMalloc((void**)&vv_index, vv_index_size));

        // VV Kernel Launch Specs
        int n_to_compute = TV_local->nCells * nbVertsInCell;
        dim3 thread_block_size = 1024,
             grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;

        // Set contiguous data in host memory -- separate scope to assist compiler
        {
            // Not host-pinning memory because this buffer won't be re-used
            host_flat_tv = new int[tv_flat_size / sizeof(int)];
            int index = 0;
            for (const auto & VertList : (*TV_local))
                for (const vtkIdType vertex : VertList)
                    host_flat_tv[index++] = vertex;
            // Device copy and host free
            // BLOCKING -- provide barrier if made asynchronous to avoid free of host
            // memory before copy completes
            CUDA_WARN(cudaMemcpyAsync(device_tv, host_flat_tv, tv_flat_size,
                                      cudaMemcpyHostToDevice, thread_stream));
            CUDA_WARN(cudaStreamSynchronize(thread_stream));
            // Free memory
            delete host_flat_tv;


            // Pre-populate vv!
            CUDA_WARN(cudaMemsetAsync(vv_computed, -1, vv_size, thread_stream));
            CUDA_WARN(cudaMemsetAsync(vv_index, 0, vv_index_size, thread_stream));


            int * partition_id_settings = new int[TV_local->nPoints]();
            for (int j = 0; j < TV_local->nPoints; j++) {
                partition_id_settings[j] = TV_local->partitionIDs[j];
            }
            CUDA_WARN(cudaMemcpyAsync(partition, partition_id_settings,
                                      partition_size, cudaMemcpyHostToDevice,
                                      thread_stream));
            CUDA_WARN(cudaStreamSynchronize(thread_stream));
            delete partition_id_settings;
        }

        // Compute the relationship
        if (_my_args.debug > DEBUG_MIN) {
            out << INFO_EMOJI << timername << " Kernel launch configuration is "
                << grid_size.x << " grid blocks with " << thread_block_size.x
                << " threads per block" << std::endl
                << INFO_EMOJI << timername << " The mesh has " << TV_local->nCells
                << " cells and " << TV_local->nPoints << " vertices" << std::endl
                << INFO_EMOJI << timername << " Tids >= " << TV_local->nCells * nbVertsInCell
                << " should auto-exit (" << (thread_block_size.x * grid_size.x) - n_to_compute
                << ")" << std::endl;
        }
        timer.tick_announce();
        char kerneltimername[32];
        sprintf(kerneltimername, "Parallel %s kernel %02d", "VV", gpu_id);
        Timer vvKernel(false, kerneltimername, _my_args.debug == NO_DEBUG);
        /*
        // DEBUG: Super serialized VV kernel
        for (int i = 0; i < TV_local->nCells; i++) {
            KERNEL_WARN(VV_kernel<<<1 KERNEL_LAUNCH_SEPARATOR
                                    4 KERNEL_LAUNCH_SEPARATOR
                                    0 KERNEL_LAUNCH SEPARATOR
                                    thread_stream>>>(
                        &device_tv[i*4], 1, 4, max_VV_local, vv_index, vv_computed));
            CUDA_WARN(cudaStreamSynchronize(thread_stream));
        }
        */
        KERNEL_WARN(VV_kernel<<<grid_size KERNEL_LAUNCH_SEPARATOR
                                thread_block_size KERNEL_LAUNCH_SEPARATOR
                                0 KERNEL_LAUNCH_SEPARATOR
                                thread_stream>>>(device_tv,
                                    TV_local->nCells,
                                    TV_local->nPoints,
                                    max_VV_local,
                                    vv_index,
                                    vv_computed));
        // Only synchronized to ensure VV kernel duration is appropriately tracked
        CUDA_WARN(cudaStreamSynchronize(thread_stream));
        vvKernel.tick();
        sprintf(intervalname, "%s VV kernel duration", timername);
        vvKernel.label_prev_interval(intervalname);
        // DEBUG: Check that every point in VV is properly found
        /*
        full_check_VV_Host(vv_size, vv_index_size, tv_flat_size, max_VV_local,
                           TV_local->nPoints, TV_local->nCells, vv_computed,
                           vv_index, device_tv, thread_stream);
        */
        // No longer need TV allocation, free it
        CUDA_WARN(cudaFree(device_tv));
        // NOTE: Asynchronous WRT CPU, we can continue to setup SFCP kernel while VV runs

        // Additional allocations and settings for SFCP kernel
        sprintf(intervalname, "%s Setup for SFCP kernel", timername);
        timer.label_next_interval(intervalname);
        timer.tick();
        unsigned int * host_CPCs, // This one is returnable!!
                     * device_CPCs;
        int * device_valences;
        double * device_scalar_values;
        const int shared_mem_size = sizeof(unsigned int) * (2+max_VV_local);
        size_t cpc_size = sizeof(unsigned int) * TV_local->nPoints * 3,
               valence_size = sizeof(int) * TV_local->nPoints * max_VV_local,
               scalar_values_size = sizeof(double) * TV_local->nPoints;

        {
            CUDA_ASSERT(cudaMalloc((void**)&device_CPCs, cpc_size));
            CUDA_ASSERT(cudaMalloc((void**)&device_valences, valence_size));
            CUDA_ASSERT(cudaMalloc((void**)&device_scalar_values, scalar_values_size));

            double * scalar_values = new double[TV_local->nPoints];
            // Scalar values from VTK
            for(int j = 0; j < TV_local->nPoints; j++) {
                scalar_values[j] = TV_local->vertexAttributes[j];
            }
            // MADE BLOCKING to prevent early free
            CUDA_WARN(cudaMemcpyAsync(device_scalar_values, scalar_values,
                                      scalar_values_size, cudaMemcpyHostToDevice,
                                      thread_stream));
            // We need CPCs to be zero'd out
            CUDA_WARN(cudaMemsetAsync(device_CPCs, 0, cpc_size, thread_stream));
            CUDA_WARN(cudaStreamSynchronize(thread_stream));
            delete scalar_values;
        }
        timer.tick_announce();

        /*
        // DEBUG: Check VV for threats to correctness / efficiency
        sprintf(intervalname, "%s Check for VV duplicates / error", timername);
        timer.label_next_interval(intervalname);
        timer.tick();
        // More intended for GALE integration: checks more specifically
        //gale_check_VV_Host(vv_size, vv_index_size, max_VV_local, vv_computed, vv_index);
        // Detects errors but not very specifically
        check_VV_errors(vv_size,vv_index_size, vv_computed, vv_index, TV_local,
                        max_VV_local, thread_stream);
        timer.tick_announce();
        */


        // Critical Points
        sprintf(intervalname, "%s Run " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm", timername);
        timer.label_next_interval(intervalname);
        // Set kernel launch parameters here
        /*
            1) Parallelize VV on second-dimension (can early-exit block if no data
               available or if a prefix-scan of your primary-dimension list shows
               that you are a duplicate)
        */
        timer.tick();
        // RESET KERNEL PARAMETERS
        n_to_compute = TV_local->nPoints * max_VV_local;
        thread_block_size.x = max_VV_local;
        grid_size.x = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;

        #if CONSTRAIN_BLOCK
        grid_size.x = 1;
        std::cerr << EXCLAIM_EMOJI << "DEBUG! Setting kernel size to 1 block (as block "
                  << FORCED_BLOCK_IDX << ")" << std::endl;
        #endif
        if (_my_args.debug > DEBUG_MIN) {
            out << timername << " Launch critPoints kernel with " << n_to_compute
                << " units of work (" << grid_size.x << " blocks, "
                << thread_block_size.x << " threads per block)" << std::endl;
        }
        sprintf(kerneltimername, "Parallel %s kernel %02d", "SFCP", gpu_id);
        Timer sfcpKernel(false, kerneltimername, _my_args.debug == NO_DEBUG);
        KERNEL_WARN(critPoints<<<grid_size KERNEL_LAUNCH_SEPARATOR
                                 thread_block_size KERNEL_LAUNCH_SEPARATOR
                                 shared_mem_size KERNEL_LAUNCH_SEPARATOR
                                 thread_stream>>>(vv_computed,
                                                    vv_index,
                                                    device_valences,
                                                    TV_local->nPoints,
                                                    max_VV_local,
                                                    device_scalar_values,
                                                    partition,
                                                    device_CPCs));
        CUDA_WARN(cudaStreamSynchronize(thread_stream)); // Make algorithm timing accurate
        sfcpKernel.tick();
        sprintf(intervalname, "%s SFCP kernel", timername);
        sfcpKernel.label_prev_interval(intervalname);
        timer.tick_announce();
        sprintf(intervalname, "%s Retrieve results from GPU", timername);
        timer.label_next_interval(intervalname);
        timer.tick();
        // Reminder! This is returnable memory by this function!
        CUDA_ASSERT(cudaMallocHost((void**)&host_CPCs, cpc_size));
        // In the event 0-values are intended, zero out the buffer PRIOR to copying?
        memset(host_CPCs, 0, cpc_size);
        // Synchronous wrt this thread, not others
        CUDA_WARN(cudaMemcpyAsync(host_CPCs, device_CPCs, cpc_size, cudaMemcpyDeviceToHost, thread_stream));
        CUDA_WARN(cudaStreamSynchronize(thread_stream));
        if (_my_args.no_partitioning) {
            // Directly inject to return value via free and swap
            CUDA_WARN(cudaFreeHost(dense_CPCs));
            dense_CPCs = host_CPCs;
        }
        else {
            // Restore original dense mapping (accumulates between loop iterations)
            for (vtkIdType i = 0; i < TV_local->nPoints; i++) {
                if (TV_local->partitionIDs[i] == 0) {
                    // Point isn't classified by THIS partition, but do NOT overwrite it!
                    // It may have been written to by a prior loop iteration!
                    continue;
                }
                vtkIdType dense_basis = inverse_partition_mapping[i] * 3,
                          host_basis  = i * 3;
                dense_CPCs[dense_basis  ] = host_CPCs[host_basis  ];
                dense_CPCs[dense_basis+1] = host_CPCs[host_basis+1];
                dense_CPCs[dense_basis+2] = host_CPCs[host_basis+2];
                // DEBUG: Announce values
                /*
                out << "Set dense CPCs values for point " << i << " between indices "
                    << dense_basis << "-" << dense_basis+2 << " (values: "
                    << dense_CPCs[dense_basis  ] << ", "
                    << dense_CPCs[dense_basis+1] << ", "
                    << dense_CPCs[dense_basis+2] << ")" << std::endl;
                */
            }
        }
        timer.tick_announce();

        sprintf(intervalname, "%s Free memory", timername);
        timer.label_next_interval(intervalname);
        timer.tick();
        if (vv_computed != nullptr) CUDA_WARN(cudaFree(vv_computed));
        if (vv_index != nullptr) CUDA_WARN(cudaFree(vv_index));
        if (partition != nullptr) CUDA_WARN(cudaFree(partition));
        if (device_CPCs != nullptr) CUDA_WARN(cudaFree(device_CPCs));
        if (device_valences != nullptr) CUDA_WARN(cudaFree(device_valences));
        if (device_scalar_values != nullptr) CUDA_WARN(cudaFree(device_scalar_values));
        if (host_CPCs != nullptr && !_my_args.no_partitioning) CUDA_WARN(cudaFreeHost(host_CPCs));
        timer.tick_announce();
    }
    return (void*)dense_CPCs;
}

int main(int argc, char *argv[]) {
    // Program initialization / argument parsing
    Timer timer(false, "Main");
    runtime_arguments args;
    parse(argc, argv, args);
    timer.tick();
    timer.interval("Argument parsing");
    std::cout << std::endl << std::endl;


    // GPU intializations
    timer.label_next_interval(RED_COLOR "Create CUDA Contexts on all GPUs" RESET_COLOR);
    timer.tick();
    int actual_gpus;
    CUDA_WARN(cudaGetDeviceCount(&actual_gpus));
    for (int i = 0; i < actual_gpus; i++) {
        cudaSetDevice(i);
        (void)cudaFree(0);
        if (! args.validate()) {
            KERNEL_WARN(dummy_kernel<<<1 KERNEL_LAUNCH_SEPARATOR 1>>>());
        }
        CUDA_ASSERT(cudaDeviceSynchronize());
    }
    timer.tick_announce();
    if (args.debug > NO_DEBUG) std::cout << std::endl << std::endl;


    /* Utilize VTK API to load the entire mesh once (read-accessible to all threads),
       should de-allocate VTK's heap as much as possible as function closes

       Vertex Attributes and Partitioning information are also collected in the
       returned datastructure

       Actions rely upon args attributes: filename, arrayname, partitioningname
     */
    std::cout << PUSHPIN_EMOJI << "Parsing vtu file: " << args.fileName
              << std::endl;
    timer.label_next_interval(GREEN_COLOR "TV" RESET_COLOR " from VTK");
    timer.tick();
    std::shared_ptr<TV_Data> TV = get_TV_from_VTK(args);
    timer.tick_announce();
    std::cout << std::endl << std::endl;


    Timer all_critical_points(false, "ALL Critical Points", args.debug == NO_DEBUG);
    // Parallelization
    int n_parallel = args.n_GPUS * args.threadNumber;
    if (args.no_partitioning) {
        n_parallel = 1;
    }
    std::cout << PUSHPIN_EMOJI << "Parallelizing across " << n_parallel
              << " threads" << std::endl;
    pthread_t threads[n_parallel];
    thread_arguments thread_args[n_parallel];
    unsigned int * return_vals[n_parallel];
    for (int i = 0; i < n_parallel; i++) {
        if (args.debug > DEBUG_MIN) {
            std::cout << INFO_EMOJI << "Make thread GPU " << i << " ready" << std::endl;
        }
        thread_args[i] = thread_arguments(i, n_parallel, TV, args.max_VV, args);
        pthread_create(&threads[i], NULL, parallel_work, (void*)&thread_args[i]);
        //return_vals[i] = (unsigned int*)parallel_work((void*)&thread_args[i]);
    }
    // Merge-able structure for final results
    // Zeroes required for correctness
    unsigned int * returned_values = new unsigned int[(3*TV->nPoints)]();
    for (int i = 0; i < n_parallel; i++) {
        unsigned int *return_val;// = return_vals[i];
        pthread_join(threads[i], (void**)&return_val);
        return_vals[i] = return_val;
        /*
        // DEBUG: Show classes PER cluster returned
        timer.label_next_interval("Export results from " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
        timer.tick();
        if (args.export_ == "/dev/null") {
            std::cerr << WARN_EMOJI << "Output to /dev/null omits function call entirely"
                      << std::endl;
        }
        else {
            export_classes(return_val,
                           (*TV),
                           args);
        }
        timer.tick_announce();
        */
        // Merge results
        for (int j = 0; j < (3*TV->nPoints); j++) {
            if (return_val[j] != 0) {
                if (returned_values[j] != 0) {
                    /*
                    std::cerr << "DOUBLE WRITE TO POINT " << j / 3
                              << " (return_val index " << j << ") -- should be owned by "
                              << i << " (" << i << " % " << n_parallel << " == "
                              << (i % n_parallel) << ")" << std::endl;
                    */
                    int k;
                    for (k = 0; k < i; k++) {
                        if (return_vals[k][j] != 0) {
                            /*
                            std::cerr << "Also found a write to " << j
                                      << " made by " << k << " (OLD value: "
                                      << return_vals[k][j] << " clashes with NEW value: "
                                      << returned_values[j] << ")" << std::endl;
                            */
                            break;
                        }
                    }
                    if (return_vals[k][j] != returned_values[j])
                        exit(EXIT_FAILURE);
                }
                returned_values[j] = return_val[j];
            }
        }
        // NOTE: Responsible to free this memory from the thread!
        CUDA_ASSERT(cudaFreeHost(return_val));
        return_vals[i] = nullptr;
    }
    std::cout << std::endl << std::endl;

    timer.label_next_interval("Full results");
    timer.tick();
    if (args.export_ == "/dev/null") {
        std::cerr << WARN_EMOJI << "Output to /dev/null omits function call entirely"
                  << std::endl;
    }
    else {
        export_classes(returned_values,
                       (*TV),
                       args);
    }
    delete[] returned_values;
    // Close timers
    timer.tick_announce();
    all_critical_points.tick_announce();
}

