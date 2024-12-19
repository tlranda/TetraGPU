// Compile: nvcc -ccbin=/home/tlranda/tools/gcc7/bin shuffle_6_to_4.cu
#include<iostream>
#include<cstdlib>
#include<cuda.h>

// Type of integer used
typedef unsigned long int bounded_int;

__device__ __inline__ void bitwiseAND_64(bounded_int val0, bounded_int val1,
                                         bounded_int val2, bounded_int val3,
                                         bounded_int laneID,
                                         bounded_int threadID,
                                         bounded_int global_read,
                                         bounded_int * __restrict__ out) {
    /*
       Pattern to create:
       0: v0 & v1
       1: v1 & v2
       2: v2 & v3
       3: v0 & v2
       4: v1 & v3
       5: v0 & v3
    */
    bounded_int left_operand = val0,
                right_operand = val3;
    //printf("bitwiseAND thread %03lu INIT CONDITION: %03lu & %03lu\n",
    //       threadID, left_operand, right_operand);
    if (laneID == 1 || laneID == 2 || laneID == 4) {
        if (laneID == 2) left_operand = val2; // laneID == 2
        else left_operand = val1; // laneID == 1 || laneID == 4
    }
    //printf("bitwiseAND thread %03lu Left Fixed: %03lu & %03lu\n",
    //       threadID, left_operand, right_operand);
    // else laneID == 0 || laneID == 3 || laneID == 5 --> left_operand = val0
    if (laneID == 0 || laneID == 1 || laneID == 3) {
        if (laneID == 0) right_operand = val1; // laneID == 0
        else right_operand = val2; // laneID == 1 || laneID == 3
    }
    //printf("bitwiseAND thread %03lu Right Fixed: %03lu & %03lu\n",
    //       threadID, left_operand, right_operand);
    // else laneID == 2 || laneID == 4 || laneID == 5 --> right_operand = val3
    out[laneID] = left_operand & right_operand;
}

__global__ void kernel_64(bounded_int * __restrict__ in,
                          bounded_int max_cells,
                          bounded_int * __restrict__ out) {
    extern __shared__ bounded_int scratchpad[];
    // LAUNCH WITH 6 THREADS PER CELL IN WHATEVER FASHION YOU LIKE, BUT YOU
    // LOSE 2 THREADS PER WARP SO YOU NEED TO OVER-ALLOCATE TO GET THEM BACK IN

    // To prove correctness, our algorithm is as follows:
    // Inputs will be groups of FOUR (with common high-bits) that have a single
    // unique lower-4 bit set (ie a group could be: 0xff001, 0xff002, 0xff004, 0xff008)
    // The operation will BITWISE AND all pairs of the group of 4 to make SIX
    // outputs in the order of 0-1, 1-2, 2-3, 0-2, 1-3, 0-3 (for the above, this
    // makes the outputs: 0xff003, 0xff006, 0xff00C, 0xff005, 0xff00A, 0xff009)

    // GLOBAL thread ID
    // Warp-local ID
    // Shuffle-6 lane ID
    // GLOBAL lane depth
    bounded_int threadID = (blockIdx.x * blockDim.x) + threadIdx.x,
                warpID = (threadIdx.x % 32),
                laneID = warpID % 6,
                laneDepth = 3*(((threadID / 32)*5) + (warpID / 6)),
                shLaneDepth = laneDepth % 480;
    // Adjust output pointers per-thread to ensure they don't overwite one
    // another's data. Each thread outputs 6 values per iteration, with at most
    // 3 iterations, so the skip value is 18
    out += (laneDepth * 6);
    //out[laneID] = 1010;

    // Early-exit:
    //      Threads that would read beyond max_cells at their base value
    //      The two straggler threads of each warp

    if (laneDepth >= max_cells || warpID > 29) return;

    out[laneID] = 1234;

    // Read your FIRST value from global memory
    // laneDepth *= 4 to use vector-addressing, not set permanently due to
    // subsequent references that need the quadruplet ID later
    bounded_int read_indicator = max_cells-laneDepth-1,
                global_read;

    // Mask for first read
    if (read_indicator >= 1 || (read_indicator == 0 && laneID < 4)) {
        global_read = in[(laneDepth*4) + laneID];
        //printf("ThreadID %03lu (laneID %lu) reads index %03lu with value %03lu\n",
        //       threadID, laneID, (laneDepth*4)+laneID, global_read);
        scratchpad[(shLaneDepth*6)+laneID] = global_read;
    }
    __syncthreads();

    // ITERATION 1: First quadruplet -- guaranteed to be useful by early-exit above
    bounded_int quad0 = scratchpad[(shLaneDepth*6)  ], //__shfl_sync(0xfffffffc, global_read, 0, 6),
                quad1 = scratchpad[(shLaneDepth*6)+1], //__shfl_sync(0xfffffffc, global_read, 1, 6),
                quad2 = scratchpad[(shLaneDepth*6)+2], //__shfl_sync(0xfffffffc, global_read, 2, 6),
                quad3 = scratchpad[(shLaneDepth*6)+3]; //__shfl_sync(0xfffffffc, global_read, 3, 6);
    //__syncthreads();
    //printf("ThreadID %03lu (laneID %lu) shuffled for values %03lu %03lu %03lu %03lu\n",
    //        threadID, laneID, quad0, quad1, quad2, quad3);
    // ALGORITHM FOR ITERATION 1
    bitwiseAND_64(quad0, quad1, quad2, quad3, laneID, threadID, global_read, out);

    // ITERATION 2: Second quadruplet -- early-exit if NOT useful
    // Adjust pointers to not overwrite previous iteration's data
    out += 6;
    out[laneID] = 1235;
    if (read_indicator == 0) { out[laneID] = 2024; return; }
    quad0 = scratchpad[(shLaneDepth*6)+4]; //__shfl_sync(0xfffffffc, global_read, 4, 6);
    quad1 = scratchpad[(shLaneDepth*6)+4+1]; //__shfl_sync(0xfffffffc, global_read, 5, 6);
    __syncthreads();
    //printf("ThreadID %03lu (laneID %lu) shuffled for values %03lu %03lu (1 of 2)\n",
    //        threadID, laneID, quad0, quad1);
    // Mask for second read
    if (read_indicator > 1 || (read_indicator == 1 & laneID < 2)) {
        // First read is exhausted, make the second global read now with
        // incremented offset
        global_read = in[(laneDepth*4) + laneID + 6];
        //printf("ThreadID %03lu (laneID %lu) reads index %03lu with value %03lu\n",
        //        threadID, laneID, (laneDepth*4)+laneID+6, global_read);
        scratchpad[(shLaneDepth*6)+laneID] = global_read;
    }
    __syncthreads();
    quad2 = scratchpad[(shLaneDepth*6)  ]; //__shfl_sync(0xfffffffc, global_read, 0, 6);
    quad3 = scratchpad[(shLaneDepth*6)+1]; //__shfl_sync(0xfffffffc, global_read, 1, 6);
    //__syncthreads();
    //printf("ThreadID %03lu (laneID %lu) shuffled for values %03lu %03lu (2 of 2)\n",
    //        threadID, laneID, quad2, quad3);
    // ALGORITHM FOR ITERATION 2
    bitwiseAND_64(quad0, quad1, quad2, quad3, laneID, threadID, global_read, out);

    // ITERATION 3: Third quadruplet -- early-exit if NOT useful
    // Adjust pointers to not overwrite previous iteration's data
    out += 6;
    out[laneID] = 1236;
    if (read_indicator == 1) { out[laneID] = 2025; return; }
    quad0 = scratchpad[(shLaneDepth*6)+2]; //__shfl_sync(0xfffffffc, global_read, 2, 6);
    quad1 = scratchpad[(shLaneDepth*6)+3]; //__shfl_sync(0xfffffffc, global_read, 3, 6);
    quad2 = scratchpad[(shLaneDepth*6)+4]; //__shfl_sync(0xfffffffc, global_read, 4, 6);
    quad3 = scratchpad[(shLaneDepth*6)+5]; //__shfl_sync(0xfffffffc, global_read, 5, 6);
    //__syncthreads();
    //printf("ThreadID %03lu (laneID %lu) shuffled for values %03lu %03lu %03lu %03lu\n",
    //        threadID, laneID, quad0, quad1, quad2, quad3);
    // ALGORITHM FOR ITERATION 3
    bitwiseAND_64(quad0, quad1, quad2, quad3, laneID, threadID, global_read, out);
    //__syncthreads();
}

const bounded_int INFO_LINENO = __LINE__+1;
// Test program's bitwise AND requires separation between low/high bits,
// naive starter differentiation for high bits limits you to a number of
// cells. To exceed this limit, use a larger datatype or adjust the
// high-bit masking algorithm. We start with UNSIGNED LONG INT (see type
// definitinition for "bounded_int" at the top of this program) as this is
// usually guaranteed to be 32-bit in C++ (ie: it isn't a short integer,
// which is often permitted for integers without a length specification)
#define MAX_SUPPORTED 26843456 // 2**(32-4)
int main(int argc, char **argv) {
    // Command line interaction
    bounded_int N_CELLS = 30; // default
    if (argc < 2) {
        std::cerr << "USAGE: " << argv[0] << " <INTEGER # CELLS>" << std::endl
                  << "(Using default of " << N_CELLS << " due to lack of "
                  << "specified input)" << std::endl;
    }
    else N_CELLS = std::atol(argv[1]);
    size_t SIZE = N_CELLS * sizeof(bounded_int);

    std::cout << "Prepare " << N_CELLS << " cells of input" << std::endl;
    if (N_CELLS > MAX_SUPPORTED) {
        std::cerr << "TOO MANY CELLS PROVISIONED, CORRECTNESS OF ALGORITHM "
                  << "AND VALIDATION CHECKS NOT GUARANTEED! REQUIRED SOURCE "
                  << "CODE MODIFICATIONS (ie: increase datatype size or adjust "
                  << "algorithm) ARE DOCUMENTED IN " << __FILE__ << ":"
                  << INFO_LINENO << std::endl;
    }


    // Memory management and algorithm setup
    bounded_int *host_input, *host_expect, *host_receive,
                *device_input, *device_output;
    // Allocations
    try {
        host_input = new bounded_int[N_CELLS*4]; // Trivial input with attributable blame
        host_expect = new bounded_int[N_CELLS*6]; // Expected outputs
        host_receive = new bounded_int[N_CELLS*6]; // Retreive answer from GPU
    } catch(std::bad_alloc&) {
        std::cerr << "One or more host memory allocations failed. "
                  << "You may need to reduce the cell count." << std::endl;
        exit(EXIT_FAILURE);
    }
    if ((cudaMalloc((void**)&device_input, SIZE*4) != cudaSuccess)
            || (cudaMalloc((void**)&device_output, SIZE*6) != cudaSuccess)) {
        std::cerr << "One or more device memory allocations failed. "
                  << "You may need to reduce the cell count." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set up host memory
    for (bounded_int i = 0; i < N_CELLS * 6; i++) host_receive[i] = -1;
    for (bounded_int i = 0; i < N_CELLS; i++) {
        // HIGH BIT MASK is cell ID shifted over the low bits
        bounded_int hi_mask = i << 4;
        //std::cout << "MASK: " << hi_mask << std::endl;
        // Total answer is HIGH BIT MASK bitwise or'd with one-hot low bit
        host_input[(i*4)  ] = hi_mask | 0x7; // 0 1 1 1
        host_input[(i*4)+1] = hi_mask | 0xB; // 1 0 1 1
        host_input[(i*4)+2] = hi_mask | 0xD; // 1 1 0 1
        host_input[(i*4)+3] = hi_mask | 0xE; // 1 1 1 0
        /*
        std::cout << "\tINPUT: "
                  <<  "{1: " << host_input[(i*4)  ] << ", 2: " << host_input[(i*4)+1]
                  << ", 3: " << host_input[(i*4)+2] << ", 4: " << host_input[(i*4)+3]
                  << "}" << std::endl;
        */
        // Expected answer is canonically sorted
        host_expect[(i*6)  ] = host_input[(i*4)  ] & host_input[(i*4)+1]; // 0x3 = 0 0 1 1
        host_expect[(i*6)+1] = host_input[(i*4)+1] & host_input[(i*4)+2]; // 0x9 = 1 0 0 1
        host_expect[(i*6)+2] = host_input[(i*4)+2] & host_input[(i*4)+3]; // 0xC = 1 1 0 0
        host_expect[(i*6)+3] = host_input[(i*4)  ] & host_input[(i*4)+2]; // 0x5 = 0 1 0 1
        host_expect[(i*6)+4] = host_input[(i*4)+1] & host_input[(i*4)+3]; // 0xA = 1 0 1 0
        host_expect[(i*6)+5] = host_input[(i*4)  ] & host_input[(i*4)+3]; // 0x6 = 0 1 1 0
        /*
        std::cout << "\tBIT_AND WITH MASK: "
                  <<  "{1&2: " << host_expect[(i*6)  ] << ", 2&3: " << host_expect[(i*6)+1]
                  << ", 3&4: " << host_expect[(i*6)+2] << ", 1&3: " << host_expect[(i*6)+3]
                  << ", 2&4: " << host_expect[(i*6)+4] << ", 1&4: " << host_expect[(i*6)+5]
                  << "}" << std::endl;
        std::cout << "\tBIT_AND without MASK: "
                  <<  "{1&2: " << host_expect[(i*6)  ]-hi_mask << ", 2&3: " << host_expect[(i*6)+1]-hi_mask
                  << ", 3&4: " << host_expect[(i*6)+2]-hi_mask << ", 1&3: " << host_expect[(i*6)+3]-hi_mask
                  << ", 2&4: " << host_expect[(i*6)+4]-hi_mask << ", 1&4: " << host_expect[(i*6)+5]-hi_mask
                  << "}" << std::endl;
        */
    }
    // Copy input and blanks over to device
    if ((cudaMemcpy(device_input, host_input, SIZE*4, cudaMemcpyHostToDevice) != cudaSuccess)
            || (cudaMemcpy(device_output, host_receive, SIZE*6, cudaMemcpyHostToDevice) != cudaSuccess)) {
        std::cerr << "One or more host->device copies failed." << std::endl;
        exit(EXIT_FAILURE);
    }


    // Set up launch configuration and execute kernel
    const bounded_int N_THREADS = 1024,
                      cells_per_block = 480,
                      SHARED_PER_BLOCK = cells_per_block * 6 * sizeof(bounded_int);
    // 6 edges required per cell
    // Up to three cells unrolled per edge group
    // -2 threads per warp of 32 threads for warp alignment
    // 6*((480+2)//3) == 6 * 160 == 960
    // ((960+29)//30)*32 == 32*32 = 1024
    // Max 1024 threads on hardware, increasing to 481 cells requires a new block to hold the new warp
    bounded_int N_BLOCKS = (N_CELLS+cells_per_block-1)/cells_per_block;
    std::cout << "Data ready, launch kernel with " << N_BLOCKS << " blocks of "
              << N_THREADS << " threads (requires: " << SHARED_PER_BLOCK
              << " bytes shmem)" << std::endl;
    if (cudaFuncSetAttribute(kernel_64, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152)
            != cudaSuccess) {
        std::cerr << "Could not set shared memory size to 49152 bytes" << std::endl;
    }
    kernel_64<<<N_BLOCKS, N_THREADS, SHARED_PER_BLOCK>>>(device_input, N_CELLS,
                device_output);
    if (cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "Kernel launch error" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "Kernel error" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Kernel complete" << std::endl;
    if (cudaMemcpy(host_receive, device_output, SIZE*6, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Device->host copy failed." << std::endl;
        exit(EXIT_FAILURE);
    }


    // Check values vs expectation
    bounded_int error = 0;
    for (bounded_int i = 0; i < N_CELLS; i++) {
        for (int j = 0; j < 6; j++) {
            bounded_int addr = (i*6)+j,
                        expect = host_expect[addr],
                        receive = host_receive[addr];
            if (expect != receive) {
                error++;
                std::cout << "Host cell " << i << " output #" << j
                          << " does not meet expectations. EXPECT: "
                          << expect << " | RECEIVE: " << receive << std::endl;
            }
        }
    }
    if (error > 0) std::cout << "Total errors: " << error
                             << " (of max possible " << N_CELLS*6 << ")"
                             << std::endl;
    else std::cout << "Validated" << std::endl;


    // Memory frees
    delete host_input;
    delete host_expect;
    delete host_receive;
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}

