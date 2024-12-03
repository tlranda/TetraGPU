// Compile: nvcc -ccbin=/home/tlranda/tools/gcc7/bin shuffle_6_to_4.cu
#include<iostream>
#include<cstdlib>
#include<cuda.h>

// Type of integer used
typedef unsigned long int bounded_int;

__device__ __inline__ void bitwiseAND_64(bounded_int val0, bounded_int val1,
                                         bounded_int val2, bounded_int val3,
                                         bounded_int laneID,
                                         bounded_int * __restrict__ out) {
    bounded_int left_operand = val0,
                right_operand = val3;
    if (laneID == 1 || laneID == 2 || laneID == 4) {
        if (laneID == 2) left_operand = val2;
        else left_operand = val1;
    }
    if (laneID == 0 || laneID == 1 || laneID == 3) {
        if (laneID == 0) right_operand = val1;
        else right_operand = val2;
    }
    out[laneID] = left_operand & right_operand;
}

__global__ void kernel_64(bounded_int * __restrict__ in,
                          bounded_int max_cells,
                          bounded_int * __restrict__ out) {
    // LAUNCH WITH 6 THREADS PER CELL IN WHATEVER FASHION YOU LIKE, BUT YOU
    // LOSE 2 THREADS PER WARP SO YOU NEED TO OVER-ALLOCATE TO GET THEM BACK IN

    // To prove correctness, our algorithm is as follows:
    // Inputs will be groups of FOUR (with common high-bits) that have a single
    // unique lower-4 bit set (ie a group could be: 0xff001, 0xff002, 0xff004, 0xff008)
    // The operation will BITWISE AND all pairs of the group of 4 to make SIX
    // outputs in the order of 0-1, 1-2, 2-3, 0-2, 1-3, 0-3 (for the above, this
    // makes the outputs: 0xff003, 0xff006, 0xff00c, 0xff005, 0xff00a, 0xff009)

    // GLOBAL thread ID
    // Warp-local ID
    // Shuffle-6 lane ID
    // GLOBAL lane depth
    bounded_int threadID = (blockIdx.x * blockDim.x) + threadIdx.x,
                warpID = (threadIdx.x % 32),
                laneID = warpID % 6,
                laneDepth = 3*(((threadID / 32)*5) + (warpID / 6));

    // Early-exit:
    //      Threads that would read beyond max_cells at their base value
    //      The two straggler threads of each warp
    if (laneDepth > max_cells || warpID > 29) return;

    // Adjust output pointers per-thread to ensure they don't overwite one
    // another's data. Each thread outputs 6 values per iteration, with at most
    // 3 iterations, so the skip value is 18
    out += (laneDepth * 18);

    // Read your FIRST value from global memory
    // laneDepth *= 4 to use vector-addressing, not set permanently due to
    // subsequent references that need the quadruplet ID later
    bounded_int global_read = in[(laneDepth*4) + laneID];

    // ITERATION 1: First quadruplet -- guaranteed to be useful by early-exit above
    bounded_int quad0 = __shfl_sync(0xfffffffc, global_read, 0, 6),
                quad1 = __shfl_sync(0xfffffffc, global_read, 1, 6),
                quad2 = __shfl_sync(0xfffffffc, global_read, 2, 6),
                quad3 = __shfl_sync(0xfffffffc, global_read, 3, 6);
    __syncthreads();
    // ALGORITHM FOR ITERATION 1
    bitwiseAND_64(quad0, quad1, quad2, quad3, laneID, out);

    // ITERATION 2: Second quadruplet -- early-exit if NOT useful
    if (laneDepth+1 > max_cells) return;
    quad0 = __shfl_sync(0xfffffffc, global_read, 4, 6);
    quad1 = __shfl_sync(0xfffffffc, global_read, 5, 6);
    __syncthreads();
    // First read is exhausted, make the second global read now with
    // incremented offset
    laneID += 6;
    global_read = in[(laneDepth*4) + laneID];
    quad2 = __shfl_sync(0xfffffffc, global_read, 0, 6);
    quad3 = __shfl_sync(0xfffffffc, global_read, 1, 6);
    __syncthreads();
    // ALGORITHM FOR ITERATION 2
    // Adjust pointers to not overwrite previous iteration's data
    out += 6;
    bitwiseAND_64(quad0, quad1, quad2, quad3, laneID, out);

    // ITERATION 3: Third quadruplet -- early-exit if NOT useful
    if (laneDepth+2 > max_cells) return;
    quad0 = __shfl_sync(0xfffffffc, global_read, 2, 6);
    quad1 = __shfl_sync(0xfffffffc, global_read, 3, 6);
    quad2 = __shfl_sync(0xfffffffc, global_read, 4, 6);
    quad3 = __shfl_sync(0xfffffffc, global_read, 5, 6);
    __syncthreads();
    // ALGORITHM FOR ITERATION 3
    // Adjust pointers to not overwrite previous iteration's data
    out += 6;
    bitwiseAND_64(quad0, quad1, quad2, quad3, laneID, out);
}

const bounded_int INFO_LINENO = __LINE__+1;
// Test program's bitwise AND requires separation between low/high bits,
// naive starter differentiation for high bits limits you to 128 "cells"
// at most. To exceed this limit, use a larger datatype or adjust the
// high-bit masking algorithm. We start with UNSIGNED LONG INT (see type
// definitinition for "bounded_int" at the top of this program) as this is
// usually guaranteed to be 32-bit for the 128 "cell" limit in C++
// (ie: it isn't a short integer as permitted for default integers)
#define MAX_SUPPORTED 128
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
    host_input = new bounded_int[N_CELLS*4]; // Trivial input with attributable blame
    host_expect = new bounded_int[N_CELLS*6]; // Expected outputs
    host_receive = new bounded_int[N_CELLS*6]; // Retreive answer from GPU
    cudaMalloc((void**)&device_input, SIZE*4);
    cudaMalloc((void**)&device_output, SIZE*6);

    // Set up host memory
    for (bounded_int i = 0; i < N_CELLS * 6; i++) host_receive[i] = -1;
    for (bounded_int i = 0; i < N_CELLS; i++) {
        // HIGH BIT MASK is cell ID shifted over the low bits
        bounded_int hi_mask = i << 4;
        // Total answer is HIGH BIT MASK bitwise or'd with one-hot low bit
        host_input[(i*4)] = hi_mask | 0x1;
        host_input[(i*4)+1] = hi_mask | 0x2;
        host_input[(i*4)+2] = hi_mask | 0x4;
        host_input[(i*4)+3] = hi_mask | 0x8;
        // Expected answer is canonically sorted
        host_expect[(i*6)] = host_input[(i*4)] & host_input[(i*4)+1];
        host_expect[(i*6)+1] = host_input[(i*4)+1] & host_input[(i*4)+2];
        host_expect[(i*6)+2] = host_input[(i*4)+2] & host_input[(i*4)+3];
        host_expect[(i*6)+3] = host_input[(i*4)] & host_input[(i*4)+2];
        host_expect[(i*6)+4] = host_input[(i*4)+1] & host_input[(i*4)+3];
        host_expect[(i*6)+5] = host_input[(i*4)] & host_input[(i*4)+3];
    }
    // Copy input and blanks over to device
    cudaMemcpy(device_input, host_input, SIZE*4, cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, host_receive, SIZE*6, cudaMemcpyHostToDevice);


    // Set up launch configuration and execute kernel
    const bounded_int N_THREADS = 1024,
                      // 6 threads per cell -- 2 threads per warp will early-exit
                      // So for every warp (32 threads), you get 5 cells
                      cells_per_block = N_THREADS / 32 / 5;
    bounded_int N_BLOCKS = (N_CELLS+cells_per_block-1)/cells_per_block;
    std::cout << "Data ready, launch kernel with " << N_BLOCKS << " blocks of "
              << N_THREADS << " threads " << std::endl;
    kernel_64<<<N_BLOCKS, N_THREADS>>>(device_input, N_CELLS, device_output);
    cudaDeviceSynchronize();
    std::cout << "Kernel complete" << std::endl;
    cudaMemcpy(host_receive, device_output, SIZE*6, cudaMemcpyDeviceToHost);


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
    if (error > 0) std::cout << "Total errors: " << error << std::endl;
    else std::cout << "Validated" << std::endl;


    // Memory frees
    delete host_input;
    delete host_expect;
    delete host_receive;
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}

