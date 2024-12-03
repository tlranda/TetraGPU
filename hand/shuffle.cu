#include<iostream>
#include<cuda.h>

__global__ void kernel(int *in, int *out0, int *out1, int *out2, int *out3) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Read your value and set distinguishable bad initial out values
    int my_read = in[tid], my_out0 = -2, my_out1 = -2, my_out2 = -2, my_out3 = -2;
    // All threads in warp participate, sharing `my_read` from the ith element in sub-groups of 4
    my_out0 = __shfl_sync(0xffffffff, my_read, 0, 4);
    my_out1 = __shfl_sync(0xffffffff, my_read, 1, 4);
    my_out2 = __shfl_sync(0xffffffff, my_read, 2, 4);
    my_out3 = __shfl_sync(0xffffffff, my_read, 3, 4);
    // Copy to output memory to ensure it can be host-validated
    out0[tid] = my_out0;
    out1[tid] = my_out1;
    out2[tid] = my_out2;
    out3[tid] = my_out3;
}

int main(void) {
    size_t SIZE = 32 * sizeof(int);
        // Trivial input that has attributable blame for mistakes
    int host_input[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31},
        // Ensure not-set values are observable
        host_blank[] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
        // Expected outputs per offset
        host_expect0[] = {0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12,16,16,16,16,20,20,20,20,24,24,24,24,28,28,28,28,28},
        host_expect1[] = {1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13,17,17,17,17,21,21,21,21,25,25,25,25,29,29,29,29,29},
        host_expect2[] = {2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14,18,18,18,18,22,22,22,22,26,26,26,26,30,30,30,30},
        host_expect3[] = {3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15,19,19,19,19,23,23,23,23,27,27,27,27,31,31,31,31},
        // Memory for retrieving answers
        host_receive0[32] = {-1}, host_receive1[32] = {-1}, host_receive2[32] = {-1}, host_receive3[32] = {-1},
        // Device variables
        *device_input, *device_out0, *device_out1, *device_out2, *device_out3;
    // Allocate device memory
    cudaMalloc((void**)&device_input, SIZE);
    cudaMalloc((void**)&device_out0, SIZE);
    cudaMalloc((void**)&device_out1, SIZE);
    cudaMalloc((void**)&device_out2, SIZE);
    cudaMalloc((void**)&device_out3, SIZE);
    // Copy input and blanks over
    cudaMemcpy(device_input, host_input, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out0, host_blank, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out1, host_blank, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out2, host_blank, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_out3, host_blank, SIZE, cudaMemcpyHostToDevice);
    // Execute kernel
    kernel<<<1,32>>>(device_input, device_out0, device_out1, device_out2, device_out3);
    cudaDeviceSynchronize();
    // Copy answers to host
    cudaMemcpy(host_receive0, device_out0, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_receive1, device_out1, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_receive2, device_out2, SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_receive3, device_out3, SIZE, cudaMemcpyDeviceToHost);
    // Check values vs expectation
    int error = 0;
    for (int i = 0; i < 32; i++) {
        if (host_receive0[i] != host_expect0[i]) {
            std::cerr << "Host 0-" << i << ": Expect " << host_expect0[i] << "; received " << host_receive0[i] << std::endl;
            error++;
        }
        if (host_receive1[i] != host_expect1[i]) {
            std::cerr << "Host 1-" << i << ": Expect " << host_expect1[i] << "; received " << host_receive1[i] << std::endl;
            error++;
        }
        if (host_receive2[i] != host_expect2[i]) {
            std::cerr << "Host 2-" << i << ": Expect " << host_expect2[i] << "; received " << host_receive2[i] << std::endl;
            error++;
        }
        if (host_receive3[i] != host_expect3[i]) {
            std::cerr << "Host 3-" << i << ": Expect " << host_expect3[i] << "; received " << host_receive3[i] << std::endl;
            error++;
        }
    }
    if (error > 0) std::cerr << "Total errors: " << error << std::endl;
    else std::cout << "Validated" << std::endl;

    return 0;
}

