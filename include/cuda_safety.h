#ifndef TETRA_CUDA_SAFETY
#define TETRA_CUDA_SAFETY

//#include <cuda.h> // idk, requires -lcuda
#include <cuda_runtime.h> // device management API, requires -lcudart
#include <iostream>

// Safety & wrappers
inline void cudaWarn(const char* file, int line, cudaError_t code, const char* debug) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Assertion Failed (" << code << ") " <<
                     "\"" << cudaGetErrorString(code) << "\" at " <<
                     file << ":" << line << std::endl;
        if (debug != NULL) std::cerr << debug << std::endl;
    }
}
inline void cudaAssert(const char* file, int line, cudaError_t code, const char* debug,
                bool abort) {
    cudaWarn(file, line, code, debug);
    if (code != cudaSuccess && abort) exit(code);
}
#define CUDA_WARN(code) { cudaWarn(__FILE__, __LINE__, code, nullptr); }
#define CUDA_WARN_MSG(code, msg) { cudaWarn(__FILE__, __LINE__, code, msg); }
#define CUDA_ASSERT(code) { cudaAssert(__FILE__, __LINE__, code, nullptr); }
#define CUDA_ASSERT_MSG(code, msg) { cudaAssert(__FILE__, __LINE__, code, msg); }

inline void kernelWarn(const char* file, int line, const char* debug) {
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed (" << code << ") " <<
                     "\"" << cudaGetErrorString(code) << "\" at " <<
                     file << ":" << line << std::endl;
        if (debug != NULL) std::cerr << debug << std::endl;
    }
}
inline void kernelAssert(const char* file, int line, const char* debug, bool abort) {
    kernelAssert(file, line, debug, false);
    if(abort) exit(EXIT_FAILURE);
}
#define KERNEL_WARN() { kernelWarn(__FILE__, __LINE__, nullptr); }
#define KERNEL_WARN_MSG(msg) { kernelWarn(__FILE__, __LINE__, msg); }
#define KERNEL_ASSERT() { kernelAssert(__FILE__, __LINE__, nullptr); }
#define KERNEL_ASSERT_MSG(msg) { kernelAssert(__FILE__, __LINE__, msg); }

inline void dummyWarn(const char* file, int line) {
    std::cout << "Dummy warning from " << file << ":" << line << std::endl;
}
#define DUMMY_WARN { dummyWarn(__FILE__, __LINE__); }

#endif

