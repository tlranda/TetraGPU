#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

void transferDataAsync(int streamCount, void** devPtrs, void** hostPtrs, size_t dataSize, cudaStream_t* streams) {
    for (int i = 0; i < streamCount; i++) {
        cudaMemcpyAsync(devPtrs[i], hostPtrs[i], dataSize / streamCount, cudaMemcpyHostToDevice, streams[i]);
    }

    for (int i = 0; i < streamCount; i++) {
        cudaStreamSynchronize(streams[i]);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_size_in_gb> <stream_count> [use_multithreading]" << std::endl;
        return 1;
    }

    float dataSizeInGB = std::stof(argv[1]);
    int streamCount = std::stoi(argv[2]);
    bool useMultithreading = (argc > 3 && std::string(argv[3]) == "1");

    // Context creation?
    cudaFree(0);
    size_t dataSizeInBytes = static_cast<size_t>(dataSizeInGB * 1024 * 1024 * 1024);
    size_t dataPerStreamInBytes = dataSizeInBytes / streamCount;

    void** devPtrs = new void*[streamCount];
    void** hostPtrs = new void*[streamCount];
    cudaStream_t* streams = new cudaStream_t[streamCount];

    for (int i = 0; i < streamCount; i++) {
        cudaMalloc(&devPtrs[i], dataPerStreamInBytes);
        hostPtrs[i] = malloc(dataPerStreamInBytes);
        cudaStreamCreate(&streams[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (useMultithreading) {
        std::vector<std::thread> threads;
        for (int i = 0; i < streamCount; i++) {
            threads.emplace_back([&, i] {
                transferDataAsync(1, &devPtrs[i], &hostPtrs[i], dataPerStreamInBytes, &streams[i]);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        transferDataAsync(streamCount, devPtrs, hostPtrs, dataSizeInBytes, streams);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double bandwidth = static_cast<double>(dataSizeInBytes) / (duration * 1e-3) / (1024 * 1024 * 1024);

    std::cout << "Data size: " << dataSizeInGB << " GB" << std::endl;
    std::cout << "Stream count: " << streamCount << std::endl;
    std::cout << "Use multithreading: " << (useMultithreading ? "Yes" : "No") << std::endl;
    std::cout << "Transfer time: " << duration << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GiB/s" << std::endl;

    for (int i = 0; i < streamCount; i++) {
        cudaFree(devPtrs[i]);
        free(hostPtrs[i]);
        cudaStreamDestroy(streams[i]);
    }

    delete[] devPtrs;
    delete[] hostPtrs;
    delete[] streams;

    return 0;
}

