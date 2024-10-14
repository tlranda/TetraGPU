// Other files in this repository
#include "argparse.h" // arguments type and parse()
#include "vtk_load.h" // TV_Data type and get_TV_from_VTK()
#include "cpu_extraction.h" // *_Data types and make_*() / elective_make_*()
#include "cuda_safety.h" // Cuda/Kernel safety wrappers
#include "cuda_extraction.h" // make_*_GPU()

__global__ void kernel(void) {
    int tid = threadIdx.x;
    printf("%d\n", tid);
}

bool check_host_vs_device_EV(const EV_Data host_EV,
                             const EV_Data device_EV) {
    // Validate that host and device agree on EV Data
    // The vectors do NOT need to be ordered the same, they just need to agree
    // On the number of edges and vertex contents of each edge
    if (host_EV.size() != device_EV.size()) {
        std::cerr << "Device EV size != Host EV size" << std::endl;
        return false;
    }
    int n_printed = 0;
    for (auto EV_Array : host_EV) {
        auto index = std::find(begin(device_EV), end(device_EV), EV_Array);
        if (index == std::end(device_EV)) {
            std::cerr << "Could not find edge (" << EV_Array[0] << ", " <<
                         EV_Array[1] << ") in device EV!" << std::endl;
            return false;
        }
        else {
            if (n_printed < 10) {
                std::cout << EV_Array[0] << " " << EV_Array[1] << std::endl;
                n_printed++;
            }
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    arguments args;
    parse(argc, argv, args);

    std::cout << "Parsing vtu file: " << args.fileName << std::endl;
    // Should utilize VTK API and then de-allocate all of its heap
    std::unique_ptr<TV_Data> tv_relationship = get_TV_from_VTK(args);

    // Adapted from TTK Explicit Triangulation
    std::cout << "Building edges..." << std::endl;
    std::unique_ptr<TE_Data> cellEdgeList = std::make_unique<TE_Data>(tv_relationship->nCells);
    std::unique_ptr<VE_Data> edgeTable = std::make_unique<VE_Data>(tv_relationship->nPoints);
    // The TE relationship simultaneously informs VE, so make both at once
    vtkIdType edgeCount = make_TE_and_VE(*tv_relationship,
                                         *cellEdgeList,
                                         *edgeTable);
    std::cout << "Built " << edgeCount << " edges." << std::endl;

    // allocate & fill edgeList in parallel (EV)
    std::unique_ptr<EV_Data> EV = elective_make_EV(*tv_relationship,
                                                   *edgeTable,
                                                   edgeCount,
                                                   args);

    // Scope to de-allocate unique ptr
    {
        std::cout << "Using GPU to compute EV" << std::endl;
        std::unique_ptr<EV_Data> device_EV = make_EV_GPU(*tv_relationship,
                                                         *EV,
                                                         edgeCount,
                                                         args);
        if(check_host_vs_device_EV(*EV, *device_EV)) {
            std::cout << "GPU EV results validated by CPU" << std::endl;
        }
        else {
            std::cerr << "ALERT! GPU EV results do NOT match CPU results!" << std::endl;
        }
    }

    // we can also get edgeStars from cellEdgeList (ET)
    std::unique_ptr<ET_Data> ET = elective_make_ET(*cellEdgeList,
                                                   edgeCount,
                                                   args);

    // Make faces, which we define based on cells and vertices so we simultaneously define TF and FV
    std::cout << "Building faces..." << std::endl;
    std::unique_ptr<TF_Data> cellFaceList = std::make_unique<TF_Data>(tv_relationship->nCells);
    std::unique_ptr<VF_Data> faceTable = std::make_unique<VF_Data>();
    faceTable.get()->resize(tv_relationship->nPoints); // guarantee space AND make indexing valid
    vtkIdType faceCount = make_TF_and_VF(*tv_relationship,
                                         *cellFaceList,
                                         *faceTable);
    std::cout << "Built " << faceCount << " faces." << std::endl;

    kernel<<<10,120993240>>>();
    DUMMY_WARN
    //KERNEL_WARN();
    //CUDA_WARN(
    cudaDeviceSynchronize();
    //);
    return 0;
}

