// Other files in this repository
#include "argparse.h" // arguments type and parse()
#include "vtk_load.h" // TV_Data type and get_TV_from_VTK()
#include "cpu_extraction.h" // *_Data types and make_*() / elective_make_*()
#include "cuda_safety.h" // Cuda/Kernel safety wrappers
#include "cuda_extraction.h" // make_*_GPU()
#include "validate.h" // check_host_vs_device_*()
#include "metrics.h" // Timer class
#include "emoji.h" // Emoji definitions

__global__ void dummy_kernel(void) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
}

int main(int argc, char *argv[]) {
    Timer timer;
    arguments args;
    parse(argc, argv, args);
    timer.tick();
    timer.interval("Argument parsing");

    // GPU initialization
    {
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

    std::cout << PUSHPIN_EMOJI << "Parsing vtu file: " << args.fileName << std::endl;
    timer.label_next_interval("TV from VTK");
    timer.tick();
    // Should utilize VTK API and then de-allocate all of its heap
    std::unique_ptr<TV_Data> tv_relationship = get_TV_from_VTK(args);
    timer.tick_announce();

    // Adapted from TTK Explicit Triangulation
    std::cout << PUSHPIN_EMOJI << "Building edges..." << std::endl;
    timer.label_next_interval("TE and VE [CPU]");
    timer.tick();
    std::unique_ptr<TE_Data> cellEdgeList = std::make_unique<TE_Data>(tv_relationship->nCells);
    std::unique_ptr<VE_Data> edgeTable = std::make_unique<VE_Data>(tv_relationship->nPoints);
    // The TE relationship simultaneously informs VE, so make both at once
    vtkIdType edgeCount = make_TE_and_VE(*tv_relationship,
                                         *cellEdgeList,
                                         *edgeTable);
    timer.tick_announce();
    std::cout << OK_EMOJI << "Built " << edgeCount << " edges." << std::endl;

    // Scope to de-allocate unique ptrs
    {
        // allocate & fill edgeList in parallel (EV)
        timer.label_next_interval("EV [CPU]");
        timer.tick();
        std::unique_ptr<EV_Data> EV = elective_make_EV(*tv_relationship,
                                                       *edgeTable,
                                                       edgeCount,
                                                       args);
        timer.tick_announce();

        std::cout << PUSHPIN_EMOJI << "Using GPU to compute EV" << std::endl;
        timer.label_next_interval("EV [GPU]");
        timer.tick();
        std::unique_ptr<EV_Data> device_EV = make_EV_GPU(*edgeTable,
                                                         tv_relationship->nPoints,
                                                         edgeCount,
                                                         args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        timer.label_next_interval("Validate GPU EV");
        timer.tick();
        if(check_host_vs_device_EV(*EV, *device_EV)) {
            std::cout << OK_EMOJI << "GPU EV results validated by CPU" << std::endl;
        }
        else {
            std::cerr << EXCLAIM_EMOJI << "ALERT! GPU EV results do NOT match CPU results!" << std::endl;
        }
        timer.tick_announce();
        #endif
    }

    std::cout << FLAG_EMOJI << "--Should not auto-print timers past this point--" << std::endl;
    // Scope to de-allocate unique ptrs
    {
        timer.label_next_interval("ET [CPU]");
        timer.tick();
        // we can also get edgeStars from cellEdgeList (ET)
        std::unique_ptr<ET_Data> ET = elective_make_ET(*cellEdgeList,
                                                       edgeCount,
                                                       args);
        timer.tick();
    }
    // Make faces, which we define based on cells and vertices so we simultaneously define TF and FV
    std::cout << PUSHPIN_EMOJI << "Building faces..." << std::endl;
    timer.label_next_interval("TF and VF [CPU]");
    timer.tick();
    std::unique_ptr<TF_Data> cellFaceList = std::make_unique<TF_Data>(tv_relationship->nCells);
    std::unique_ptr<VF_Data> faceTable = std::make_unique<VF_Data>(tv_relationship->nPoints);
    //faceTable.get()->resize(); // guarantee space AND make indexing valid
    vtkIdType faceCount = make_TF_and_VF(*tv_relationship,
                                         *cellFaceList,
                                         *faceTable);
    timer.tick();
    std::cout << OK_EMOJI << "Built " << faceCount << " faces." << std::endl;
    timer.tick(); // bonus tick -- open interval

    return 0;
}

