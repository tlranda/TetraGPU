// Other files in this repository
#include "argparse.h" // arguments type and parse()
#include "vtk_load.h" // TV_Data type and get_TV_from_VTK()
#include "cpu_extraction.h" // *_Data types and make_*() / elective_make_*()
#include "cuda_safety.h" // Cuda/Kernel safety wrappers
#include "cuda_extraction.h" // make_*_GPU()
#include "validate.h" // check_host_vs_device_*()
#include "metrics.h" // Timer class
#include "emoji.h" // Emoji definitions

/* Drives relationship creation and possibly validation (if enabled and called
   for). The dummy kernel is present to ensure CUDA context creation does not
   interfere with metric timing etc regardless of how you decide to collect
   the performance data.

   Data is loaded via VTK format (Only .vtu file type is tested (unstructured
   tetrahedral mesh); I can generate the ascii one but valid binary files
   should be OK too).

   We repeat the pattern as follows:
    * Create any mandatory data on CPU (ie: required by GPU to compute target
        relationship)
    * Elect to create the CPU version of the data (raw performance differential
        and possibly used for validation later on)
    * Create the GPU version of the data and retrieve its host-arranged version
    * If validating, validate the GPU answer using CPU results
*/

__global__ void dummy_kernel(void) {
    //int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
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
    std::unique_ptr<TV_Data> TV = get_TV_from_VTK(args);
    timer.tick_announce();

    // MANDATORY: VE (red) [TV walk with semantic ordering to prevent dupes]
    // OPTIONAL: TE (green) [TV walk with semantic ordering to prevent dupes]
    std::cout << PUSHPIN_EMOJI << "Building edges..." << std::endl;
    std::unique_ptr<TE_Data> TE = std::make_unique<TE_Data>(TV->nCells);
    std::unique_ptr<VE_Data> VE = std::make_unique<VE_Data>(TV->nPoints);
    timer.tick();
    vtkIdType edgeCount;
    if (args.build_TE() || args.build_ET()) {
        timer.label_next_interval(GREEN_COLOR "TE" RESET_COLOR " and " RED_COLOR "VE" RESET_COLOR " [CPU]");
        edgeCount = make_TE_and_VE(*TV, *TE, *VE);
    }
    else {
        timer.label_next_interval(RED_COLOR "VE" RESET_COLOR " [CPU]");
        edgeCount = make_VE(*TV, *VE);
    }
    timer.tick_announce();
    std::cout << OK_EMOJI << "Built " << edgeCount << " edges." << std::endl;

    // OPTIONAL: EV (green) [VE']
    if (args.build_EV()) {
        // CPU
        timer.label_next_interval(GREEN_COLOR "EV" RESET_COLOR " [CPU]");
        timer.tick();
        std::unique_ptr<EV_Data> EV = elective_make_EV(*VE, TV->nPoints,
                                                       edgeCount, args);
        timer.tick_announce();

        // GPU
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute " GREEN_COLOR "EV" RESET_COLOR << std::endl;
        timer.label_next_interval(GREEN_COLOR "EV" RESET_COLOR " [GPU]");
        timer.tick();
        std::unique_ptr<EV_Data> device_EV = make_EV_GPU(*VE, TV->nPoints,
                                                         edgeCount, false, args);
        timer.tick_announce();

        #ifdef VALIDATE_GPU
        // VALIDATION
        if (args.validate()) {
            timer.label_next_interval("Validate GPU " GREEN_COLOR "EV" RESET_COLOR);
            timer.tick();
            if (check_host_vs_device_EV(*EV, *device_EV)) {
                std::cout << OK_EMOJI << "GPU " GREEN_COLOR "EV" RESET_COLOR " results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU " GREEN_COLOR "EV" RESET_COLOR " results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
    }

    // OPTIONAL: TE (green) [TV x VE]
    // Required for ET as well
    if (args.build_TE() || args.build_ET()) {
        // CPU already prepared, GPU
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute " GREEN_COLOR "TE" RESET_COLOR << std::endl;
        timer.label_next_interval(GREEN_COLOR "TE" RESET_COLOR " [GPU]");
        timer.tick();
        std::unique_ptr<TE_Data> device_TE = make_TE_GPU(*TV, *VE, TV->nPoints,
                                                         edgeCount, TV->nCells,
                                                         false, args);
        timer.tick_announce();

        #ifdef VALIDATE_GPU
        // VALIDATION
        if (args.validate() && args.build_TE()) {
            timer.label_next_interval("Validate GPU " GREEN_COLOR "TE" RESET_COLOR);
            timer.tick();
            if (check_host_vs_device_TE(*TE, *device_TE)) {
                std::cout << OK_EMOJI << "GPU " GREEN_COLOR "TE" RESET_COLOR " results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU " GREEN_COLOR "TE" RESET_COLOR " results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif

        // OPTIONAL: ET (red) [TE' == (TV x VE)']
        if (args.build_ET()) {
            // CPU
            timer.label_next_interval(RED_COLOR "ET" RESET_COLOR " [CPU]");
            timer.tick();
            // we can also get edgeStars from TE (ET)
            std::unique_ptr<ET_Data> ET = elective_make_ET(*TE, edgeCount, args);
            timer.tick_announce();

            // GPU
            std::cout << PUSHPIN_EMOJI << "Using GPU to compute " RED_COLOR "ET" RESET_COLOR << std::endl;
            std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
            timer.label_next_interval(RED_COLOR "ET" RESET_COLOR " [GPU]");
            timer.tick();
            std::unique_ptr<ET_Data> device_ET = make_ET_GPU(*TV, *VE, TV->nPoints,
                                                             edgeCount, false,
                                                             args);
            timer.tick_announce();

            #ifdef VALIDATE_GPU
            // VALIDATION
            if (args.validate()) {
                timer.label_next_interval("Validate GPU " RED_COLOR "ET" RESET_COLOR);
                timer.tick();
                if (check_host_vs_device_ET(*ET, *device_ET)) {
                    std::cout << OK_EMOJI << "GPU " RED_COLOR "ET" RESET_COLOR " results validated by CPU"
                              << std::endl;
                }
                else {
                    std::cerr << EXCLAIM_EMOJI
                              << "ALERT! GPU " RED_COLOR "ET" RESET_COLOR " results do NOT match CPU results!"
                              << std::endl;
                }
                timer.tick_announce();
            }
            #endif
        }
    }

    // MANDATORY: VF (red) [TV walk with semantic ordering to prevent dupes]
    // OPTIONAL: TF (green) [TV walk with semantic ordering to prevent dupes]
    std::cout << PUSHPIN_EMOJI << "Building faces..." << std::endl;
    std::unique_ptr<TF_Data> TF = std::make_unique<TF_Data>();
    std::unique_ptr<VF_Data> VF = std::make_unique<VF_Data>(TV->nPoints);
    vtkIdType faceCount;
    timer.tick();
    if (args.build_TF()) {
        // RESIZE initializes the std::arrays in this frame and permits passing,
        // to other functions; if you use reserve instead, the allocation is
        // only valid during make_TF_and_VF() and afterwards you will get valid
        // and (likely)defined behavior of all 0's :(
        TF->resize(TV->nCells);
        timer.label_next_interval(GREEN_COLOR "TF" RESET_COLOR " and " RED_COLOR "VF" RESET_COLOR " [CPU]");
        faceCount = make_TF_and_VF(*TV, *TF, *VF);
    }
    else {
        timer.label_next_interval(RED_COLOR "VF" RESET_COLOR " [CPU]");
        faceCount = make_VF(*TV, *VF);
    }
    timer.tick_announce();
    std::cout << OK_EMOJI << "Built " << faceCount << " faces." << std::endl;

    // OPTIONAL: TF (green) [TV x VF]
    if (args.build_TF()) {
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute " GREEN_COLOR "TF" RESET_COLOR << std::endl;
        timer.label_next_interval(GREEN_COLOR "TF" RESET_COLOR " [GPU]");
        timer.tick();
        std::unique_ptr<TF_Data> device_TF = make_TF_GPU(*TV, *VF, TV->nPoints,
                                                         faceCount, TV->nCells,
                                                         false, args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        if (args.validate()) {
            timer.label_next_interval("Validate GPU " GREEN_COLOR "TF" RESET_COLOR);
            timer.tick();
            if(check_host_vs_device_TF(*TF, *device_TF)) {
                std::cout << OK_EMOJI << "GPU " GREEN_COLOR "TF" RESET_COLOR " results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU " GREEN_COLOR "TF" RESET_COLOR " results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
    }

    // OPTIONAL: FV (green) [VF']
    if (args.build_FV()) {
        std::cout << PUSHPIN_EMOJI << "Using CPU to compute " GREEN_COLOR "FV" RESET_COLOR << std::endl;
        timer.label_next_interval(GREEN_COLOR "FV" RESET_COLOR " [CPU]");
        timer.tick();
        std::unique_ptr<FV_Data> FV = elective_make_FV(*VF, faceCount, args);
        timer.tick_announce();
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute " GREEN_COLOR "FV" RESET_COLOR << std::endl;
        timer.label_next_interval(GREEN_COLOR "FV" RESET_COLOR " [GPU]");
        timer.tick();
        std::unique_ptr<FV_Data> device_FV = make_FV_GPU(*VF, TV->nPoints,
                                                         faceCount, false, args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        if (args.validate()) {
            timer.label_next_interval("Validate GPU " GREEN_COLOR "FV" RESET_COLOR);
            timer.tick();
            if(check_host_vs_device_FV(*FV, *device_FV)) {
                std::cout << OK_EMOJI << "GPU " GREEN_COLOR "FV" RESET_COLOR " results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU " GREEN_COLOR "FV" RESET_COLOR " results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
    }

    // OPTIONAL: FE (green) [VF' x VE]
    if (args.build_FE()) {
        std::cout << PUSHPIN_EMOJI << "Using CPU to compute " GREEN_COLOR "FE" RESET_COLOR << std::endl;
        timer.label_next_interval(GREEN_COLOR "FE" RESET_COLOR " [CPU]");
        timer.tick();
        std::unique_ptr<FE_Data> FE = elective_make_FE(*VF, *VE, TV->nPoints,
                                                       edgeCount, faceCount,
                                                       args);
        timer.tick_announce();
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute " GREEN_COLOR "FE" RESET_COLOR << std::endl;
        std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
        timer.label_next_interval(GREEN_COLOR "FE" RESET_COLOR " [GPU]");
        timer.tick();
        std::unique_ptr<FE_Data> device_FE = make_FE_GPU(*VF, *VE, TV->nPoints,
                                                         edgeCount, faceCount,
                                                         false, args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        if (args.validate()) {
            timer.label_next_interval("Validate GPU " GREEN_COLOR "FE" RESET_COLOR);
            timer.tick();
            if(check_host_vs_device_FE(*FE, *device_FE)) {
                std::cout << OK_EMOJI << "GPU " GREEN_COLOR "FE" RESET_COLOR " results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU " GREEN_COLOR "FE" RESET_COLOR " results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
    }

    // OPTIONAL: FT (red) [(TV x VF)' | VF' x TV']
    if (args.build_FT()) {
        std::cerr << EXCLAIM_EMOJI << "FT not implemented yet" << std::endl;
    }
    // OPTIONAL: EF (red) [(TV x VE)' | VE' x TV']
    if (args.build_EF()) {
        std::cerr << EXCLAIM_EMOJI << "EF not implemented yet" << std::endl;
    }
    // OPTIONAL: VT (red) [TV']
    if (args.build_VT()) {
        std::cerr << EXCLAIM_EMOJI << "VT not implemented yet" << std::endl;
    }
    // OPTIONAL: TT (yellow) [TV x TV']
    if (args.build_TT()) {
        std::cerr << EXCLAIM_EMOJI << "TT not implemented yet" << std::endl;
    }
    // OPTIONAL: FF (yellow) [TF' x TF]
    if (args.build_FF()) {
        std::cerr << EXCLAIM_EMOJI << "FF not implemented yet" << std::endl;
    }
    // OPTIONAL: EE (yellow) [EV' x VE]
    if (args.build_EE()) {
        std::cerr << EXCLAIM_EMOJI << "EE not implemented yet" << std::endl;
    }
    // OPTIONAL: VV (yellow) [TV' x TV]
    if (args.build_VV()) {
        std::cout << PUSHPIN_EMOJI << "Using CPU to compute " YELLOW_COLOR "VV" RESET_COLOR << std::endl;
        timer.label_next_interval(YELLOW_COLOR "VV" RESET_COLOR " [CPU]");
        timer.tick();
        std::unique_ptr<VV_Data> VV = elective_make_VV(*TV, TV->nPoints, args);
        timer.tick_announce();
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute " YELLOW_COLOR "VV" RESET_COLOR << std::endl;
        timer.label_next_interval(YELLOW_COLOR "VV" RESET_COLOR " [GPU]");
        timer.tick();
        std::unique_ptr<VV_Data> device_VV = make_VV_GPU(*TV, TV->nCells,
                                                         TV->nPoints, false, args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        if (args.validate()) {
            timer.label_next_interval("Validate GPU " YELLOW_COLOR "VV" RESET_COLOR);
            timer.tick();
            if (check_host_vs_device_VV(*VV, *device_VV)) {
                std::cout << OK_EMOJI << "GPU " YELLOW_COLOR "VV" RESET_COLOR " results validated by CPU"
                          << std::endl;
            }
            else {
                    std::cerr << EXCLAIM_EMOJI
                              << "ALERT! GPU " YELLOW_COLOR "VV" RESET_COLOR " results do NOT match CPU results!"
                              << std::endl;
            }
            timer.tick_announce();
        }
        #endif
    }

    // Check the VTK data!
    timer.label_next_interval(CYAN_COLOR "VTK test" RESET_COLOR " [CPU]");
    timer.tick();
    double sum = 0.0;
    for (vtkIdType i = 0; i < TV->nPoints; i++) {
        sum += TV->vertexAttributes[i];
    }
    std::cout << "Attribute sum: " << sum << std::endl;
    timer.tick_announce();

    timer.tick(); // bonus tick -- open interval to demonstrate behavior!

    return 0;
}

