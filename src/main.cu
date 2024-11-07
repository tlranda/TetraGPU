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

    // MANDATORY: TV (green) [from storage]
    std::cout << PUSHPIN_EMOJI << "Parsing vtu file: " << args.fileName
              << std::endl;
    timer.label_next_interval("TV from VTK");
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
    if (args.build_TE()) {
        timer.label_next_interval("TE and VE [CPU]");
        edgeCount = make_TE_and_VE(*TV, *TE, *VE);
    }
    else {
        timer.label_next_interval("VE [CPU]");
        edgeCount = make_VE(*TV, *VE);
    }
    timer.tick_announce();
    std::cout << OK_EMOJI << "Built " << edgeCount << " edges." << std::endl;

    // OPTIONAL: EV (green) [VE']
    if (args.build_EV()) {
        // CPU
        timer.label_next_interval("EV [CPU]");
        timer.tick();
        std::unique_ptr<EV_Data> EV = elective_make_EV(*VE, TV->nPoints,
                                                       edgeCount, args);
        timer.tick_announce();

        // GPU
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute EV" << std::endl;
        timer.label_next_interval("EV [GPU]");
        timer.tick();
        std::unique_ptr<EV_Data> device_EV = make_EV_GPU(*VE, TV->nPoints,
                                                         edgeCount, args);
        timer.tick_announce();

        #ifdef VALIDATE_GPU
        // VALIDATION
        if (args.validate()) {
            timer.label_next_interval("Validate GPU EV");
            timer.tick();
            if (check_host_vs_device_EV(*EV, *device_EV)) {
                std::cout << OK_EMOJI << "GPU EV results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU EV results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
    }

    // OPTIONAL: TE (green) [TV x VE]
    if (args.build_TE()) {
        // CPU already prepared, GPU
        std::cout << PUSHPIN_EMOJI << "Using GPU to compute TE" << std::endl;
        std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
        /*
        timer.label_next_interval("TE [GPU]");
        timer.tick();
        std::unique_ptr<TE_Data> device_TE = make_TE_GPU(*TE, *VE, n_edges, args);
        timer.tick_announce();

        #ifdef VALIDATE_GPU
        // VALIDATION
        if (args.validate()) {
            timer.label_next_interval("Validate GPU TE");
            timer.tick();
            if (check_host_vs_device_TE(*TE, *device_TE)) {
                std::cout << OK_EMOJI << "GPU TE results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU TE results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
        */

        // OPTIONAL: ET (red) [TE' == (TV x VE)']
        if (args.build_ET()) {
            // CPU
            timer.label_next_interval("ET [CPU]");
            timer.tick();
            // we can also get edgeStars from TE (ET)
            std::unique_ptr<ET_Data> ET = elective_make_ET(*TE, edgeCount, args);
            timer.tick_announce();

            // GPU
            std::cout << PUSHPIN_EMOJI << "Using GPU to compute ET" << std::endl;
            std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
            /*
            timer.label_next_interval("ET [GPU]");
            timer.tick();
            std::unique_ptr<ET_Data> device_ET = make_ET_GPU(*TE, args);
            timer.tick_announce();

            #ifdef VALIDATE_GPU
            // VALIDATION
            if (args.validate()) {
                timer.label_next_interval("Validate GPU ET");
                timer.tick();
                if (check_host_vs_device_ET(*ET, *device_ET)) {
                    std::cout << OK_EMOJI << "GPU ET results validated by CPU"
                              << std::endl;
                }
                else {
                    std::cerr << EXCLAIM_EMOJI
                              << "ALERT! GPU ET results do NOT match CPU results!"
                              << std::endl;
                }
                timer.tick_announce();
            }
            #endif
            */
        }
    }

    // MANDATORY: VF (red) [TV walk with semantic ordering to prevent dupes]
    // OPTIONAL: TF (green) [TV walk with semantic ordering to prevent dupes]
    std::cout << PUSHPIN_EMOJI << "Building faces..." << std::endl;
    std::unique_ptr<TF_Data> TF = std::make_unique<TF_Data>(TV->nCells);
    std::unique_ptr<VF_Data> VF = std::make_unique<VF_Data>(TV->nPoints);
    vtkIdType faceCount;
    timer.tick();
    if (args.build_TF()) {
        timer.label_next_interval("TF and VF [CPU]");
        faceCount = make_TF_and_VF(*TV, *TF, *VF);
        /*
        for (vtkIdType i = 0; i < TV->nPoints; i++) {
            std::cout << "Vertex " << i << std::endl;
            for (auto face : (*VF)[i]) {
                std::cout << "\tFirst Face " << face.id << std::endl;
                break;
            }
            std::cout << "\tn_faces " << (*VF)[i].size() << std::endl;
        }
        */
    }
    else {
        timer.label_next_interval("VF [CPU]");
        faceCount = make_VF(*TV, *VF);
    }
    timer.tick_announce();
    std::cout << OK_EMOJI << "Built " << faceCount << " faces." << std::endl;

    // OPTIONAL: TF (green) [TV x VF]
    if (args.build_TF()) {
        std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
        timer.label_next_interval("TF [GPU]");
        timer.tick();
        std::unique_ptr<TF_Data> device_TF = make_TF_GPU(*TV, *VF, TV->nPoints,
                                                         faceCount, TV->nCells,
                                                         args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        if (args.validate()) {
            timer.label_next_interval("Validate GPU TF");
            timer.tick();
            if(check_host_vs_device_TF(*TF, *device_TF)) {
                std::cout << OK_EMOJI << "GPU TF results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU TF results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
    }

    // OPTIONAL: FV (green) [VF']
    if (args.build_FV()) {
        std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
        /*
        timer.label_next_interval("FV [GPU]");
        timer.tick();
        std::unique_ptr<FV_Data> device_FV = make_FV_GPU(*VF, TV->nPoints,
                                                         faceCount, args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        if (args.validate()) {
            timer.label_next_interval("Validate GPU FV");
            timer.tick();
            if(check_host_vs_device_FV(*FV, *device_FV)) {
                std::cout << OK_EMOJI << "GPU FV results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU FV results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
        */
    }

    // OPTIONAL: FE (green) [VF' x VE]
    if (args.build_FE()) {
        std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
        /*
        timer.label_next_interval("FE [GPU]");
        timer.tick();
        std::unique_ptr<FE_Data> device_FE = make_FE_GPU(*VF, *VE, TV->nPoints,
                                                         faceCount, args);
        timer.tick_announce();
        #ifdef VALIDATE_GPU
        if (args.validate()) {
            timer.label_next_interval("Validate GPU FE");
            timer.tick();
            if(check_host_vs_device_FE(*FE, *device_FE)) {
                std::cout << OK_EMOJI << "GPU FE results validated by CPU"
                          << std::endl;
            }
            else {
                std::cerr << EXCLAIM_EMOJI
                          << "ALERT! GPU FE results do NOT match CPU results!"
                          << std::endl;
            }
            timer.tick_announce();
        }
        #endif
        */
    }

    // MIA: TT (yellow) [TV x TV']
    // MIA: FF (yellow) [TF' x TF]
    // MIA: EE (yellow) [EV' x VE]
    // MIA: VV (yellow) [TV' x TV]
    // MIA: FT (red) [(TV x VF)' | VF' x TV']
    // MIA: EF (red) [(TV x VE)' | VE' x TV']
    // MIA: VT (red) [TV']

    // Critical Points: FT = TF', VV = (V*') x (*V) for any of TV, FV, EV, VF, VE
    timer.tick(); // bonus tick -- open interval

    return 0;
}

