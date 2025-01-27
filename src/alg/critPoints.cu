// Other files in this repository
#include "argparse.h" // Arguments and parse() -- to be swapped out!
#include "vtk_load.h" // TV_Data type and get_TV_from_VTK()
#include "cuda_safety.h" // Cuda/Kernel safety wrappers
#include "cuda_extraction.h" // make_*_GPU()
#include "metrics.h" // Timer class
#include "emoji.h" // Emoji definitions
// Include for testing, not for final
#ifdef VALIDATE_GPU
#include "cpu_extraction.h" // make_*() and elective_make_*()
#include "validate.h" // Check Host-vs-Device answers on relationships
#endif

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

/* CriticalPoints kernel should:
        1) Parallelize VV on second-dimension (can early-exit block if no data
           available or if a prefix-scan of your primary-dimension list shows
           that you are a duplicate)
        2) Read the scalar value used for point classification and classify
           yourself relative to your primary-dimension scalar value as upper or
           lower neighbor
        -- VV-PARALLEL SYNC REQUIRED --
        3) For all other threads sharing your neighborhood classification, scan
           their connectivity in VV. If you connect to at least one, then you
           share a component with that neighbor -- the lowest-ranked neighbor
           will log +1 component of this type and all others exit. If you fail
           to locate any connections to others in your class, then you have 2+
           components and are immediately a saddle -- increment your component
           counter and exit. It does not matter if this "over-counts" the
           number of components!
        -- VV-PARALLEL SYNC REQUIRED --
        4) Classification is performed as follows: Exactly 1 upper component is
           a maximum; exactly 1 lower component is a minimum; two or more upper
           or lower components is a saddle; other values are regular.
*/

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

    // Usually VE and VF are also mandatory, but CritPoints does not require
    // these relationships! Skip them!

    // OPTIONAL: VV (yellow) [TV' x TV]
    // REQUIRED for CritPoints
    #ifdef VALIDATE_GPU
    // CPU version ONLY required when we are validating results
    std::cout << PUSHPIN_EMOJI << "Using CPU to compute " YELLOW_COLOR "VV" RESET_COLOR << std::endl;
    timer.label_next_interval(YELLOW_COLOR "VV" RESET_COLOR " [CPU]");
    timer.tick();
    std::unique_ptr<VV_Data> VV = elective_make_VV(*TV, TV->nPoints, args);
    timer.tick_announce();
    #endif
    std::cout << PUSHPIN_EMOJI << "Using GPU to compute " YELLOW_COLOR "VV" RESET_COLOR << std::endl;
    timer.label_next_interval(YELLOW_COLOR "VV" RESET_COLOR " [GPU]");
    timer.tick();
    // Have to make a max VV guess
    vtkIdType max_VV_guess = get_approx_max_VV(*TV, TV->nPoints);
    device_VV * dvv = make_VV_GPU_return(*TV, TV->nCells, TV->nPoints,
                                         max_VV_guess, true, args);
    timer.tick_announce();

    // Critical Points
    timer.label_next_interval("Run " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    timer.tick();
    timer.tick_announce();
    #ifdef VALIDATE_GPU
    timer.label_next_interval("Validate " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    timer.tick();
    std::cerr << WARN_EMOJI << "No validation for critical points yet!" << std::endl;
    timer.tick_announce();
    #endif
}
