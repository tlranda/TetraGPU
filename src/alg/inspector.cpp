#include "alg/inspector.h"

/* Drives an inspector to look at the target portion of the mesh for me.

   Mesh data is loaded via the VTK unstructured format (.vtu). Only scalar data
   is supported on the mesh at this time (not vectors on the mesh). We only
   classify points as regular, maximum, minimum, or saddles (we do not inspect
   sub-classes of saddles).
*/

// For parity with Paraview, a tie needs to be broken by lower vertex ID being "lower" than the other one
#define CLASSIFY 1 - (\
        (scalar_values[my_2d] == scalar_values[my_1d] ? \
         my_2d < my_1d : \
         scalar_values[my_2d] < scalar_values[my_1d]) \
        << 1)
#define MAXIMUM_CLASS 1
#define MINIMUM_CLASS 2
#define REGULAR_CLASS 3
#define SADDLE_CLASS  4
#define FORCED_BLOCK_IDX 4

void critPointsCPU(const vtkIdType * __restrict__ VV,
                   const unsigned long long * __restrict__ VV_index,
                   vtkIdType * __restrict__ valences,
                   const vtkIdType points,
                   const vtkIdType max_VV_guess,
                   const double * __restrict__ scalar_values,
                   unsigned int * __restrict__ classes) {
    const vtkIdType minTID = (max_VV_guess * FORCED_BLOCK_IDX),
                    maxTID = minTID + VV_index[FORCED_BLOCK_IDX],
                    my_1d = FORCED_BLOCK_IDX;
    /* GPU VV is non-deterministic, output it */
    for (vtkIdType tid = minTID; tid < maxTID; tid++) {
        std::cerr << CYAN_COLOR << "VV[" << tid << "] = " << VV[tid]
                  << RESET_COLOR << std::endl;
    }
    for (vtkIdType tid = minTID; tid < maxTID; tid++) {
        const vtkIdType my_2d = VV[tid];
        // Prefix scan and other early-exits
        if (VV_index[my_1d] <= 0 || my_2d < 0) {
            std::cerr << "TID " << tid << " has no work and exits" << std::endl;
            continue;
        }
        bool early_exit = false;
        for (vtkIdType i = minTID; i < tid; i++) {
            if (VV[i] == my_2d) {
                early_exit = true;
            }
        }
        if (early_exit) {
            std::cerr << "TID " << tid << " would work on " << my_2d << " but is preempted in VV and exits" << std::endl;
            continue;
        }
        // BEYOND THIS POINT, YOU ARE AN ACTUAL WORKER THREAD ON THE PROBLEM
        /* critPointsA */
        // Classify yourself relative to your 1d point
        const vtkIdType my_class = CLASSIFY;
        if (scalar_values[my_2d] == scalar_values[my_1d]) {
            std::cerr << "TIEBREAKER NEEDED" << std::endl;
        }
        valences[tid] = my_class;
        std::cerr << "Block " << FORCED_BLOCK_IDX << " Thread " << tid
                  << YELLOW_COLOR << " valence between " << my_1d << " and "
                  << my_2d << " is " << my_class << RESET_COLOR << std::endl;
    }
    for (vtkIdType tid = minTID; tid < maxTID; tid++) {
        const vtkIdType my_2d = VV[tid];
        // Prefix scan and other early-exits
        if (VV_index[my_1d] <= 0 || my_2d < 0) {
            continue;
        }
        bool early_exit = false;
        for (vtkIdType i = minTID; i < tid; i++) {
            if (VV[i] == my_2d) {
                early_exit = true;
            }
        }
        if (early_exit) {
            continue;
        }
        // BEYOND THIS POINT, YOU ARE AN ACTUAL WORKER THREAD ON THE PROBLEM
        // Classify yourself relative to your 1d point
        const vtkIdType my_class = valences[tid];
        /* critPointsB */
        const vtkIdType max_my_1d = (my_1d * max_VV_guess) + VV_index[my_1d];
        bool done = false, burdened = true;
        int inspect_step = 0;
        for(vtkIdType i = my_1d * max_VV_guess; !done && (i < max_my_1d); i++) {
            // Found same valence connected to 1D at a point with a lower index than you
            const vtkIdType candidate_component_2d = VV[i]; //[(my_1d*max_VV_guess)+(i-(my_1d*max_VV_guess))];
            if (valences[i] == my_class && candidate_component_2d < my_2d) {
                std::cerr << "Block " << FORCED_BLOCK_IDX << " Thread " << tid
                          << " Inspect " << inspect_step << " Possible shared "
                          << "component with VV[" << my_1d << "][" << i-(my_1d*max_VV_guess)
                          << "] (" << candidate_component_2d << ")" << std::endl;
                const vtkIdType start_2d = candidate_component_2d*max_VV_guess,
                                stop_2d  = start_2d + VV_index[candidate_component_2d];
                for(vtkIdType j = start_2d; !done && (j < stop_2d); j++) {
                    // Do you see your 2d in their 2d?
                    if (VV[j] == my_2d) {
                        // Shared component!
                        burdened = (candidate_component_2d > my_2d); // lower one writes!
                        //done = burdened; // EXPERIMENTAL: Only done if you're burdened
                        std::cerr << "Block " << FORCED_BLOCK_IDX << " Thread "
                                  << tid << " Inspect " << inspect_step
                                  << " Shares component at VV[" << candidate_component_2d
                                  << "][" << j-(candidate_component_2d*max_VV_guess)
                                  << "] (" << (my_class == -1 ? "Upper" : "Lower")
                                  << ", " << (burdened ? "Burdened to write" : "Not writing")
                                  << ")" << std::endl;
                    }
                    inspect_step++;
                }
            }
        }
        if (burdened) {
            const vtkIdType memoffset = ((my_1d*3)+((my_class+1)/2));
            const unsigned int old = classes[memoffset];
            classes[memoffset]++;
            std::cerr << "Block " << FORCED_BLOCK_IDX << " Thread " << tid
                      << RED_COLOR << " Writes " << (my_class == -1 ? "Upper" : "Lower")
                      << " component @ mem offset " << memoffset << " (old: "
                      << old << ")" << RESET_COLOR << std::endl;
            std::cerr << "Possible intersects" << std::endl;
            for (vtkIdType j = my_2d*max_VV_guess; j < (my_2d*max_VV_guess)+VV_index[my_2d]; j++) {
                std::cerr << "\t" << VV[j] << std::endl;
            }
        }
    }
    vtkIdType tid = minTID;
    /* critPointsC */
    // Limit classification to lowest-ranked thread for single write
    if (my_1d * max_VV_guess == tid) {
        const vtkIdType my_classes = my_1d*3;
        const unsigned int upper = classes[my_classes],
                           lower = classes[my_classes+1];
        std::cerr << "Block " << FORCED_BLOCK_IDX << " Thread " << tid
                  << " Verdict: my_1d (" << my_1d << ") has " << upper
                  << " upper and " << lower << " lower" << std::endl;
        if (upper >= 1 && lower == 0) classes[my_classes+2] = MINIMUM_CLASS;
        else if (upper == 0 && lower >= 1) classes[my_classes+2] = MAXIMUM_CLASS;
        else if (/* upper >= 1 and upper == lower /**/ upper == 1 && lower == 1 /**/) classes[my_classes+2] = REGULAR_CLASS;
        else classes[my_classes+2] = SADDLE_CLASS;
    }
}

void export_classes(unsigned int * classes, vtkIdType n_classes, arguments & args) {
    std::ofstream output_fstream; // Used for file handle to indicated name
    std::streambuf * output_buffer; // Buffer may point to stdout or file handle
    if (args.export_ == "") {
        // No export provided by user, write to stdout
        output_buffer = std::cout.rdbuf();
        std::cerr << WARN_EMOJI << YELLOW_COLOR << "No export file; outputting "
                  "classes to stdout" << RESET_COLOR << std::endl;
    }
    else {
        // Put results in the indicated file
        output_fstream.open(args.export_);
        output_buffer = output_fstream.rdbuf();
        std::cerr << INFO_EMOJI << "Outputting classes to " << args.export_
                                << std::endl;
    }
    // Used for actual file handling
    std::ostream out(output_buffer);
    std::string class_names[] = {"NULL", "min", "max", "regular", "saddle"};
    vtkIdType n_insane = 0;
    //for (vtkIdType i = 0; i < n_classes; i++) {
    for (vtkIdType i = FORCED_BLOCK_IDX; i < FORCED_BLOCK_IDX+1; i++) {
        // The classification information is provided, then the class:
        // {# upper, # lower, class}
        // CLASSES = {'minimum': 1, 'maximum': 2, 'regular': 3, 'saddle': 4}
        unsigned int n_upper  = classes[(i*3)],
                     n_lower  = classes[(i*3)+1],
                     my_class = classes[(i*3)+2];
        // Misclassification sanity checks
        if ((n_upper >= 1 && n_lower == 0 && my_class != MINIMUM_CLASS) ||
            (n_upper == 0 && n_lower >= 1 && my_class != MAXIMUM_CLASS) ||
            (n_upper == 1 && n_lower == 1 && my_class != REGULAR_CLASS) ||
            ((n_upper > 1 && n_lower > 1) && my_class != SADDLE_CLASS)) {
            out << "INSANITY DETECTED (" << n_upper << ", " << n_lower << ") FOR POINT " << i << std::endl;
            n_insane++;
        }
        /*
        out << "A Class " << i << " = " << my_class << std::endl;
        */
        out << "A Class " << i << " = " << class_names[my_class] << "(Upper: "
            << n_upper << ", Lower: " << n_lower << ")" << std::endl;
    }
    if (n_insane > 0) {
        std::cerr << WARN_EMOJI << RED_COLOR << "Insanity detected; "
                     "GPU did not agree on its own answers for " << n_insane
                  << " points." << RESET_COLOR << std::endl;
    }
}

int main(int argc, char *argv[]) {
    Timer timer(false, "Main");
    arguments args;
    parse(argc, argv, args);
    timer.tick();
    timer.interval("Argument parsing");

    // MANDATORY: TV (green) [from storage]
    std::cout << PUSHPIN_EMOJI << "Parsing vtu file: " << args.fileName
              << std::endl;
    timer.label_next_interval(GREEN_COLOR "TV" RESET_COLOR " from VTK");
    timer.tick();
    // Should utilize VTK API and then de-allocate all of its heap
    // Also loads the vertex attributes (host-side) and sets them in
    // TV->vertexAttributes (one scalar per vertex)
    std::unique_ptr<TV_Data> TV = get_TV_from_VTK(args); // args.filename
    timer.tick_announce();

    // Usually VE and VF are also mandatory, but CritPoints does not require
    // these relationships! Skip them!

    // OPTIONAL: VV (yellow) [TV' x TV]
    // REQUIRED for CritPoints
    std::cout << PUSHPIN_EMOJI << "Using CPU to compute " YELLOW_COLOR "VV" RESET_COLOR << std::endl;
    timer.label_next_interval(YELLOW_COLOR "VV" RESET_COLOR " [GPU]");
    timer.tick();
    // Have to make a max VV guess
    vtkIdType max_VV_guess = get_approx_max_VV(*TV, TV->nPoints);
    std::unique_ptr<VV_Data> VV = make_VV_GPU(*TV, TV->nCells, TV->nPoints, true, args);
    timer.tick_announce();

    // Critical Points
    timer.label_next_interval("Allocate " CYAN_COLOR "Critical Points" RESET_COLOR " memory");
    timer.tick();
    // #Upper, #Lower, Classification
    size_t classes_size = /* sizeof(unsigned int) **/ TV->nPoints * 3,
           // Upper/lower per adjacency
           valences_size = /* sizeof(vtkIdType) **/ TV->nPoints * max_VV_guess,
           scalars_size = /* sizeof(double) **/ TV->nPoints;
    // CPC = actual critical points classifications
    // valences = adjacency upper/lower classification PRIOR to point classification
    unsigned int *host_CPC = new unsigned int[classes_size];
    // Class values
    for (vtkIdType i = 0; i < TV->nPoints * 3; i++) {
        host_CPC[i] = 0;
    }
    vtkIdType *valences = new vtkIdType[valences_size],
              *vv_flat = new vtkIdType[valences_size];
    unsigned long long *vv_index_flat = new unsigned long long[scalars_size];
    for(vtkIdType i = 0; i < TV->nPoints; i++) {
        valences[i] = 0;
        vv_index_flat[i] = (*VV)[i].size();
        for (vtkIdType x = 0; x < vv_index_flat[i]; x++) {
            vv_flat[(i*max_VV_guess)+x] = (*VV)[i][x];
            std::cerr << "VV connects " << i << " to " << (*VV)[i][x] << std::endl;
        }
    }
    double    *scalar_values = new double[scalars_size];
    // Scalar values from VTK
    for(vtkIdType i = 0; i < TV->nPoints; i++) {
        scalar_values[i] = TV->vertexAttributes[i];
        //std::cerr << "TV value for point " << i << ": " << TV->vertexAttributes[i] << std::endl;
        std::cout << "A Scalar value for point " << i << ": " << scalar_values[i] << std::endl;
    }
    timer.tick_announce();
    timer.label_next_interval("Run " CYAN_COLOR "Critical Points" RESET_COLOR " algorithm");
    const vtkIdType n_to_compute = TV->nPoints * max_VV_guess;
    dim3 thread_block_size = max_VV_guess,
         grid_size = (n_to_compute + thread_block_size.x - 1) / thread_block_size.x;
    std::cerr << EXCLAIM_EMOJI << "DEBUG! Setting kernel size to 1 block "
              << "(as block " << FORCED_BLOCK_IDX << ")" << std::endl;
    timer.tick();
    critPointsCPU(vv_flat,
                  vv_index_flat,
                  valences,
                  TV->nPoints,
                  max_VV_guess,
                  scalar_values,
                  host_CPC);
    timer.tick_announce();
    timer.tick();
    export_classes(host_CPC, TV->nPoints, args);
    timer.tick_announce();
    if (host_CPC != nullptr) delete host_CPC;
    if (valences != nullptr) delete valences;
    if (scalar_values != nullptr) delete scalar_values;
    if (vv_flat != nullptr) delete vv_flat;
    if (vv_index_flat != nullptr) delete vv_index_flat;
    return 0;
}

