#include "argparse.h"

/* Handles logic for retrieving command line arguments; if you want to change
 * a default argument value, you should hop over to include/datatypes.h.in!
*/

std::string usage(char* argv0,
                  const struct option* options,
                  const option_map & help_info,
                  const option_map & metavars) {
    // Stringify usage of arguments
    std::stringstream ss;
    ss << "Usage: " << argv0 << " [options]" << std::endl << "Options:"
       << std::endl;
    for (int i = 0; options[i].name != nullptr; ++i) {
        // Flag names
        if (options[i].val != 1)
            ss << "-" << static_cast<char>(options[i].val) << " | ";
        ss << "--" << options[i].name;
        // metavar / arguments
        auto it = metavars.find(options[i].name);
        switch (options[i].has_arg) {
            case no_argument:
                ss << std::endl;
                break;
            case required_argument:
                ss << " <";
                // Metavar lookup
                if (it != metavars.end()) ss << it->second;
                else ss << options[i].name;
                ss << ">" << std::endl;
                break;
            case optional_argument:
                ss << " [";
                // Metavar lookup
                if (it != metavars.end()) ss << it->second;
                else ss << options[i].name;
                ss << "]" << std::endl;
                break;
            default:
                break;
        }
        // help information
        it = help_info.find(options[i].name);
        if (it != metavars.end()) {
            ss << "\t\t" << it->second << std::endl;
        }
    }
    return ss.str();
}

void parse(int argc, char *argv[], runtime_arguments & args) {
    /*
     * Command-line parsing
     *
     * Output as many helpful error-state messages as possible if invalid.
     * If valid, set values in the provided struct appropriately
     */
    int c, bad_args = 0;
    int arg_flags[14] = {0};
    // Disable getopt's automatic error messages so we can catch it via '?'
    opterr = 0;
    // Getopt option declarations
    const char * optionstring = "hi:t:e:a:p:v:g:"
    ;
    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"input", required_argument, 0, 'i'},
        {"threads", required_argument, 0, 't'},
        {"export", required_argument, 0, 'e'},
        {"arrayname", required_argument, 0, 'a'},
        {"partitioningname", required_argument, 0, 'p'},
        {"max_VV", required_argument, 0, 'v'},
        {"gpus", required_argument, 0, 'g'},
        #ifdef VALIDATE_GPU
        {"validate", no_argument, &arg_flags[0], 1},
        #endif
        {"build_TE", no_argument, &arg_flags[1], 1},
        {"build_EV", no_argument, &arg_flags[2], 1},
        {"build_ET", no_argument, &arg_flags[3], 1},
        {"build_TF", no_argument, &arg_flags[4], 1},
        {"build_FV", no_argument, &arg_flags[5], 1},
        {"build_FE", no_argument, &arg_flags[6], 1},
        {"build_FT", no_argument, &arg_flags[7], 1},
        {"build_EF", no_argument, &arg_flags[8], 1},
        {"build_VT", no_argument, &arg_flags[9], 1},
        {"build_TT", no_argument, &arg_flags[10], 1},
        {"build_FF", no_argument, &arg_flags[11], 1},
        {"build_EE", no_argument, &arg_flags[12], 1},
        {"build_VV", no_argument, &arg_flags[13], 1},
        {0,0,0,0}
    };
    const option_map help_info = {
        {"help", "Print this help message and exit"},
        {"input", "Tetrahedral mesh input (.vtu only)"},
        {"threads", "CPU thread limit for parallelism"},
        {"export", "File to export CritPoints classifications to"},
        {"arrayname", "Array to use for scalar data (as string name)"},
        {"partitioningname", "Array to use for multi-GPU partitioning (as string name)"},
        {"max_VV", "Override estimation of max VV with integer value"},
        {"gpus", "Set number of GPUs to use (larger than detected is warning, but will emulate behavior)"},
        #ifdef VALIDATE_GPU
        {"validate", "Check GPU results using CPU"},
        #endif
        {"build_TE", "Build the TE relationship"},
        {"build_EV", "Build the EV relationship"},
        {"build_ET", "Build the ET relationship"},
        {"build_TF", "Build the TF relationship"},
        {"build_FV", "Build the FV relationship"},
        {"build_FE", "Build the FE relationship"},
        {"build_FT", "Build the FT relationship"},
        {"build_EF", "Build the EF relationship"},
        {"build_VT", "Build the VT relationship"},
        {"build_TT", "Build the TT relationship"},
        {"build_FF", "Build the FF relationship"},
        {"build_EE", "Build the EE relationship"},
        {"build_VV", "Build the VV relationship"},
    };
    const option_map metavars = {
        {"input", "input.vtu"},
        {"export", "classes.txt"},
        {"arrayname", "my_scalar_data_name"},
        {"partitioningname", "my_partition_data_name"},
        {"max_VV", "(INT>0, preferably multiple of 32)"},
        {"gpus", "(INT>=0)"}
    };
    std::stringstream errors;

    // Begin parsing
    while (1) {
        int option_index = 0;
        c = getopt_long(argc, argv, optionstring, long_options, &option_index);
        if (c == -1) break;
        switch (c) {
            case 0:
                // I need to remind myself of the actual significance of this case
                break;
            case 'i':
                args.fileName = std::string(optarg);
                break;
            case 't':
                args.threadNumber = atoi(optarg);
                if (args.threadNumber == 0) {
                    // Indicates 0 or an error in processing, fortunately
                    // 0 is an invalid value for us as well in this context.
                    std::cerr << "Thread argument must be integer >= 1" <<
                                 std::endl;
                    bad_args += 1;
                }
                break;
            case 'e':
                args.export_ = std::string(optarg);
                break;
            case '?':
                errors << WARN_EMOJI << "Unrecognized argument: "
                       << argv[optind-1] << std::endl;
                bad_args += 1;
                break;
            case 'a':
                args.arrayname = std::string(optarg);
                break;
            case 'p':
                args.partitioningname = std::string(optarg);
                break;
            case 'v':
                args.max_VV = atoi(optarg);
                if (args.max_VV <= 0) {
                    std::cerr << "Max VV must be >= 1" << std::endl;
                    bad_args += 1;
                }
                else if (args.max_VV % 32 != 0) {
                    std::cerr << WARN_EMOJI << "Max VV is preferred to be a "
                                               "multiple of 32!" << std::endl;
                }
                break;
            case 'g':
                args.n_GPUS = atoi(optarg);
                break;
            case 'h':
                std::string help = usage(argv[0],
                                         long_options,
                                         help_info,
                                         metavars);
                std::cout << help;
                exit(EXIT_SUCCESS);
        }
    }
    // Final parsing
    if (args.n_GPUS == 0) {
        // Ensure at least one GPU exists, else error
        int check_gpus;
        CUDA_WARN(cudaGetDeviceCount(&check_gpus));
        if (check_gpus == 0) {
            errors << EXCLAIM_EMOJI
                   << "No GPUs detected on the system"
                   << std::endl;
            bad_args += 1;
        }
        else {
            std::cerr << INFO_EMOJI << "Auto-detect " << check_gpus
                      << " gpus on the system." << std::endl;
        }
        // Assign to use all GPUs
        args.n_GPUS = check_gpus;
    }
    else {
        // Warn about behaviors if more GPUs requested than detected
        int check_gpus;
        CUDA_WARN(cudaGetDeviceCount(&check_gpus));
        if (check_gpus < args.n_GPUS) {
            std::cerr << WARN_EMOJI
                      << "Fewer GPUs detected on the system (" << check_gpus
                      << ") than requested (" << args.n_GPUS << ")"
                      << std::endl;
        }
        else {
            std::cerr << OK_EMOJI
                      << "Found all " << args.n_GPUS << " / " << check_gpus
                      << " on the system. Proceed."
                      << std::endl;
        }
    }
    // Truncate to single-GPU execution with warning if no partitioning provided!
    if (args.partitioningname == "" && args.n_GPUS > 1) {
        std::cerr << WARN_EMOJI
                  << "No partitioning name given (-p/--partitioningname), only ONE GPU can be used!"
                  << std::endl;
        args.n_GPUS = 1;
    }
    // Display parsed values
    if (args.fileName.empty()) {
        errors << EXCLAIM_EMOJI
               << "Must supply an input filename via -i | --input"
               << std::endl;
        bad_args += 1;
    }
    else {
        std::cout << INFO_EMOJI << "Dataset: " << args.fileName << std::endl;
    }
    std::cout << INFO_EMOJI << "CPU threads: " << args.threadNumber
              << std::endl
              << INFO_EMOJI << "Export: " << (args.export_ == "" ?
                                              "[n/a]" :
                                              args.export_) << std::endl
              << INFO_EMOJI << "Array name: " << (args.arrayname == "" ?
                                                "[default to first array]" :
                                                args.arrayname) << std::endl
              << INFO_EMOJI << "Partitioning name: " << (args.partitioningname == "" ?
                                                "NO PARTITIONING -- SINGLE GPU" :
                                                args.partitioningname) << std::endl
              << INFO_EMOJI << "Max VV: " << (args.max_VV == -1 ?
                                                "[Estimated from mesh]" :
                                                std::to_string(args.max_VV))
                            << std::endl
              << INFO_EMOJI << "GPUs: " << args.n_GPUS << std::endl;
    // Set bit flags
    c = 0;
    for (int bit_value : arg_flags) {
        #ifndef VALIDATE_GPU
        if (c == 0) {
            c++;
            continue;
        }
        #endif
        if (!args.arg_flags[c]) args.arg_flags[c] = bit_value != 0;
        std::cout << INFO_EMOJI << "Bit flag [" << c << ": "
                  << args.flag_names[c] << "]: "
                  << (args.arg_flags[c] ? "true" : "false") << std::endl;
        c++;
    }

    if (bad_args != 0) {
        std::cerr << errors.str();
        exit(EXIT_FAILURE);
    }
}

