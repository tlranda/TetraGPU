#include "argparse.h"

/* Handles logic for retrieving command line arguments; if you want to change
 * a default argument value, you should hop over to include/datatypes.h.in!
*/

std::string usage(char* argv0, const struct option* options,
                  const arguments & args, const option_map & help_info,
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

void parse(int argc, char *argv[], arguments& args) {
    /*
     * Command-line parsing
     *
     * Output as many helpful error-state messages as possible if invalid.
     * If valid, set values in the provided struct appropriately
     */
    int c, bad_args = 0;
    int arg_flags[7] = {0};
    // Disable getopt's automatic error messages so we can catch it via '?'
    opterr = 0;
    // Getopt option declarations
    const char * optionstring = "hi:t:";
    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"input", required_argument, 0, 'i'},
        {"threads", required_argument, 0, 't'},
        #ifdef VALIDATE_GPU
        {"validate", no_argument, &arg_flags[0], 1},
        #endif
        {"build_TE", no_argument, &arg_flags[1], 1},
        {"build_EV", no_argument, &arg_flags[2], 1},
        {"build_ET", no_argument, &arg_flags[3], 1},
        {"build_TF", no_argument, &arg_flags[4], 1},
        {"build_FV", no_argument, &arg_flags[5], 1},
        {"build_FE", no_argument, &arg_flags[6], 1},
        {0,0,0,0}
    };
    const option_map help_info = {
        {"help", "Print this help message and exit"},
        {"input", "Tetrahedral mesh input (.vtu only)"},
        {"threads", "CPU thread limit for parallelism"},
        #ifdef VALIDATE_GPU
        {"validate", "Check GPU results using CPU"},
        #endif
        {"build_TE", "Build the TE relationship"},
        {"build_EV", "Build the EV relationship"},
        {"build_ET", "Build the ET relationship"},
        {"build_TF", "Build the TF relationship"},
        {"build_FV", "Build the FV relationship"},
        {"build_FE", "Build the FE relationship"},
    };
    const option_map metavars = {
        {"input", "input.vtu"},
    };
    std::string help = usage(argv[0], long_options, args, help_info, metavars);
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
            case '?':
                errors << WARN_EMOJI << "Unrecognized argument: "
                       << argv[optind-1] << std::endl;
                bad_args += 1;
                break;
            case 'h':
                std::cout << help;
                exit(EXIT_SUCCESS);
        }
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
              << std::endl;

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

