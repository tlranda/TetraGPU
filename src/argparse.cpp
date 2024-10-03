#include "argparse.h"

void parse(int argc, char *argv[], arguments& args) {
    /*
     * Command-line parsing
     *
     * Output as many helpful error-state messages as possible if invalid.
     * If valid, set values in the provided struct appropriately
     */
    int c, bad_args = 0;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " -i <input.vtu> [options]" <<
                     std::endl << "Missing required input .vtu file!" <<
                     std::endl;
        bad_args += 1;
    }
    // Disable getopt's automatic error messages so we can catch it via '?'
    opterr = 0;
    // Getopt option declarations
    const char * optionstring = "hi:t:";
    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"input", required_argument, 0, 'i'},
        {"threads", required_argument, 0, 't'},
        {0,0,0,0}
    };

    while (1) {
        int option_index = 0;
        c = getopt_long(argc, argv, optionstring, long_options, &option_index);
        if (c == -1) break;
        switch (c) {
            case 0:
                // I need to remind myself of the actual significance of this case
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " -i <input.vtu> " <<
                             "[options]" << std::endl;
                std::cout << "\t-h | --help\n\t\t" <<
                             "Print this help message and exit" << std::endl;
                std::cout << "\t-i <input.vtu> | --input <input.vtu>\n\t\t" <<
                             "Tetrahedral mesh input (.vtu only)" << std::endl;
                std::cout << "\t-t <threads> | --threads <threads>\n\t\t" <<
                             "CPU thread limit for parallelism" << std::endl;
                exit(EXIT_SUCCESS);
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
                std::cerr << "Unrecognized argument: " << argv[optind-1] <<
                             std::endl;
                bad_args += 1;
                break;
        }
    }

    // Filename must be given
    if (args.fileName.empty()) {
        std::cerr << "Must supply an input filename via -i | --input" <<
                     std::endl;
        bad_args += 1;
    }

    if (bad_args != 0) exit(EXIT_FAILURE);
}

