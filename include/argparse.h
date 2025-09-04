#ifndef TETRA_ARGPARSE
#define TETRA_ARGPARSE

#include <iostream> // std::{cerr,cout,endl}
#include <sstream> // std::stringstream
#include <string> // std::string
#include <map> // std::map
#include <getopt.h> // getopt_long()
#include <cstdlib>  // atoi
#include <array> // std::array
#include "cuda_safety.h" // Safety allows for Cuda runtime check for GPU count with error handling
#include "datatypes.h" // runtime_arguments struct
#include "emoji.h" // Emoji chars

std::string usage(const char* argv0,
                  const struct option * options,
                  const option_map & help_info,
                  const option_map & metavars);
void parse(int argc, char *argv[], runtime_arguments& args);

#endif

