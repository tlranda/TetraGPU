#ifndef TETRA_ARGPARSE
#define TETRA_ARGPARSE

#include <iostream> // std::{cerr,cout,endl}
#include <getopt.h> // getopt_long()
#include <cstdlib>  // atoi

// Argument values to be stored here, with defaults if they are optional
typedef struct argstruct {
    std::string fileName;
    int threadNumber = 1;
} arguments;

void parse(int argc, char *argv[], arguments& args);
#endif

