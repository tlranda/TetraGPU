#ifndef TETRA_ARGPARSE
#define TETRA_ARGPARSE

#include <iostream> // std::{cerr,cout,endl}
#include <getopt.h> // getopt_long()
#include <cstdlib>  // atoi
#include "datatypes.h" // arguments struct

void parse(int argc, char *argv[], arguments& args);
#endif

