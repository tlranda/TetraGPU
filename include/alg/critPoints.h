#ifndef TETRA_ALG_CRITPOINTS
#define TETRA_ALG_CRITPOINTS

#include <pthread.h> // POSIX CPU thread parallelism
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

__global__ void critPoints(const int * __restrict__ VV,
                           const unsigned int * __restrict__ VV_index,
                           int * __restrict__ valences,
                           const int points,
                           const int max_VV_guess,
                           const double * __restrict__ scalar_values,
                           const unsigned int * __restrict__ partition,
                           unsigned int * __restrict__ classes);

void export_classes(unsigned int * classes,
                    vtkIdType n_classes,
                    runtime_arguments & args);

#endif

