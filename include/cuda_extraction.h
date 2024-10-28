#ifndef TETRA_CUDA_RELATIONS
#define TETRA_CUDA_RELATIONS

#include "cuda_safety.h" // Safety and wrappers, includes iostream and cuda runtime
#include "cpu_extraction.h" // CPU-side datastructures and types
#include "datatypes.h" // Structs and types
#include "emoji.h" // emoji

// MAKE PRECOMPUTED DATA AVAILABLE ON GPU
void make_TV_for_GPU(vtkIdType * device_tv,
                     const TV_Data & tv_relationship);

void make_VE_for_GPU(vtkIdType * device_vertices,
                     vtkIdType * device_edges,
                     const VE_Data & ve_relationship,
                     const vtkIdType n_edges,
                     const vtkIdType n_verts);

// REQUEST RELATIONSHIPS FROM GPU

// vector of array of vertices in an edge
                                     // vector of vector of EdgeData
std::unique_ptr<EV_Data> make_EV_GPU(const VE_Data & edgeTable,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const arguments args);

// GPU KERNELS
__global__ void EV_kernel(vtkIdType * __restrict__ vertices,
                          vtkIdType * __restrict__ edges,
                          const vtkIdType n_edges,
                          vtkIdType * __restrict__ ev);

#endif

