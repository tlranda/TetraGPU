#ifndef TETRA_CUDA_RELATIONS
#define TETRA_CUDA_RELATIONS

#include "cuda_safety.h" // Safety and wrappers, includes iostream and cuda runtime
#include "cpu_extraction.h" // CPU-side datastructures and types

// vector of array of vertices in an edge
                                     // vector of array of vertices in a tetra
std::unique_ptr<EV_Data> make_EV_GPU(const TV_Data & tv_relationship,
                                     // vector of vector of EdgeData
                                     const VE_Data & edgeTable,
                                     const vtkIdType n_edges,
                                     const arguments args);

void make_TV_for_GPU(vtkIdType * device_tv,
                     const TV_Data & tv_relationship);

void make_VE_for_GPU(vtkIdType * device_ve,
                     vtkIdType * device_offset,
                     const VE_Data & ve_relationship,
                     const vtkIdType n_edges,
                     const vtkIdType n_verts);

#endif
