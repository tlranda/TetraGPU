#ifndef TETRA_CUDA_RELATIONS
#define TETRA_CUDA_RELATIONS

#include "cuda_safety.h" // Safety and wrappers, includes iostream and cuda runtime
#include "cpu_extraction.h" // CPU-side datastructures and types
#include "datatypes.h" // Structs and types
#include "emoji.h" // emoji
#include "metrics.h" // timer

// MAKE PRECOMPUTED DATA AVAILABLE ON GPU
void make_TV_for_GPU(vtkIdType * device_tv,
                     const TV_Data & tv_relationship);
void make_VE_for_GPU(vtkIdType * device_vertices,
                     vtkIdType * device_edges,
                     const VE_Data & ve_relationship,
                     const vtkIdType n_edges,
                     const vtkIdType n_verts);

// REQUEST RELATIONSHIPS FROM GPU
// EV = VE'
std::unique_ptr<EV_Data> make_EV_GPU(const VE_Data & edgeTable,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const arguments args);
// TE = TV x VE
std::unique_ptr<TE_Data> make_TE_GPU(const TE_Data & TE,
                                     const VE_Data & VE,
                                     const vtkIdType n_edges,
                                     const arguments args);
// ET = TE'
std::unique_ptr<ET_Data> make_ET_GPU(const TE_Data & TE,
                                     const arguments args);
// Possible alternative: ET = (TV x VE)'
// std::unique_ptr<ET_Data> make_ET_GPU(const TV_Data & TV,
//                                      const VE_Data & VE,
//                                      const vtkIdType n_points,
//                                      const vtkIdType n_edges,
//                                      const arguments args);
// TF = TV x VF
std::unique_ptr<TF_Data> make_TF_GPU(const TV_Data & TV,
                                     const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const arguments args);
// FV = VF'
std::unique_ptr<FV_Data> make_FV_GPU(const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const arguments args);
// FE = VF' x VE
std::unique_ptr<FV_Data> make_FV_GPU(const VF_Data & VF,
                                     const VE_Data & VE,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const arguments args);

// GPU KERNELS
__global__ void EV_kernel(vtkIdType * __restrict__ vertices,
                          vtkIdType * __restrict__ edges,
                          const vtkIdType n_edges,
                          vtkIdType * __restrict__ ev);

#endif

