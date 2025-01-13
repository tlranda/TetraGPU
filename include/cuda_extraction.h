#ifndef TETRA_CUDA_RELATIONS
#define TETRA_CUDA_RELATIONS

#include "cuda_safety.h" // Safety and wrappers, includes iostream and cuda runtime
#include "cpu_extraction.h" // CPU-side datastructures and types
#include "datatypes.h" // Structs and types
#include "emoji.h" // emoji
#include "metrics.h" // timer

// MAKE PRECOMPUTED DATA AVAILABLE ON GPU
void make_TV_for_GPU(vtkIdType ** device_tv,
                     const TV_Data & tv_relationship);
void make_VE_for_GPU(vtkIdType ** device_vertices,
                     vtkIdType ** device_edges,
                     vtkIdType ** device_first_vertex,
                     const VE_Data & ve_relationship,
                     const vtkIdType n_edges,
                     const vtkIdType n_verts);
void make_VF_for_GPU(vtkIdType ** device_vertices,
                     vtkIdType ** device_faces,
                     vtkIdType ** device_first_faces,
                     const VF_Data & vf_relationship,
                     const vtkIdType n_verts,
                     const vtkIdType n_faces);

// REQUEST RELATIONSHIPS FROM GPU
// EV = VE'
std::unique_ptr<EV_Data> make_EV_GPU(const VE_Data & edgeTable,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const arguments args);
// TE = TV x VE
std::unique_ptr<TE_Data> make_TE_GPU(const TV_Data & TV,
                                     const VE_Data & VE,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const vtkIdType n_cells,
                                     const arguments args);
// ET = TE'
std::unique_ptr<ET_Data> make_ET_GPU(const TE_Data & TE,
                                     const arguments args);
// Possible alternative: ET = (TV x VE)'
 std::unique_ptr<ET_Data> make_ET_GPU(const TV_Data & TV,
                                      const VE_Data & VE,
                                      const vtkIdType n_points,
                                      const vtkIdType n_edges,
                                      const arguments args);
// TF = TV x VF
std::unique_ptr<TF_Data> make_TF_GPU(const TV_Data & TV,
                                     const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const vtkIdType n_cells,
                                     const arguments args);
// FV = VF'
std::unique_ptr<FV_Data> make_FV_GPU(const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const arguments args);
// FE = VF' x VE
std::unique_ptr<FE_Data> make_FE_GPU(const VF_Data & VF,
                                     const VE_Data & VE,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const vtkIdType n_faces,
                                     const arguments args);
// VV = TV' x TV
vtkIdType get_approx_max_VV(const TV_Data & TV, const vtkIdType n_points);
std::unique_ptr<VV_Data> make_VV_GPU(const TV_Data & TV,
                                     const vtkIdType n_cells,
                                     const vtkIdType n_points,
                                     const arguments args);
// GPU KERNELS
__global__ void EV_kernel(const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ edges,
                          const vtkIdType n_edges,
                          vtkIdType * __restrict__ ev);
__global__ void TF_kernel(const vtkIdType * __restrict__ tv,
                          const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ faces,
                          const vtkIdType * __restrict__ first_faces,
                          const vtkIdType n_cells,
                          const vtkIdType n_faces,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ tf);
__device__ void te_combine(vtkIdType quad0, vtkIdType quad1,
                           vtkIdType quad2, vtkIdType quad3,
                           const vtkIdType laneID,
                           vtkIdType * __restrict__ te,
                           const vtkIdType * __restrict__ vertices,
                           const vtkIdType * __restrict__ edges,
                           const vtkIdType n_points,
                           const vtkIdType * __restrict__ index);
__global__ void TE_kernel(const vtkIdType * __restrict__ tv,
                          const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ edges,
                          const vtkIdType * __restrict__ first_index,
                          const vtkIdType n_cells,
                          const vtkIdType n_edges,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ te);
__global__ void FV_kernel(const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ faces,
                          const vtkIdType n_faces,
                          vtkIdType * __restrict__ fv);
__global__ void VV_kernel(const vtkIdType * __restrict__ tv,
                          const vtkIdType n_cells,
                          const vtkIdType n_points,
                          const vtkIdType offset,
                          unsigned long long int * __restrict__ index,
                          vtkIdType * __restrict__ vv);

#endif

