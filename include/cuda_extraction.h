#ifndef TETRA_CUDA_RELATIONS
#define TETRA_CUDA_RELATIONS

#include "cuda_safety.h" // Safety and wrappers, includes iostream and cuda runtime
#include "cpu_extraction.h" // CPU-side datastructures and types
#include "datatypes.h" // Structs and types
#include "emoji.h" // emoji
#include "metrics.h" // timer

// MAKE PRECOMPUTED DATA AVAILABLE ON GPU
void make_TV_for_GPU(const TV_Data & tv_relationship);

extern vtkIdType * device_VE_vertices, * device_VE_edges,
                 * device_VE_first_vertex;
void make_VE_for_GPU(const VE_Data & ve_relationship,
                     const vtkIdType n_edges,
                     const vtkIdType n_verts);

extern vtkIdType * device_VF_vertices, * device_VF_faces,
                 * device_VF_first_faces;
void make_VF_for_GPU(const VF_Data & vf_relationship,
                     const vtkIdType n_verts,
                     const vtkIdType n_faces);

// REQUEST RELATIONSHIPS FROM GPU
// EV = VE'
vtkIdType * make_EV_GPU_return(const VE_Data & edgeTable,
                               const vtkIdType n_points,
                               const vtkIdType n_edges,
                               const bool free_transients);
std::unique_ptr<EV_Data> make_EV_GPU(const VE_Data & edgeTable,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const bool free_transients);
__global__ void EV_kernel(const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ edges,
                          const vtkIdType n_edges,
                          vtkIdType * __restrict__ ev);
// TF = TV x VF
vtkIdType * make_TF_GPU_return(const TV_Data & TV,
                               const VF_Data & VF,
                               const vtkIdType n_points,
                               const vtkIdType n_faces,
                               const vtkIdType n_cells,
                               const bool free_transients);
std::unique_ptr<TF_Data> make_TF_GPU(const TV_Data & TV,
                                     const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const vtkIdType n_cells,
                                     const bool free_transients);
__global__ void TF_kernel(const int * __restrict__ tv,
                          const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ faces,
                          const vtkIdType * __restrict__ first_faces,
                          const vtkIdType n_cells,
                          const vtkIdType n_faces,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ tf);
// TE = TV x VE
vtkIdType * make_TE_GPU_return(const TV_Data & TV,
                               const VE_Data & VE,
                               const vtkIdType n_points,
                               const vtkIdType n_edges,
                               const vtkIdType n_cells,
                               const bool free_transients);
std::unique_ptr<TE_Data> make_TE_GPU(const TV_Data & TV,
                                     const VE_Data & VE,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const vtkIdType n_cells,
                                     const bool free_transients);
__device__ void te_combine(vtkIdType quad0, vtkIdType quad1,
                           vtkIdType quad2, vtkIdType quad3,
                           const vtkIdType laneID,
                           vtkIdType * __restrict__ te,
                           const vtkIdType * __restrict__ vertices,
                           const vtkIdType * __restrict__ edges,
                           const vtkIdType n_points,
                           const vtkIdType * __restrict__ index);
__global__ void TE_kernel(const int * __restrict__ tv,
                          const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ edges,
                          const vtkIdType * __restrict__ first_index,
                          const vtkIdType n_cells,
                          const vtkIdType n_edges,
                          const vtkIdType n_points,
                          vtkIdType * __restrict__ te);
// FV = VF'
vtkIdType * make_FV_GPU_return(const VF_Data & VF,
                               const vtkIdType n_points,
                               const vtkIdType n_faces,
                               const bool free_transients);
std::unique_ptr<FV_Data> make_FV_GPU(const VF_Data & VF,
                                     const vtkIdType n_points,
                                     const vtkIdType n_faces,
                                     const bool free_transients);
__global__ void FV_kernel(const vtkIdType * __restrict__ vertices,
                          const vtkIdType * __restrict__ faces,
                          const vtkIdType n_faces,
                          vtkIdType * __restrict__ fv);
// FE = VF' x VE
vtkIdType * make_FE_GPU_return(const VF_Data & VF,
                               const VE_Data & VE,
                               const vtkIdType n_points,
                               const vtkIdType n_edges,
                               const vtkIdType n_faces,
                               const bool free_transients);
std::unique_ptr<FE_Data> make_FE_GPU(const VF_Data & VF,
                                     const VE_Data & VE,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const vtkIdType n_faces,
                                     const bool free_transients);
// No kernel yet -- To be implemented
// VV = TV' x TV
int get_approx_max_VV(const TV_Data & TV, const vtkIdType n_points, const int debug);
device_VV * make_VV_GPU_return(const TV_Data & TV,
                               const int n_cells,
                               const int n_points,
                               const int max_VV_guess,
                               const bool free_transients);
std::unique_ptr<VV_Data> make_VV_GPU(const TV_Data & TV,
                                     const int n_cells,
                                     const int n_points,
                                     const bool free_transients);
__global__ void VV_kernel(const int * __restrict__ tv,
                          const int n_cells,
                          const int n_points,
                          const int offset,
                          unsigned int * __restrict__ index,
                          int * __restrict__ vv);

device_VT * make_VT_GPU_return(const TV_Data & TV);
device_VT * make_VT_GPU_return(const VT_Data & VT);

// -- NOT IMPLEMENTED BEYOND THIS LINE --

// ET = TE'
/*
std::unique_ptr<ET_Data> make_ET_GPU(const TE_Data & TE,
                                     const runtime_arguments args);
// Possible alternative: ET = (TV x VE)'
std::unique_ptr<ET_Data> make_ET_GPU(const TV_Data & TV,
                                     const VE_Data & VE,
                                     const vtkIdType n_points,
                                     const vtkIdType n_edges,
                                     const bool free_transients,
                                     const runtime_arguments args);
*/

#endif

