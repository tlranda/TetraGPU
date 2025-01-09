#ifndef TETRA_CPU_RELATIONS
#define TETRA_CPU_RELATIONS

#include <algorithm> // std::sort
#include "datatypes.h" // Structures etc

vtkIdType make_TE_and_VE(const TV_Data & tv_relationship,
                         TE_Data & cellEdgeList,
                         VE_Data & edgeTable);
vtkIdType make_VE(const TV_Data & tv_relationship,
                  VE_Data & edgeTable);
std::unique_ptr<EV_Data> elective_make_EV(const VE_Data & edgeTable,
                                          const vtkIdType n_points,
                                          const vtkIdType n_edges,
                                          const arguments args);
std::unique_ptr<ET_Data> elective_make_ET(const TE_Data & cellEdgeList,
                                          const vtkIdType n_edges,
                                          const arguments args);
std::unique_ptr<ET_Data> elective_make_ET(const TE_Data & cellEdgeList,
                                          const vtkIdType n_edges,
                                          const arguments args);
std::unique_ptr<FV_Data> elective_make_FV(const VF_Data & VF,
                                          const vtkIdType n_faces,
                                          const arguments args);
std::unique_ptr<FE_Data> elective_make_FE(const VF_Data & VF,
                                          const VE_Data & VE,
                                          const vtkIdType n_points,
                                          const vtkIdType n_edges,
                                          const vtkIdType n_faces,
                                          const arguments args);
vtkIdType make_TF_and_VF(const TV_Data & tv_relationship,
                         TF_Data & cellFaceList,
                         VF_Data & faceTable);
vtkIdType make_VF(const TV_Data & tv_relationship,
                  VF_Data & faceTable);
#endif

