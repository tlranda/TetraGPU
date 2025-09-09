#ifndef TETRA_VTK_LOAD
#define TETRA_VTK_LOAD

// VTK library requirements
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkPointData.h>
// Arguments
#include "argparse.h"
#include "datatypes.h"
#include "emoji.h"
// STL
#include <set>

int checkCellTypes(vtkPointSet *object);
std::shared_ptr<TV_Data> get_TV_from_VTK(const runtime_arguments args);

#endif

