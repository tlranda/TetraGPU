#ifndef TETRA_VTK_LOAD
#define TETRA_VTK_LOAD
// VTK library requirements
#include <vtkCellTypes.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
// Arguments
#include "argparse.h"

const int nbVertsInCell = 4; // verifiable at VTK load via VTK offsets[1]-offsets[0]

struct TV_Data {
    vtkIdType nPoints, nCells;
    // For some reason, vector of array is experimentally 20x faster than unique_ptr?
    //std::unique_ptr<vtkIdType[]> cells;
    std::vector<std::array<vtkIdType,nbVertsInCell>> cells;

    TV_Data(vtkIdType _points, vtkIdType _cells) {
        nPoints = _points;
        nCells = _cells;
        //cells = std::make_unique<vtkIdType[]>(_points * _cells);
        cells.resize(nCells);
    }

    std::vector<std::array<vtkIdType,nbVertsInCell>>::iterator begin() {
        return cells.begin();
    }
    std::vector<std::array<vtkIdType,nbVertsInCell>>::const_iterator begin() const {
        return cells.begin();
    }
    std::vector<std::array<vtkIdType,nbVertsInCell>>::iterator end() {
        return cells.end();
    }
    std::vector<std::array<vtkIdType,nbVertsInCell>>::const_iterator end() const {
        return cells.end();
    }
};

int checkCellTypes(vtkPointSet *object);
std::unique_ptr<TV_Data> get_TV_from_VTK(const arguments args);

#endif

