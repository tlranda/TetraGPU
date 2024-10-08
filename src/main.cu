// Other files in this repository
#include "argparse.h" // arguments type and parse()
#include "vtk_load.h" // TV_Data type and get_TV_from_VTK()
#include "cpu_extraction.h" // *_Data types and make_*() / elective_make_*()
#include "cuda_extraction.h"

__global__ void kernel(void) {
    int tid = threadIdx.x;
    printf("%d\n", tid);
}

int main(int argc, char *argv[]) {
    arguments args;
    parse(argc, argv, args);

    std::cout << "Parsing vtu file: " << args.fileName << std::endl;
    // Should utilize VTK API and then de-allocate all of its heap
    std::unique_ptr<TV_Data> tv_relationship = get_TV_from_VTK(args);

    // Adapted from TTK Explicit Triangulation
    std::cout << "Building edges..." << std::endl;
    std::unique_ptr<TE_Data> cellEdgeList = std::make_unique<TE_Data>(tv_relationship->nCells);
    std::unique_ptr<VE_Data> edgeTable = std::make_unique<VE_Data>(tv_relationship->nPoints);
    // The TE relationship simultaneously informs VE, so make both at once
    vtkIdType edgeCount = make_TE_and_VE(*tv_relationship,
                                         *(cellEdgeList.get()),
                                         *(edgeTable.get()));
    std::cout << "Built " << edgeCount << " edges." << std::endl;

    // allocate & fill edgeList in parallel (EV)
    std::unique_ptr<EV_Data> EV = elective_make_EV(*tv_relationship,
                                                   *(edgeTable.get()),
                                                   edgeCount,
                                                   args);

    // we can also get edgeStars from cellEdgeList (ET)
    std::unique_ptr<ET_Data> ET = elective_make_ET(*(cellEdgeList.get()),
                                                   edgeCount,
                                                   args);

    // Make faces, which we define based on cells and vertices so we simultaneously define TF and FV
    std::cout << "Building faces..." << std::endl;
    std::unique_ptr<TF_Data> cellFaceList = std::make_unique<TF_Data>(tv_relationship->nCells);
    std::unique_ptr<VF_Data> faceTable = std::make_unique<VF_Data>();
    faceTable.get()->resize(tv_relationship->nPoints); // guarantee space AND make indexing valid
    vtkIdType faceCount = make_TF_and_VF(*tv_relationship,
                                         *(cellFaceList.get()),
                                         *(faceTable.get()));
    std::cout << "Built " << faceCount << " faces." << std::endl;

    kernel<<<10,120993240>>>();
    DUMMY_WARN
    //KERNEL_WARN();
    //CUDA_WARN(
    cudaDeviceSynchronize();
    //);
    return 0;
}

