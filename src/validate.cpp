#include "validate.h"

const int MAX_TO_PRINT = 10;

bool check_host_vs_device_EV(const EV_Data & host_EV,
                             const EV_Data & device_EV) {
    // Validate that host and device agree on EV Data
    // The vectors do NOT need to be ordered the same, they just need to agree
    // On the number of edges and vertex contents of each edge
    if (host_EV.size() != device_EV.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device EV size (" << device_EV.size()
                  << ") != Host EV size (" << host_EV.size()
                  << ")" << std::endl;
        return false;
    }
    unsigned int idx = 0,
                 n_printed = 0,
                 n_found = 0,
                 n_failures = 0,
                 n_inverted = 0,
                 n_failures_before_early_exit = 0,
                 n_failures_to_print = MAX_TO_PRINT;
    for (const auto EV_Array : host_EV) {
        if (idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process edge " << idx << " ("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " % | " << n_inverted << " inverted edges found "
                      << 100 * n_inverted / static_cast<float>(idx)
                      << " %)" << std::endl;
        idx++;
        auto index = std::find(begin(device_EV), end(device_EV), EV_Array);
        if (index == std::end(device_EV)) {
            // Look for inverted edge
            std::array<vtkIdType,nbVertsInEdge> reversed_edge{EV_Array[1], EV_Array[0]};
            index = std::find(begin(device_EV), end(device_EV), reversed_edge);
            if (index == std::end(device_EV)){
                n_failures++;
                if (n_failures <= n_failures_to_print)
                    std::cerr << WARN_EMOJI << "Could not find edge ("
                              << EV_Array[0] << ", " << EV_Array[1]
                              << ") in device EV!" << std::endl;
                if (n_failures_before_early_exit > 0 &&
                    n_failures >= n_failures_before_early_exit) return false;
            }
            else {
                n_found++;
                n_inverted++;
                if (n_printed < 10) {
                    std::cout << WARN_EMOJI
                              << "Matched INVERTED edge between host and device ("
                              << reversed_edge[0] << ", " << reversed_edge[1] << ")"
                              << std::endl;
                    n_printed++;
                }
            }
        }
        else {
            n_found++;
            if (n_printed < 10) {
                std::cout << OK_EMOJI << "Matched edge between host and device ("
                          << EV_Array[0] << ", " << EV_Array[1] << ")"
                          << std::endl;
                n_printed++;
            }
        }
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " edges" << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " edges" << std::endl;
    return n_failures == 0;
}

bool check_host_vs_device_TF(TF_Data & host, TF_Data & device) {
    if (host.size() != device.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device TF size (" << device.size()
                  << ") != Host TF size (" << host.size()
                  << ")" << std::endl;
        return false;
    }
    /* The faces can technically come in any order and be valid, however that
     * can make the search to validate take a long time. It ALSO can mess up
     * helpful tools like std::find and std::next_permutation if the back-end
     * implementation attempts to take shortcuts based on comparables like our
     * basic datatypes (integers of some size). THEREFORE, you need to sort
     * each array in both the host and the device vectors PRIOR to searching
     * for matches. Then comparisons via std::find() are able to guarantee
     * matches are located (or not, when absent) in one shot. */
    std::for_each(host.begin(), host.end(),
            [](std::array<vtkIdType,nbFacesInCell>& arr) {
                std::sort(arr.begin(), arr.end());
            });
    std::for_each(device.begin(), device.end(),
            [](std::array<vtkIdType,nbFacesInCell>& arr) {
                std::sort(arr.begin(), arr.end());
            });


    long long int idx = 0,
                  n_printed = 0,
                  n_found = 0,
                  n_failures = 0,
                  n_failures_before_early_exit = 0,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto FaceArray : host) {
        if (idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process cell " << idx << " ("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        idx++;
        auto index = std::find(begin(device), end(device), FaceArray);
        if (index == std::end(device)) {
            n_failures++;
            if (n_failures <= n_failures_to_print) {
                std::cerr << WARN_EMOJI << "Could not find set of faces ("
                          << FaceArray[0] << ", " << FaceArray[1] << ", "
                          << FaceArray[2] << ", " << FaceArray[3]
                          << ") in device TF!" << std::endl;
                std::cerr << WARN_EMOJI << "Expected match at index " << idx-1
                          << " between host (" << FaceArray[0] << ", "
                          << FaceArray[1] << ", " << FaceArray[2] << ", "
                          << FaceArray[3] << ") and device ("
                          << device[idx-1][0] << ", " << device[idx-1][1]
                          << ", " << device[idx-1][2] << ", "
                          << device[idx-1][3] << ")" << std::endl;
                n_printed++;
            }
            if (n_failures_before_early_exit > 0 &&
                n_failures >= n_failures_before_early_exit) return false;
        }
        else {
            n_found++;
            if (n_printed < MAX_TO_PRINT) {
                std::cout << OK_EMOJI << "Matched face between host " << idx-1
                          << " and device " << std::distance(device.begin(), index)
                          << " (" << FaceArray[0] << ", " << FaceArray[1]
                          << ", " << FaceArray[2] << ", " << FaceArray[3]
                          << ")" << std::endl;
                n_printed++;
            }
            /*
            else if (idx-1 != std::distance(device.begin(), index)) {
                std::cerr << WARN_EMOJI << "Host and device indices MATCHED, "
                          << "but in an unexpected location (host: " << idx-1
                          << ", device: "
                          << std::distance(device.begin(), index) << ")"
                          << std::endl;
            }
            */
        }
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " faces" << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " faces" << std::endl;
    return n_failures == 0;
}

bool check_host_vs_device_TE(const TE_Data & host, const TE_Data & device) {
    if (host.size() != device.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device TE size (" << device.size()
                  << ") != Host TE size (" << host.size()
                  << ")" << std::endl;
        return false;
    }
    long long int idx = 0,
                  n_printed = 0,
                  n_found = 0,
                  n_failures = 0,
                  n_failures_before_early_exit = 0,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto EdgeArray : host) {
        if (idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process cell " << idx << " ("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        idx++;
        auto index = std::find(begin(device), end(device), EdgeArray);
        if (index == std::end(device)) {
            n_failures++;
            if (n_failures <= n_failures_to_print) {
                std::cerr << WARN_EMOJI << "Could not find set of edges ("
                    << EdgeArray[0] << ", " << EdgeArray[1] << ", "
                    << EdgeArray[2] << ", " << EdgeArray[3] << ") in device TE!"
                    << std::endl;
                std::cerr << WARN_EMOJI << "Expected match at index " << idx-1
                    << " between host (" << EdgeArray[0] << ", "
                    << EdgeArray[1] << ", " << EdgeArray[2] << ","
                    << EdgeArray[3] << ") and device (" << device[idx-1][0]
                    << ", " << device[idx-1][1] << ", " << device[idx-1][2]
                    << ", " << device[idx-1][3] << ")" << std::endl;
                n_printed++;
            }
            if (n_failures_before_early_exit > 0 &&
                    n_failures >= n_failures_before_early_exit) return false;
        }
        else {
            n_found++;
            if (n_printed < 10) {
                std::cout << OK_EMOJI << "Matched edge set between host and device ("
                    << EdgeArray[0] << ", " << EdgeArray[1] << ", "
                    << EdgeArray[2] << ", " << EdgeArray[3] << ")"
                    << std::endl;
                n_printed++;
            }
        }
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " cells" << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " cells" << std::endl;
    return n_failures == 0;
}

