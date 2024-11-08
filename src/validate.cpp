#include "validate.h"

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
                 n_failures_to_print = 10;
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

bool check_host_vs_device_TF(const TF_Data & host, const TF_Data & device) {
    if (host.size() != device.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device TF size (" << device.size()
                  << ") != Host TF size (" << host.size()
                  << ")" << std::endl;
        return false;
    }
    unsigned int idx = 0,
                 n_printed = 0,
                 n_found = 0,
                 n_failures = 0,
                 n_inverted = 0,
                 n_failures_before_early_exit = 0,
                 n_failures_to_print = 10;
    std::cerr << WARN_EMOJI << "This function is not fully implemented! "
              << "It should properly validate a 100\% conforming GPU output, "
              << "but cannot check for near-misses" << std::endl;
    for (const auto FaceArray : host) {
        if (idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process cell " << idx << " ("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " % | " << n_inverted << " mis-ordered faces found "
                      << 100 * n_inverted / static_cast<float>(idx)
                      << " %)" << std::endl;
        idx++;
        auto index = std::find(begin(device), end(device), FaceArray);
        if (index == std::end(device)) {
            // Look for mis-ordered face
            // auto reordered_face = ; // Swap 0,1,2 orders
            //index = std::find(begin(device), end(device), reordered_face);
            if (index == std::end(device)){
                n_failures++;
                if (n_failures <= n_failures_to_print)
                    std::cerr << WARN_EMOJI << "Could not find face ("
                              << FaceArray[0] << ", " << FaceArray[1] << ", "
                              << FaceArray[2] << ") in device EV!"
                              << std::endl;
                if (n_failures_before_early_exit > 0 &&
                    n_failures >= n_failures_before_early_exit) return false;
            }
            else {
                n_found++;
                n_inverted++;
                if (n_printed < 10) {
                    std::cout << WARN_EMOJI
                              << "Matched INVERTED face between host and device ("
                              //<< reordered_face[0] << ", " << reordered_face[1]
                              //<< ", " << reordered_face[2]
                              << ")" << std::endl;
                    n_printed++;
                }
            }
        }
        else {
            n_found++;
            if (n_printed < 10) {
                std::cout << OK_EMOJI << "Matched face between host and device ("
                          << FaceArray[0] << ", " << FaceArray[1] << ", "
                          << FaceArray[2] << ")" << std::endl;
                n_printed++;
            }
        }
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " faces" << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " faces" << std::endl;
    return n_failures == 0;
}
