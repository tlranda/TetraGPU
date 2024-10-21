#include "validate.h"

bool check_host_vs_device_EV(const EV_Data host_EV,
                             const EV_Data device_EV) {
    // Validate that host and device agree on EV Data
    // The vectors do NOT need to be ordered the same, they just need to agree
    // On the number of edges and vertex contents of each edge
    if (host_EV.size() != device_EV.size()) {
        std::cerr << "Device EV size (" << device_EV.size()
                  << ") != Host EV size (" << host_EV.size()
                  << ")" << std::endl;
        return false;
    }
    unsigned int idx = 0,
                 n_printed = 0,
                 n_failures = 0,
                 n_failures_before_early_exit = 0,
                 n_failures_to_print = 10;
    for (const auto EV_Array : host_EV) {
        if (idx % 1000 == 0)
            std::cerr << "Process edge " << idx << std::endl;
        idx++;
        auto index = std::find(begin(device_EV), end(device_EV), EV_Array);
        if (index == std::end(device_EV)) {
            n_failures++;
            if (n_failures <= n_failures_to_print)
                std::cerr << "Could not find edge (" << EV_Array[0] << ", "
                          << EV_Array[1] << ") in device EV!" << std::endl;
            if (n_failures_before_early_exit > 0 &&
                n_failures >= n_failures_before_early_exit) return false;
        }
        else {
            if (n_printed < 10) {
                std::cout << "Matched edge between host and device ("
                          << EV_Array[0] << ", " << EV_Array[1] << ")"
                          << std::endl;
                n_printed++;
            }
        }
    }
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << "Failed to match " << n_failures << " edges" << std::endl;
    return n_failures == 0;
}


