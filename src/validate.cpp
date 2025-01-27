#include "validate.h"

/* Validate GPU ("device") relationships via the provided CPU ("host") version

   Most data is permutable -- ie it doesn't matter what order vertices appear
   in an edge, as long as all vertices are present and correct.
   You can sort within every entry to make this easy at guaranteed up-front
   cost, or you can check permutations whenever you fail to locate the expected
   permutation. Up to you, validation is not expected to be highly performant
   but it should be CORRECT.

   To aid myself, I usually have the following pattern of checks/debug in all
   functions
    * Check the size first; it's the obvious way to know something went horribly
      wrong
    * Print the first "MAX_TO_PRINT" entries all the time. Makes you feel OK that
      things are proceeding as they should.
    * Print errors EVERY TIME they occur, up to "MAX_TO_PRINT" (OK values in
      the first "MAX_TO_PRINT" checks do not count against this)
    * You can tap out early after "MAX_ERRORS" are observed to prevent flooding
      the console when things are very bad.
    * If you track permutations, it can be nice to report that, but it is not
      required. Sometimes I am delusional and think I'll use it to make
      validation cohere with the device order more but that's not going to
      happen.
    * Use std::find() on the device memory to look for host values. The host
      WILL have every value, the device should generally agree on indexing
      order but it doesn't strictly have to. Usually not finding it would mean
      that you are looking for permuted orders at the expected index, don't go
      overboard looking for permutations across the entire set of returned
      values.
    * Return the boolean value true when OK, false when at least 1 error is
      found.
*/

const int MAX_TO_PRINT = 10;
const int MAX_ERRORS = 0;
const bool PRINT_PROGRESS = false;

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
                 n_failures_before_early_exit = MAX_ERRORS,
                 n_failures_to_print = MAX_TO_PRINT;
    for (const auto EV_Array : host_EV) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
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
                if (n_printed < MAX_TO_PRINT) {
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
            if (n_printed < MAX_TO_PRINT) {
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
                  n_failures_before_early_exit = MAX_ERRORS,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto FaceArray : host) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
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
                  n_failures_before_early_exit = MAX_ERRORS,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto EdgeArray : host) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process cell " << idx << " ("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        idx++;
        auto index = std::find(begin(device), end(device), EdgeArray);
        if (index == std::end(device)) {
            // Check for re-order at expected idx
            vtkIdType scan_found = 0;
            for (const vtkIdType look_for : EdgeArray) {
                for (vtkIdType i = 0; i < nbEdgesInCell; i++) {
                    if (device[idx-1][i] == look_for) {
                        scan_found++;
                        break;
                    }
                }
            }
            if (scan_found == nbEdgesInCell) {
                n_found++;
                if (n_printed < 10) {
                    std::cout << OK_EMOJI << "Matched edge set between host and device ("
                        << EdgeArray[0] << ", " << EdgeArray[1] << ", "
                        << EdgeArray[2] << ", " << EdgeArray[3] << ", "
                        << EdgeArray[4] << ", " << EdgeArray[5] << ")"
                        << std::endl;
                    n_printed++;
                }
            }
            else {
                n_failures++;
                if (n_failures <= n_failures_to_print) {
                    std::cerr << WARN_EMOJI << "Could not find set of edges ("
                        << EdgeArray[0] << ", " << EdgeArray[1] << ", "
                        << EdgeArray[2] << ", " << EdgeArray[3] << ", "
                        << EdgeArray[4] << ", " << EdgeArray[5] << ") in device TE!"
                        << std::endl;
                    std::cerr << WARN_EMOJI << "Expected match at index " << idx-1
                        << " between host (" << EdgeArray[0] << ", "
                        << EdgeArray[1] << ", " << EdgeArray[2] << ", "
                        << EdgeArray[3] << ", " << EdgeArray[4] << ", "
                        << EdgeArray[5] << ") and device (" << device[idx-1][0]
                        << ", " << device[idx-1][1] << ", " << device[idx-1][2]
                        << ", " << device[idx-1][3] << ", " << device[idx-1][4]
                        << ", " << device[idx-1][5] << ")" << std::endl;
                    n_printed++;
                }
                if (n_failures_before_early_exit > 0 &&
                        n_failures >= n_failures_before_early_exit) return false;
            }
        }
        else {
            n_found++;
            if (n_printed < MAX_TO_PRINT) {
                std::cout << OK_EMOJI << "Matched edge set between host and device ("
                    << EdgeArray[0] << ", " << EdgeArray[1] << ", "
                    << EdgeArray[2] << ", " << EdgeArray[3] << ", "
                    << EdgeArray[4] << ", " << EdgeArray[5] << ")"
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

bool check_host_vs_device_FV(FV_Data & host_FV,
                             FV_Data & device_FV) {
    // Validate that host and device agree on FV Data
    // Notably, the 'id' field of FV should be the lowest vertex ID from host;
    // the device just needs to have all 3 correct vertices in some order
    if (host_FV.size() != device_FV.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device FV size (" << device_FV.size()
                  << ") != Host FV size (" << host_FV.size()
                  << ")" << std::endl;
        return false;
    }
    /* Sorting should carry minimal cost here as there are only 3 elements
     * to compare
     */
    for (FaceData& face : host_FV) {
        vtkIdType i = face.id,
                  j = face.middleVert,
                  k = face.highVert;
        // Enfoce i < j < k
        if (i > j) std::swap(i,j);
        if (j > k) std::swap(j,k);
        // Assign in order
        face.id = i;
        face.middleVert = j;
        face.highVert = k;
    }
    for (FaceData& face : device_FV) {
        vtkIdType i = face.id,
                  j = face.middleVert,
                  k = face.highVert;
        // Enfoce i < j < k
        if (i > j) std::swap(i,j);
        if (j > k) std::swap(j,k);
        // Assign in order
        face.id = i;
        face.middleVert = j;
        face.highVert = k;
    }

    unsigned int idx = 0,
                 n_printed = 0,
                 n_found = 0,
                 n_failures = 0,
                 n_failures_before_early_exit = MAX_ERRORS,
                 n_failures_to_print = MAX_TO_PRINT;
    for (const auto face : host_FV) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process edge " << idx << " ("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        idx++;
        auto index = std::find(begin(device_FV), end(device_FV), face);
        if (index == std::end(device_FV)) {
            n_failures++;
            if (n_failures <= n_failures_to_print)
                std::cerr << WARN_EMOJI << "Could not find face (" << face.id
                          << ", " << face.middleVert << ", " << face.highVert
                          << ") in device FV!" << std::endl;
            if (n_failures_before_early_exit > 0 &&
                n_failures >= n_failures_before_early_exit) return false;
        }
        else {
            n_found++;
            if (n_printed < MAX_TO_PRINT) {
                std::cout << OK_EMOJI << "Matched face between host and device ("
                          << face.id << ", " << face.middleVert << ", "
                          << face.highVert << ")" << std::endl;
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

bool check_host_vs_device_ET(const ET_Data & host_ET, const ET_Data & device_ET) {
    std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
    return false;
}
bool check_host_vs_device_FE(const FE_Data & host_FE, const FE_Data & device_FE) {
    std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
    return false;
}
bool check_host_vs_device_FT(const FT_Data & host_FT, const FT_Data & device_FT) {
    std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
    return false;
}
bool check_host_vs_device_EF(const EF_Data & host_EF, const EF_Data & device_EF) {
    std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
    return false;
}
bool check_host_vs_device_VT(const VT_Data & host_VT, const VT_Data & device_VT) {
    std::cerr << EXCLAIM_EMOJI << "Not implemented yet" << std::endl;
    return false;
}
bool check_host_vs_device_TT(const TT_Data & host_TT, const TT_Data & device_TT) {
    if (host_TT.size() != device_TT.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device TT size (" << device_TT.size()
                  << ") != Host TT size (" << host_TT.size()
                  << ")" << std::endl;
        return false;
    }
    long long int idx = 0,
                  n_printed = 0,
                  n_found = 0,
                  n_failures = 0,
                  n_failures_before_early_exit = MAX_ERRORS,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto AdjArray : host_TT) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process cell " << idx << "("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        // Adjacencies are always aligned on first vector dimension, order within
        // does not matter so just find it there or not
        for (const auto cell : AdjArray) {
            auto index = std::find(begin(device_TT[idx]), end(device_TT[idx]), cell);
            if (index == std::end(device_TT[idx])) {
                n_failures++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cerr << WARN_EMOJI << "Could not find cell ID "
                              << cell << " as being adjacent to cell ID "
                              << idx << " in device_TT!" << std::endl;
                    n_printed++;
                }
                if (n_failures_before_early_exit > 0 &&
                        n_failures >= n_failures_before_early_exit) return false;
            }
            else {
                n_found++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cout << OK_EMOJI << "Matched cell ID " << cell
                              << " as adjacent to cell ID " << idx
                              << std::endl;
                    n_printed++;
                }
            }
        }
        idx++;
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " adjacencies"
              << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " adjacencies" << std::endl;
    return n_failures == 0;
}
bool check_host_vs_device_FF(const FF_Data & host_FF, const FF_Data & device_FF) {
    if (host_FF.size() != device_FF.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device FF size (" << device_FF.size()
                  << ") != Host FF size (" << host_FF.size()
                  << ")" << std::endl;
        return false;
    }
    long long int idx = 0,
                  n_printed = 0,
                  n_found = 0,
                  n_failures = 0,
                  n_failures_before_early_exit = MAX_ERRORS,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto AdjArray : host_FF) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process face " << idx << "("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        // Adjacencies are always aligned on first vector dimension, order within
        // does not matter so just find it there or not
        for (const auto face : AdjArray) {
            auto index = std::find(begin(device_FF[idx]), end(device_FF[idx]), face);
            if (index == std::end(device_FF[idx])) {
                n_failures++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cerr << WARN_EMOJI << "Could not find face ID "
                              << face << " as being adjacent to face ID "
                              << idx << " in device_FF!" << std::endl;
                    n_printed++;
                }
                if (n_failures_before_early_exit > 0 &&
                        n_failures >= n_failures_before_early_exit) return false;
            }
            else {
                n_found++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cout << OK_EMOJI << "Matched face ID " << face
                              << " as adjacent to face ID " << idx
                              << std::endl;
                    n_printed++;
                }
            }
        }
        idx++;
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " adjacencies"
              << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " adjacencies" << std::endl;
    return n_failures == 0;
}
bool check_host_vs_device_EE(const EE_Data & host_EE, const EE_Data & device_EE) {
    if (host_EE.size() != device_EE.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device EE size (" << device_EE.size()
                  << ") != Host EE size (" << host_EE.size()
                  << ")" << std::endl;
        return false;
    }
    long long int idx = 0,
                  n_printed = 0,
                  n_found = 0,
                  n_failures = 0,
                  n_failures_before_early_exit = MAX_ERRORS,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto AdjArray : host_EE) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process edge " << idx << "("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        // Adjacencies are always aligned on first vector dimension, order within
        // does not matter so just find it there or not
        for (const auto edge : AdjArray) {
            auto index = std::find(begin(device_EE[idx]), end(device_EE[idx]), edge);
            if (index == std::end(device_EE[idx])) {
                n_failures++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cerr << WARN_EMOJI << "Could not find edge ID "
                              << edge << " as being adjacent to edge ID "
                              << idx << " in device_EE!" << std::endl;
                    n_printed++;
                }
                if (n_failures_before_early_exit > 0 &&
                        n_failures >= n_failures_before_early_exit) return false;
            }
            else {
                n_found++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cout << OK_EMOJI << "Matched edge ID " << edge
                              << " as adjacent to edge ID " << idx
                              << std::endl;
                    n_printed++;
                }
            }
        }
        idx++;
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " adjacencies"
              << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " adjacencies" << std::endl;
    return n_failures == 0;
}
bool check_host_vs_device_VV(const VV_Data & host_VV, const VV_Data & device_VV) {
    if (host_VV.size() != device_VV.size()) {
        std::cerr << EXCLAIM_EMOJI << "Device VV size (" << device_VV.size()
                  << ") != Host VV size (" << host_VV.size()
                  << ")" << std::endl;
        return false;
    }
    long long int idx = 0,
                  n_printed = 0,
                  n_found = 0,
                  n_failures = 0,
                  n_overages = 0,
                  n_sum_amount_over = 0,
                  n_biggest_over = 0,
                  n_failures_before_early_exit = MAX_ERRORS,
                  n_failures_to_print = MAX_TO_PRINT;
    for (const auto AdjArray : host_VV) {
        if (PRINT_PROGRESS && idx % 1000 == 0)
            std::cerr << INFO_EMOJI << "Process vertex " << idx << "("
                      << n_failures << " failures so far "
                      << 100 * n_failures / static_cast<float>(idx)
                      << " %)" << std::endl;
        // Adjacencies are always aligned on first vector dimension, order within
        // does not matter so just find it there or not
        // However, our implementation COULD result in duplicate data. Check for
        // that as a non-failure condition
        long long int overage = device_VV[idx].size() - AdjArray.size();
        if (overage != 0) {
            n_overages++;
            if (n_printed < MAX_TO_PRINT) {
                std::cerr << WARN_EMOJI << overage
                          << " extra entries detected in vertex " << idx
                          << std::endl;
                n_printed++;
            }
            n_sum_amount_over += overage;
            if (overage > n_biggest_over) n_biggest_over = overage;
        }
        for (const auto vertex : AdjArray) {
            auto index = std::find(begin(device_VV[idx]), end(device_VV[idx]), vertex);
            if (index == std::end(device_VV[idx])) {
                n_failures++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cerr << WARN_EMOJI << "Could not find vertex ID "
                              << vertex << " as being adjacent to vertex ID "
                              << idx << " in device_VV!" << std::endl;
                    n_printed++;
                }
                if (n_failures_before_early_exit > 0 &&
                        n_failures >= n_failures_before_early_exit) return false;
            }
            else {
                n_found++;
                if (n_printed < MAX_TO_PRINT) {
                    std::cout << OK_EMOJI << "Matched vertex ID " << vertex
                              << " as adjacent to vertex ID " << idx
                              << std::endl;
                    n_printed++;
                }
            }
        }
        idx++;
    }
    std::cerr << INFO_EMOJI << "Matched " << n_found << " adjacencies"
              << std::endl;
    if (n_failures_before_early_exit == 0 && n_failures > 0)
        std::cerr << EXCLAIM_EMOJI << "Failed to match " << n_failures
                  << " adjacencies" << std::endl;
    if (n_overages > 0) {
        std::cerr << WARN_EMOJI << n_overages << " vertices had extra entries, "
                  << "with a total of " << n_sum_amount_over << " extra"
                  << std::endl;
        std::cerr << "Biggest amount over on a single vertex: "
                  << n_biggest_over << std::endl;
    }
    return n_failures == 0;
}

