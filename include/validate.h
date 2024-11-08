#ifndef TETRA_VALIDATION
#define TETRA_VALIDATION

#include "datatypes.h"
#include "emoji.h"

bool check_host_vs_device_EV(const EV_Data & host_EV, const EV_Data & device_EV);
bool check_host_vs_device_TE(const TE_Data & host_TE, const TE_Data & device_TE);
bool check_host_vs_device_ET(const ET_Data & host_ET, const ET_Data & device_ET);
bool check_host_vs_device_TF(const TF_Data & host_TF, const TF_Data & device_TF);
bool check_host_vs_device_FV(const FV_Data & host_FV, const FV_Data & device_FV);
bool check_host_vs_device_FE(const FE_Data & host_FE, const FE_Data & device_FE);

#endif

