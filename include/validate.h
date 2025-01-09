#ifndef TETRA_VALIDATION
#define TETRA_VALIDATION

#include "datatypes.h"
#include "emoji.h"

bool check_host_vs_device_TE(const TE_Data & host_TE, const TE_Data & device_TE);
bool check_host_vs_device_EV(const EV_Data & host_EV, const EV_Data & device_EV);
bool check_host_vs_device_ET(const ET_Data & host_ET, const ET_Data & device_ET);
// These two drop the 'const'-ness to permit sorting in place that facilitates validation
bool check_host_vs_device_TF(TF_Data & host_TF, TF_Data & device_TF);
bool check_host_vs_device_FV(FV_Data & host_FV, FV_Data & device_FV);
bool check_host_vs_device_FE(const FE_Data & host_FE, const FE_Data & device_FE);
bool check_host_vs_device_FT(const FT_Data & host_FT, const FT_Data & device_FT);
bool check_host_vs_device_EF(const EF_Data & host_EF, const EF_Data & device_EF);
bool check_host_vs_device_VT(const VT_Data & host_VT, const VT_Data & device_VT);
bool check_host_vs_device_TT(const TT_Data & host_TT, const TT_Data & device_TT);
bool check_host_vs_device_FF(const FF_Data & host_FF, const FF_Data & device_FF);
bool check_host_vs_device_EE(const EE_Data & host_EE, const EE_Data & device_EE);
bool check_host_vs_device_VV(const VV_Data & host_VV, const VV_Data & device_VV);

#endif

