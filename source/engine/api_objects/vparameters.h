#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VParametersCore;

class VParameters {
public:
    VParameters();
    VParameters(VSession session, string sBuiltin, VDict kwArgs = {});
    VParameters(const VParameters& src);
    VParameters(VSession session, VHParameters handle);
    VParameters(VParametersCore* core);
    virtual ~VParameters();
    VParameters& operator =(const VParameters& src);
    VHParameters cloneCore();
    VHParameters cloneHandle();
    VParametersCore* getClone();
    VParametersCore* getCore();
    bool isValid();
    void closeHandle();
    VSession session();
    int getNth();
    int getRefCnt();
    void incRefCount();
protected:
    VParametersCore* m_core;
public:
    VParameters(VSession session, VList params, string sDevice);

    void getWeights(VList& terms, VTensorDict& weights, bool bGrad);

    string getDevice();

    void zero_grad();
    void init_weights();

    VList getParams();

    static void load_param(VDict pmset, string name, FILE* fid);

protected:
    void m_lookupTensors(VList& terms, VTensorDict& tensors, VDict& nums, VList params, string kind, string def_name);
    void m_lookupTensors(VList& terms, VTensorDict& tensors, VDict& nums, VDict params, string kind, string def_name);
};
