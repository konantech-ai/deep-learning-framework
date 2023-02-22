#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vmodule.h"
#include "../api_objects/vparameters.h"

class VOptimizerCore;

class VOptimizer {
public:
    VOptimizer();
    VOptimizer(VSession session, string sBuiltin, VDict kwArgs = {});
    VOptimizer(const VOptimizer& src);
    VOptimizer(VSession session, VHOptimizer handle);
    VOptimizer(VOptimizerCore* core);
    virtual ~VOptimizer();
    VOptimizer& operator =(const VOptimizer& src);
    VHOptimizer cloneCore();
    VHOptimizer cloneHandle();
    VOptimizerCore* getClone();
    VOptimizerCore* getCore();
    bool isValid();
    void closeHandle();
    VSession session();
    int getNth();
    int getRefCnt();
    void incRefCount();
protected:
    VOptimizerCore* m_core;
public:
    VOptimizer(VSession session, VParameters parameters, string sBuiltin, VDict kwArgs);

    void set_option(VDict kwArgs);
    void step();

    static VList GetBuiltinNames();

    static void createParam(VSession session, string name, VShape shape, bool needGrad, string init_method, float init_arg, VDict pm);
    static void createEmptyParam(VSession session, string name, VDict pm);
    static void createAffineParam(VSession session, VShape wshape, bool use_bias, string init_method, float init_arg, VDict pm, string prefix = "");
    static void createRnnParam(VSession session, int nGates, int64 nRecurSize, int64 nInputSize, int64 nLayers, bool bi_direct, bool use_bias, VDict pm);
    //static void createLstmParam(VSession session, VShape wshape, bool use_bias, VDict pm, string prefix = "");
    static void createBiasParam(VSession session, VShape bshape, string init_method, float init_arg, VDict pm);
    static void createBatchNormalParam(VSession session, VShape wshape, bool rescale, bool shift, VDict pm);

    VTensor preproc(VTensor param, VDict pmset, VExecTracer tracer);

protected:
    static void ms_createUniformParam(VSession session, VShape wishape, VShape wrshape, float range, bool use_bias, VDict pm, string prefix);

protected:
    static VStrList ms_builtin;

};
