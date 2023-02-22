#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vmodule.h"
#include "../local_objects/vhypermanager.h"


class VOptimizerCore : public VObjCore {
protected:
    friend class VOptimizer;
protected:
    VOptimizerCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VOptimizerCore* clone() { return (VOptimizerCore*)clone_core(); }
    ~VOptimizerCore();
    void m_onCreate();
    void m_onDelete();
    VSession session() { return m_session; }
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;
    int m_nCheckCode;
    static int ms_nCheckCode;

protected:
    OptAlgorithm m_optAlgorithm;
    VParameters m_parameters;
    VExecTracer m_execTracer;

    HYPER_KEY m_learning_rate;

    HYPER_KEY m_l1Decay;
    HYPER_KEY m_l2Decay;

    HYPER_KEY m_ro1;            // for Adam
    HYPER_KEY m_ro2;            // for Adam
    HYPER_KEY m_epsilon;        // for Adam
    HYPER_KEY m_sigma;          // for AdaGrad, RMSProp
    HYPER_KEY m_decay;          // for RMSProp
    HYPER_KEY m_momentum;       // for Momentum

    bool m_bUseDecay;

    string m_initMethod;    // for Momentum, AdaGrad, RMSProp

protected:
    void m_setup(VParameters parameters);
    void m_setOption(VDict kwArgs);

    void m_prepareParam(VList params, int nDevice);
    void m_prepareParam(VDict params, int nDevice);
    void m_preparePmSet(VDict pmset, int nDevice);

    void m_step(VList params, VExecTracer tracer);
    void m_optimize(VDict pmset, VExecTracer tracer);
    VTensor m_applyDecay(VDict pmset, VTensor grad, VExecTracer tracer);
    VTensor m_evalAdamDelta(VDict pmset, VTensor grad, VExecTracer tracer);
    VTensor m_evalMomentumDelta(VDict pmset, VTensor grad, VExecTracer tracer);
    VTensor m_preprocNesterovParam(VTensor param, VDict pmset, VExecTracer tracer);
    VTensor m_evalAdaGradDelta(VDict pmset, VTensor grad, VExecTracer tracer);
    VTensor m_evalRMSPropDelta(VDict pmset, VTensor grad, VExecTracer tracer);

    static VDict ms_init_module_pmset(VSession session, VShape shape, int nDevice, bool bNeedGrad, string init_type, float init_arg);
    static VDict ms_create_empty_pmset(VSession session);
    static VTensor ms_init_module_pm_tensor(VSession session, VShape shape, int nDevice, bool bNeedGrad, string init_type, float init_arg);
};
