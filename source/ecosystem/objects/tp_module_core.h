#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"
#include "../objects/tp_loss.h"
#include "../objects/tp_optimizer.h"
#include "../objects/tp_parameters.h"

class TpStreamOut;

class EModuleCore : public EcoObjCore {
protected:
    friend class EModule;
protected:
    EModuleCore(ENN nn, VHModule hEModule);
    ~EModuleCore();
    EModuleCore* clone() { return (EModuleCore*)clone_core(); }
    void m_setup();
    void m_delete();
protected:
    ENN m_nn;
    //VHModule m_hModule;

protected:
    void m_setup(string sName, string sBuiltin, EModuleType moduleType, VDict kwArgs);
    //void m_setup(EModuleCore* pSrcCore);

    string m_desc(int depth, int nth, int64& pm_total, string pos);

    void m_saveModel(TpStreamOut& fout);

protected:
    string m_sName;
    string m_sBuiltin;
    string m_device;

    bool m_train;

    EModuleType m_moduleType;

    //EModuleList m_children;

    VDict m_kwArgs;

    //ELoss m_loss;
    //EOptimizer m_optimizer;
    EParameters m_parameters;

    friend class ModuleExt;
};
