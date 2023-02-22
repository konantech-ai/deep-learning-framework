#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"

class ELossCore : public EcoObjCore {
protected:
    friend class ELoss;
protected:
    ELossCore(ENN nn, VHLoss hELoss);
    ~ELossCore();
    ELossCore* clone() { return (ELossCore*)clone_core(); }
    void m_setup();
    void m_delete();
protected:
    ENN m_nn;
    //VHLoss m_hLoss;

protected:
    void m_setup(string sName, string sBuiltin, VDict kwArgs);

protected:
    string m_sBuiltin;
    string m_sName;

    ELossDict m_losses;
    //ETensorDict m_result;
};
