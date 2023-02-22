#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"

class EParametersCore : public EcoObjCore {
protected:
    friend class EParameters;
protected:
    EParametersCore(ENN nn, VHParameters hParameters);
    ~EParametersCore();
    EParametersCore* clone() { return (EParametersCore*)clone_core(); }
    void m_setup();
    void m_delete();
protected:
    ENN m_nn;
    //VHParameters m_hParameters;

public:
};