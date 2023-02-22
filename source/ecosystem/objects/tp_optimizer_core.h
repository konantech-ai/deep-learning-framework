#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_parameters.h"
#include "../objects/tp_nn.h"

class EOptimizerCore : public EcoObjCore {
protected:
    friend class EOptimizer;
protected:
    EOptimizerCore(ENN nn, VHOptimizer hOptimizer);
    ~EOptimizerCore();
    EOptimizerCore* clone() { return (EOptimizerCore*)clone_core(); }
    void m_setup();
    void m_delete();
protected:
    ENN m_nn;
    //VHOptimizer m_hOptimizer;

public:
    string m_sBuiltin;
    EParameters m_parameters;
    VDict m_kwArgs;
};
