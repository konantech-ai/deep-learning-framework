#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"

class EFunctionCore : public EcoObjCore{
protected:
    friend class EFunction;
protected:
    EFunctionCore(ENN nn, VHFunction hFunction);
    ~EFunctionCore();
    EFunctionCore* clone() { return (EFunctionCore*)clone_core(); }
    void m_setup();
    void m_delete();
protected:
    ENN m_nn;
    //VHFunction m_hFunction;

protected:
    string m_name;
};
