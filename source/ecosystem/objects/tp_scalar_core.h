#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_nn.h"
#include "../utils/tp_utils.h"

class EScalarCore : public VObjCore {
protected:
    friend class EScalar;

protected:
    EScalarCore(ENN nn);
    EScalarCore* clone() { return (EScalarCore*)clone_core(); }
    void m_setup();
protected:
    ENN m_nn;

protected:
    float m_value;
    ETensor m_srcTensor;
};
