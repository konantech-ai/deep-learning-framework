#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"

class ETensorDataCore : public VObjCore {
protected:
    friend class ETensorData;
protected:
    ETensorDataCore(ENN nn);
    ETensorDataCore* clone() { return (ETensorDataCore*)clone_core(); }
    void m_setup();
protected:
    ENN m_nn;

protected:
    ~ETensorDataCore();

    int64 m_byteSize;
    void* m_pData;

    void m_fillFloatData(VShape shape, int64 nth, VList values);
    void m_fillIntData(VShape shape, int64 nth, VList values);
    void m_fillData(VShape shape, float value);
    void m_fillData(VShape shape, int value);
};
