#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"

class EMetricCore : public EcoObjCore{
protected:
    friend class EMetric;
protected:
    EMetricCore(ENN nn, VHMetric hMetric);
    ~EMetricCore();
    EMetricCore* clone() { return (EMetricCore*)clone_core(); }
    void m_setup();
    void m_delete();
protected:
    ENN m_nn;
    //VHMetric m_hMetric;

protected:
    void m_setup(string sName, string sBuiltin, VDict kwArgs);

protected:
    string m_sBuiltin;
    string m_sName;

    EMetricDict m_inferences;
};
