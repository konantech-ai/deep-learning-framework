#pragma once

#include "../utils/tp_common.h"

class EMetricCore;

class EMetric {
public:
    EMetric();
    EMetric(ENN nn);
    EMetric(ENN nn, VHMetric hMetric);
    EMetric(const EMetric& src);
    EMetric(EMetricCore* core);
    virtual ~EMetric();
    EMetric& operator =(const EMetric& src);
    operator VHMetric();
    bool isValid();
    void close();
    ENN nn();
    EMetricCore* getCore();
    EMetricCore* cloneCore();
    int meNth();
    int meRefCnt();
    int handleNth();
    int handleRefCnt();
    EMetricCore* createApiClone();
protected:
    EMetricCore* m_core;
public:

public:
    EMetric(ENN nn, VHMetric hMetric, EMetricDict inferences);
    //ELoss(ENN nn, string sName, VDict kewArgs = {});

    ETensor __call__(ETensor pred);
    ETensorDict __call__(ETensorDict preds);

    virtual ETensorDict evaluate(ETensorDict preds);

protected:
    void m_setExpressions(VDict infTerms, VDict subTerms);
};
