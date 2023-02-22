#pragma once

#include "../utils/tp_common.h"

class ELossCore;
class ELoss {
public:
    ELoss();
    ELoss(ENN nn);
    ELoss(ENN nn, VHLoss hELoss);
    ELoss(const ELoss& src);
    ELoss(ELossCore* core);
    virtual ~ELoss();
    ELoss& operator =(const ELoss& src);
    operator VHLoss();
    bool isValid();
    void close();
    ENN nn();
    ELossCore* getCore();
    ELossCore* cloneCore();
    int meNth();
    int meRefCnt();
    int handleNth();
    int handleRefCnt();
    ELossCore* createApiClone();

protected:
    ELossCore* m_core;

public:
    ELoss(ENN nn, VHLoss hLoss, ELossDict losses);
    //ELoss(ENN nn, string sName, VDict kewArgs = {});

    ETensor __call__(ETensor pred, ETensor y, bool download_all = false);
    ETensorDict __call__(ETensorDict preds, ETensorDict ys, bool download_all = false);

    virtual ETensorDict evaluate(ETensorDict preds, ETensorDict ys, bool download_all = false);

    ETensor eval_accuracy(ETensor pred, ETensor y, bool download_all = false);
    ETensorDict eval_accuracy(ETensorDict preds, ETensorDict ys, bool download_all = false);

    void backward();

protected:
    void m_setExpressions(VDict lossTerms, VDict subTerms);
    //ETensor m_call(ETensorDict pred, ETensorDict y);
};
