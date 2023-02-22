#pragma once

#include "../api/vdefine.h"

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VLossCore;

class VLoss {
public:
    VLoss();
    VLoss(VSession session, string sBuiltin, VDict kwArgs = {});
    VLoss(const VLoss& src);
    VLoss(VSession session, VHLoss handle);
    VLoss(VLossCore* core);
    virtual ~VLoss();
    VLoss& operator =(const VLoss& src);
    VHLoss cloneCore();
    VHLoss cloneHandle();
    VLossCore* getClone();
    VLossCore* getCore();
    bool isValid();
    void closeHandle();
    VSession session();
    int getNth();
    int getRefCnt();
    void incRefCount();
protected:
    VLossCore* m_core;
public:
    VTensorDict evaluate(VTensorDict preds, VTensorDict ys, bool download_all);
    VTensorDict eval_accuracy(VTensorDict preds, VTensorDict ys, bool download_all);

    void backward();

    static VList GetBuiltinNames();

protected:
    VExecTracer m_getTracer(VTensorDict xs, int mission);

    VTensorDict m_preprocLossOpnd(VTensorDict preds, VTensorDict ys);
    VTensorDict m_evaluateLoss(VTensorDict xs, int nDevice, VExecTracer tracer);
    VTensorDict m_evaluateMultipleLoss(VTensorDict preds, VTensorDict ys, bool download_all);
    VTensorDict m_evaluateCustomLoss(VTensorDict xs, int nDevice, VExecTracer tracer, bool download_all);

    static VStrList ms_builtin;

};
