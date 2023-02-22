#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VMetricCore;

class VMetric {
public:
    VMetric();
    VMetric(VSession session, string sBuiltin, VDict kwArgs = {});
    VMetric(const VMetric& src);
    VMetric(VSession session, VHMetric handle);
    VMetric(VMetricCore* core);
    virtual ~VMetric();
    VMetric& operator =(const VMetric& src);
    VHMetric cloneCore();
    VHMetric cloneHandle();
    VMetricCore* getClone();
    VMetricCore* getCore();
    bool isValid();
    void closeHandle();
    VSession session();
    int getNth();
    int getRefCnt();
    void incRefCount();
protected:
    VMetricCore* m_core;
public:
    VTensorDict evaluate(VTensorDict preds);

    static VList GetBuiltinNames();

protected:
    static VStrList ms_builtin;

    };
