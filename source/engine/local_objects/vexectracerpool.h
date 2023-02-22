#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VExecTracerPoolCore;

class VExecTracerPool {
public:
    VExecTracerPool();
    VExecTracerPool(VSession session, string sBuiltin, VDict kwArgs = {});
    VExecTracerPool(const VExecTracerPool& src);
    VExecTracerPool(VExecTracerPoolCore* core);
    virtual ~VExecTracerPool();
    VExecTracerPool& operator =(const VExecTracerPool& src);
    VExecTracerPoolCore* getClone();
    VExecTracerPoolCore* getCore();
    void destroyCore();
    VSession session() const;
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VExecTracerPoolCore* m_core;

public:
	bool openAndExecute(VTensorDict xs, int nMaxTracer, VTensorDict& preds, VExecTracer& tracer);

};
