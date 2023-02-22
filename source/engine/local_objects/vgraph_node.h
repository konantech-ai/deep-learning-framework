#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vexectracer.h"
#include "../local_objects/vcbbackinfo.h"

class VGraphNodeCore;

class VGraphNode {
public:
    VGraphNode();
    VGraphNode(VSession session, string sBuiltin, VDict kwArgs = {});
    VGraphNode(const VGraphNode& src);
    VGraphNode(VGraphNodeCore* core);
    virtual ~VGraphNode();
    VGraphNode& operator =(const VGraphNode& src);
    VGraphNodeCore* getClone();
    VGraphNodeCore* getCore();
    void destroyCore();
    VSession session() const;
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VGraphNodeCore* m_core;

public:
    VGraphNode(VSession session, string sExp, VList params, int& pos, VDeviceManager devman, VGraphCore* pGraphCore);
    VGraphNode(bool bDeepCopy, const VGraphNode& src, VDeviceManager devman, VGraphCore* pGraphCore);
    VGraphNode(VGraphNodeCore* pSrc, VGraphOpCode opCode, vector<VGraphNode> children);

    VTensor evaluate(VDict args, VCbBackInfo cbInfo, VExecTracer tracer);
    
    static string GraphOpCodeName(VGraphOpCode opcode);

protected:
    bool m_lookupNeedGrad(VTensorDict xs);

    VTensor m_createTensor(VValue value, VExecTracer tracer);

    static bool ms_bExprTrace;

};
