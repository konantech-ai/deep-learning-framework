#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vmodule.h"
#include "../api_objects/voptimizer.h"
#include "../local_objects/vgraph_node.h"
#include "../local_objects/vexectracer.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vcbbackinfo.h"

class VGraphCore : public VObjCore {
protected:
    friend class VGraph;
protected:
    VGraphCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VGraphCore* clone() { return (VGraphCore*)clone_core(); }
    VSession session() { return m_session; }
    void m_setup();
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;

protected:
    friend class VGraphNode;

    VGraphNode m_rootNode;
    //VModuleCore* m_pModuleCore;

    VDeviceManager m_deviceMan;
    VOptimizer m_pOptimizer;

    int m_nTouchVersion;

    string m_sNetName;
    //VModuleCore* m_pNetModuleCore;

    VTensorDict m_xs;
    VTensorDict m_ys;

    bool m_train;
    bool m_noGrad;

    int m_device;

    VTensorDict m_sideTerms; // concat 레이어 등에서 인자 정보로 접근되는 추가적 텐서 정보를 나타내는데 사용함

    VTensorDict* m_pCustomTerms; // custom loss 수식 처리에만 사용함
    VGraphDict m_graphs; // custom loss 수식 처리에만 사용함

    bool m_pmAllocaed;  // 모듈 파라미터가 실제 메모리 할당을 받았는지 구조만 보여주기 위한 더미 상태인지를 나타냄

    static int ms_nNextTouchVersion;

public:
    VTensor seekInput(string varName);  // Module에서 레이어 처리를 직접 하는 경우에 정보 제공
    VTensor optimizerPreproc(VTensor param, VDict pmset, VExecTracer tracer);
    bool isParamAllocated() { return m_pmAllocaed; }

protected:
    VTensor m_seekTerm(string varName, VCbBackInfo cbInfo, VExecTracer tracer);

    void m_keepOutput(string varName, VTensor tensor);

};
