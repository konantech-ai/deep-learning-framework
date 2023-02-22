#pragma once

#include "../api/vdefine.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vhypermanager.h"
#include "../local_objects/vcbitem.h"

// VSessionCore는 m_session 멤버를 가지지 말아야 하므로 DECLARE_CORE_BASE_BEGIN 매크로를 이용한다.

class VCbBackSlot;

class VSessionCore : public VObjCore {
protected:
    friend class VSession;

protected:
    VSessionCore();
    virtual ~VSessionCore();

    VSessionCore* clone() { return (VSessionCore*)clone_core(); }
    VSessionCore* cloneHandle() { return (VSessionCore*)clone_handle(); }

protected:
    VDict m_propDict;

    VDeviceManager m_deviceManager;
    VHyperManager m_hyperManager;

    bool m_noGrad;
    bool m_noTracer;

    VCbCustomModuleExec* m_customModuleExecCbFunc;
    VCbFreeReportBuffer* m_freeReportBufferCbFunc;

    void* m_pCustomModuleExecInst;
    void* m_pCustomModuleExecAux;

    void* m_pFreeReportBufferInst;
    void* m_pFreeReportBufferAux;

    bool m_bNeedForwardCallbeck;
    bool m_bNeedBackwardCallbeck;

    map<int, VCbItem> m_cbForwardItemMap;
    map<int, VCbItem> m_cbBackwardItemMap;

    vector<VCbBackSlot> m_callbackSlots;

    VFuncCbHandlerInfo m_funcCbHadlerInfo;

    map<string,VFunction> m_userFuncMap;
    
    VException m_lastError;

    VDict m_macros;

    int m_nCheckCode;
    static int ms_nCheckCode;

protected:
    int m_filterCheck(VCbItem item, string name, bool train, int nDevice, bool bPre);
    int m_filterCheck(VCbItem item, string name, int nDevice);

    void m_invokeCallback(
        VModuleCore* pStarter, VCbItem item, string name, bool train, int nDevice, bool bPre,
        bool noGrad, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VExecTracer tracer);

    void m_registCallbackSlot(VCbBackSlot slot);

    bool m_lookupUserDefinedFunctions(string opCode, VValue* pFunction);

    void m_closeObjectInfo();
};
