#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VParameters;
class VCbItem;

class VCbBackSlotCore : public VObjCore {
protected:
    friend class VCbBackSlot;
protected:
    VCbBackSlotCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VCbBackSlotCore* clone() { return (VCbBackSlotCore*)clone_core(); }
    VSession session() { return m_session; }
    void m_setup();
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;

protected:
    void m_addCbRequestSlot(
        VModuleCore* pStarter, VCbItem item, string name, int nDevice,
        VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VTensorMap devConvParams);

    void m_freeDependentTensors();
    void m_fillAndInvokeOnFull(VTensor tensor, VTensor grad, VExecTracer tracer);
    void m_replace(int nOldId, VTensor newTensor);
    void m_invokeCallback(bool bPre, VExecTracer tracer);

protected:
    friend class VCbBackInfoCore;

    VCbBackwardModule* m_pCbFunc;
    VCbClose* m_pCbClose;

    VDict m_instInfo;
    VDict m_statusInfo;

    VDict m_cbFilters;

    VTensorDict m_tensors[4];
    VTensorDict m_gradients[4];

    VMap m_tensorMap;

    //int m_nUnresolvedCount;
    int m_nUnresolvedInputCount;
    int m_nUnresolvedOutputCount;

};
