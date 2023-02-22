#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vgraph.h"

class VLossCore : public VObjCore {
protected:
    friend class VLoss;
protected:
    VLossCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VLossCore* clone() { return (VLossCore*)clone_core(); }
    ~VLossCore();
    void m_onCreate();
    void m_onDelete();
    VSession session() { return m_session; }
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;
    int m_nCheckCode;
    static int ms_nCheckCode;

protected:
    VGraph m_graph;
    VGraphDict m_graphs;
    VTensorDict m_preds;
    VTensorDict m_losses;
    VTensorDict m_staticTensors;

    VTensorDict m_slices;

    VList m_downloadTerms;

    VExecTracer m_execTracer[3][2][3];    // [0:train, 1:test], [0:forward, 1:backward, 2:accuracy]
};
