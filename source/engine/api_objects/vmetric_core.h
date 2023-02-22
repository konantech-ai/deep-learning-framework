#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vgraph.h"

class VMetricCore : public VObjCore {
protected:
    friend class VMetric;
protected:
    VMetricCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VMetricCore* clone() { return (VMetricCore*)clone_core(); }
    ~VMetricCore();
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

    VList m_downloadTerms;

    VExecTracer m_execTracer;
};
