#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

typedef vector< VExecTracer> VExecTracerList;

class VExecTracerPoolCore : public VObjCore {
protected:
    friend class VExecTracerPool;
protected:
    VExecTracerPoolCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VExecTracerPoolCore* clone() { return (VExecTracerPoolCore*)clone_core(); }
    VSession session() { return m_session; }
    void m_setup();
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;

protected:
	VExecTracerList m_tracers;

};
