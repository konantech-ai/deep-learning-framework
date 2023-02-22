#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VFunctionCore : public VObjCore {
protected:
    friend class VFunction;
protected:
    VFunctionCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VFunctionCore* clone() { return (VFunctionCore*)clone_core(); }
    ~VFunctionCore();
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
    string m_sName;
    void* m_pCbAux;
};