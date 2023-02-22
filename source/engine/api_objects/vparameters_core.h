#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vmodule.h"

class VParametersCore : public VObjCore {
protected:
    friend class VParameters;
protected:
    VParametersCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VParametersCore* clone() { return (VParametersCore*)clone_core(); }
    ~VParametersCore();
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
    VList m_params;
    string m_sDevice;

protected:
    void m_zero_grad(VList params, int nDevice);
    void m_init_weights(VList params, int nDevice);
};
