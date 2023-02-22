#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VCbItem;

class VCbBackInfoCore : public VObjCore {
protected:
    friend class VCbBackInfo;
protected:
    VCbBackInfoCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VCbBackInfoCore* clone() { return (VCbBackInfoCore*)clone_core(); }
    VSession session() { return m_session; }
    void m_setup();
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;

protected:
    VTensorMap m_devConvParams;

protected:
    void m_addCbRequestSlot(
        VModuleCore* pStarter, VCbItem item, string name, int nDevice,
        VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params);

};
