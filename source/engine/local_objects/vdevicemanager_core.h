#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VDeviceManagerCore : public VObjCore {
protected:
    friend class VDeviceManager;
protected:
    VDeviceManagerCore(string sBuiltin = "", VDict kwArgs = {});
    VDeviceManagerCore* clone() { return (VDeviceManagerCore*)clone_core(); }
    void m_setup();
protected:
    string m_sBuiltin;
    VDict m_propDict;

protected:
    map<std::thread::id, int> m_curDevice;
	map<int,int> m_moduleDeviceFlags;
};
