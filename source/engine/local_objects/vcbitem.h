#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VCbItemCore;

class VCbItem {
public:
    VCbItem();
    VCbItem(VSession session, string sBuiltin, VDict kwArgs = {});
    VCbItem(const VCbItem& src);
    VCbItem(VCbItemCore* core);
    virtual ~VCbItem();
    VCbItem& operator =(const VCbItem& src);
    VCbItemCore* getClone();
    VCbItemCore* getCore();
    void destroyCore();
    VSession session() const;
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VCbItemCore* m_core;
public:
	VCbItem(VSession session, void* pCbFunc, void* pCbClose, VDict filters, VDict instInfo);
	VCbItem(VSession session, void* pCbFunc, void* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict);

	void* getCbFunc();
	void* getCbClose();

	VDict getInstInfo();

	VDict getFilters();

	VDict getStatusInfo();
	VDict getTensorDict();
	VDict getGradDict();

protected:
};
