#pragma once

class VTensor;
class VCbItem;

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VCbBackInfoCore;

class VCbBackInfo{
public:
    VCbBackInfo();
    VCbBackInfo(VSession session, string sBuiltin, VDict kwArgs = {});
    VCbBackInfo(const VCbBackInfo& src);
    VCbBackInfo(VCbBackInfoCore* core);
    virtual ~VCbBackInfo();
    VCbBackInfo& operator =(const VCbBackInfo& src);
    VCbBackInfoCore* getClone();
    VCbBackInfoCore* getCore();
    void destroyCore();
    VSession session() const;
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VCbBackInfoCore* m_core;
public:
	void addDeviceConvInfo(int nHostId, VTensor tensor);
	void addCbRequestSlot(
		VModuleCore* pStarter, VCbItem item, string name, int nDevice, bool bPre,
		VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params);

protected:
};
