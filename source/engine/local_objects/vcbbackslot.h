#pragma once

class VTensor;
class VCbItem;

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VCbBackSlotCore;

class VCbBackSlot {
public:
    VCbBackSlot();
    VCbBackSlot(VSession session, string sBuiltin, VDict kwArgs = {});
    VCbBackSlot(const VCbBackSlot& src);
    VCbBackSlot(VCbBackSlotCore* core);
    virtual ~VCbBackSlot();
    VCbBackSlot& operator =(const VCbBackSlot& src);
    VCbBackSlotCore* getClone();
    VCbBackSlotCore* getCore();
    void destroyCore();
    VSession session() const;
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VCbBackSlotCore* m_core;
public:
	friend class VCbBackInfoCore;
	
	void addCbRequestSlot(
		VModuleCore* pStarter, VCbItem item, string name, int nDevice,
		VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters param, VTensorMap devConvParamss);
		
	void close();
	void fillAndInvokeOnFull(VTensor tensor, VTensor grad, VExecTracer tracer);
	void replace(int nOldId, VTensor newTensor);
	
protected:
};
