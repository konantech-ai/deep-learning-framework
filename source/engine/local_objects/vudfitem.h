#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VUDFItemCore;
class VFunctionCore;

class VUDFItem {
public:
    VUDFItem();
    VUDFItem(VSession session, string sBuiltin, VDict kwArgs = {});
    VUDFItem(const VUDFItem& src);
    VUDFItem(VUDFItemCore* core);
    virtual ~VUDFItem();
    VUDFItem& operator =(const VUDFItem& src);
    VUDFItemCore* getClone();
    VUDFItemCore* getCore();
    void destroyCore();
    VSession session() const;
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VUDFItemCore* m_core;

public:
	VUDFItem(VSession session, VFunctionCore* functor, VTensor y, VTensorList operands, VDict opArgs);

	void setGrad(int nth, VTensor ygrad, VTensor xgrad);

	//VUDFItemCore* getCore();
	// 
	VFunctionCore* getFunctor();
	VTensorList getXs();
	VDict getOpArgs();
	VTensor getY();
	VTensor getYGrad();
	VTensorList getXGrads();

	void dump(string title);
protected:
};
