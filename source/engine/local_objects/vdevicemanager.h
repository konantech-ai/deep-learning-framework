#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VDeviceManagerCore;

class VDeviceManager{
public:
    VDeviceManager();
    VDeviceManager(string sBuiltin, VDict kwArgs = {});
    VDeviceManager(const VDeviceManager& src);
    VDeviceManager(VDeviceManagerCore* core);
    virtual ~VDeviceManager();
    VDeviceManager& operator =(const VDeviceManager& src);
    VDeviceManagerCore* getClone();
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VDeviceManagerCore* m_core;
public:
	void setUsingCudaFlag(int nModuleId, int flag);

	static int getDeviceCount();

	int getUsingDeviceCount(int nModuleId);
	int getNthUsingDevice(int nModuleId, int nth);

	int getCurDevice();
	int setCurDevice(int nDevice, VExecTracer tracer);	// 원복을 위해 기존 디바이스 번호를 반환한다.

	int64 getMaxBatchSize(int64 nNeedSizePerbatch);

};
