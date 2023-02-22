#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

typedef float* HYPER;
typedef int HYPER_KEY;

#define HYPER_ACCESS(x) (x ? *x : 0)
#define HYPER_FETCH(x) session().hyper_man().fetch(device(), x)
#define HYPER_FETCH_CPU(x) session().hyper_man().fetch(-1, x)
#define HYPER_FETCH_DEV(dev, x) session().hyper_man().fetch(dev, x)
#define HYPER_REGIST(x) x = session().hyper_man().regist()
#define HYPER_REGVAL(x) session().hyper_man().registValue(x)
#define HYPER_REGVAL_CORE(x) m_session.hyper_man().registValue(x)
#define HYPER_SET(x, v) session().hyper_man().set(x, v)
#define HYPER_GET(x) session().hyper_man().get(x)

class VHyperManagerCore;

class VHyperManager {
public:
    VHyperManager();
    VHyperManager(string sBuiltin, VDict kwArgs = {});
    VHyperManager(const VHyperManager& src);
    VHyperManager(VHyperManagerCore* core);
    virtual ~VHyperManager();
    VHyperManager& operator =(const VHyperManager& src);
    VHyperManagerCore* getClone();
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VHyperManagerCore* m_core;

public:
	int regist();
	int registValue(float value);
	float* fetch(int device, int key);
	void set(int key, float value);
	float get(int key);

};
