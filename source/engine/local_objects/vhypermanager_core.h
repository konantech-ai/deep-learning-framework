#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

#define PAGE_SIZE 1024

class VHyperManagerCore : public VObjCore {
protected:
	friend class VHyperManager;
protected:
	VHyperManagerCore(string sBuiltin = "", VDict kwArgs = {});
	VHyperManagerCore* clone() { return (VHyperManagerCore*)clone_core(); }
	void m_setup();
protected:
	string m_sBuiltin;
	VDict m_propDict;
	
protected:
	virtual ~VHyperManagerCore();

	int m_nAllocSize;
	int m_nUsingCount;

	float* m_pHyperBase;

	map<int, vector<bool>> m_dirtyMaps;
	map<int, vector<float*>> m_hyperMaps;

protected:
	void m_allocAdditionalPage();
	void m_setValue(int key, float value);
	void m_dirtyCheck(int device, int page);

};
