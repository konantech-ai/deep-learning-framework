#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vtensor.h"

class VFunctionCore;

class VUDFItemCore : public VObjCore {
protected:
	friend class VUDFItem;
protected:
	VUDFItemCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
	VUDFItemCore* clone() { return (VUDFItemCore*)clone_core(); }
	VSession session() { return m_session; }
	void m_setup();
protected:
	VSession m_session;
	string m_sBuiltin;
	VDict m_propDict;

protected: // 공용 필드
	virtual ~VUDFItemCore();
	VFunctionCore* m_functor;
	
	VTensor m_y;
	VTensorList m_xs;

	VTensor m_ygrad;
	VTensorList m_xgrads;

	VDict m_opArgs;

	static int ms_intCnt;

};
