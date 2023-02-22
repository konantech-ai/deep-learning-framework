#pragma once

#include "../include/vapi.h"

class VExceptionCore : VObjCore {
protected:
	VExceptionCore() : VObjCore(VObjType::exception) {
		m_nErrCode = 0;
		m_nRefCount = 1;
	}

	void destroy() {
		if (--m_nRefCount <= 0) delete this;
	}

	VExceptionCore* clone() {
		m_nRefCount++;
		return this;
	}

	friend class VException;

	int m_nErrCode;
	int m_nRefCount;
	
	VList m_msgStack;
};

class VException {
public:
	VException();
	VException(VExceptionCore* core);
	VException(const VException& src);
	VException(int nErrCode, string file, int line);
	VException(int nErrCode, string sParam, string file, int line);
	VException(int nErrCode, string sParam1, string sParam2, string file, int line);
	VException(int nErrCode, string sParam1, string sParam2, string sParam3, string file, int line);
	VException(int nErrCode, const VException& src, string file, int line);
	VException(int nErrCode, const VException& src, string sParam1, string file, int line);

	virtual ~VException();

	VException& operator =(const VException& src);

	VExceptionCore* cloneCore() { return m_core->clone(); }

	VRetCode GetErrorCode();
	VList GetErrorMessageList();

	static VList GetErrorMessageList(VRetCode nErrorCode);

protected:
	VExceptionCore* m_core;

	string m_createMessage(int nErrCode, string file, int line, string sParam1 = "", string sParam2 = "", string sParam3 = "");
	
	static void ms_createMessage(char* pBuf, int nErrCode, const char* p1, const char* p2, const char* p3);
};
