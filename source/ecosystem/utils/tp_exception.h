#pragma once

#include "../utils/tp_common.h"

#define VERR_SHAPE_TENSOR							20000

class TpExceptionCore : VObjCore {
protected:
	TpExceptionCore() : VObjCore(VObjType::exception) {
		m_nErrCode = 0;
		m_nRefCount = 1;
	}

	void destroy() {
		if (--m_nRefCount <= 0) delete this;
	}

	TpExceptionCore* clone() {
		m_nRefCount++;
		return this;
	}

	friend class TpException;

	int m_nErrCode;
	int m_nRefCount;
	VList m_msgStack;
};

class TpException {
public:
	TpException();
	TpException(TpExceptionCore* core);
	TpException(const TpException& src);
	TpException(int retCode, VList messages);
	TpException(int nErrCode, string sParam1, TpException ex, string file, int line);
	TpException(int nErrCode, string file, int line);
	TpException(int nErrCode, string sParam, string file, int line);
	//TpException(int nErrCode, string sParam1, string sParam2, string file, int line);
	//TpException(int nErrCode, const TpException& src, string file, int line);

	virtual ~TpException();

	TpException& operator =(const TpException& src);

	TpExceptionCore* cloneCore() { return m_core->clone(); }

	VRetCode GetErrorCode();

	string GetDetailErrorMessage(int indent);

	VList GetErrorMessageList();

	static VList GetErrorMessageList(VRetCode nErrorCode);

protected:
	TpExceptionCore* m_core;

	string m_createMessage(int nErrCode, string file, int line, string sParam1 = "", string sParam2 = "");

	static void ms_createMessage(char* pBuf, int nErrCode, const char* p1, const char* p2);
};

/*
class TpException {
public:
	TpException();
	TpException(int nErrCode, VList errMessages, string file, int line);
	TpException(int nErrCode, string file, int line);
	TpException(int nErrCode, string sParam, string file, int line);
	TpException(int nErrCode, string sParam1, string sParam2, string file, int line);
	TpException(const TpException& src);

	virtual ~TpException();

	TpException& operator =(const TpException& src);

	VRetCode GetErrorCode();

	string GetDetailErrorMessage(int indent);
	
	VList GetErrorMessageList();

	static VList GetErrorMessageList(VRetCode nErrorCode);

protected:
	int m_nErrCode;
	string m_sParam1;
	string m_sParam2;
	string m_file;
	int m_line;
};
*/