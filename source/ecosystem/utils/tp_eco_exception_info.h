#pragma once

#include "../utils/tp_common.h"
#include "../utils/tp_exception.h"

class TpEcoExceptionInfoCore : VObjCore {
protected:
	TpEcoExceptionInfoCore() : VObjCore(VObjType::exception) {
		m_nErrCode = 0;
		m_nRefCount = 1;
	}

	void destroy() {
		if (--m_nRefCount <= 0) delete this;
	}

	TpEcoExceptionInfoCore* clone() {
		m_nRefCount++;
		return this;
	}

	friend class TpEcoExceptionInfo;

	int m_nErrCode;
	int m_nRefCount;
	VList m_msgStack;
};

class TpEcoExceptionInfo {
public:
	TpEcoExceptionInfo(TpException& ex, string func, string cinfo);
	TpEcoExceptionInfo(const TpEcoExceptionInfo& src);

	virtual ~TpEcoExceptionInfo();

	TpEcoExceptionInfo& operator =(const TpEcoExceptionInfo& src);

	int get_error_code() const;
	string get_error_message(int indent) const;

	static bool getDevelopperMode();
	static void setDevelopperMode(bool bForDevelopper);

protected:
	TpEcoExceptionInfoCore* m_core;

	static bool m_bForDevelopper;
};
