#include "../utils/tp_eco_exception_info.h"
#include "../utils/tp_utils.h"

bool TpEcoExceptionInfo::m_bForDevelopper = true;

TpEcoExceptionInfo::TpEcoExceptionInfo(TpException& ex, string func, string cinfo) {
	m_core = new TpEcoExceptionInfoCore();

	m_core->m_nErrCode = ex.GetErrorCode();
	m_core->m_msgStack = ex.GetErrorMessageList();

	string msg = cinfo + "::" + func + "() 호출에서 오류가 발생하였습니다.";

	m_core->m_msgStack.push_back(msg);
}

TpEcoExceptionInfo::~TpEcoExceptionInfo() {
	m_core->destroy();
}

TpEcoExceptionInfo::TpEcoExceptionInfo(const TpEcoExceptionInfo& src) {
	m_core = src.m_core->clone();
}

TpEcoExceptionInfo& TpEcoExceptionInfo::operator =(const TpEcoExceptionInfo& src) {
	if (this != &src) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}

	return *this;
}

int TpEcoExceptionInfo::get_error_code() const {
	return m_core->m_nErrCode;
}

string TpEcoExceptionInfo::get_error_message(int indent) const {
	string delimeter = std::string(indent, ' ');

	if (m_bForDevelopper) {
		return delimeter + TpUtils::join(m_core->m_msgStack, "\n" + delimeter);
	}
	else {
		int64 count = m_core->m_msgStack.size();
		return delimeter + (string)m_core->m_msgStack[0] + "\n" + delimeter + (string)m_core->m_msgStack[count-1];
	}
}

bool TpEcoExceptionInfo::getDevelopperMode() {
	return m_bForDevelopper;
}

void TpEcoExceptionInfo::setDevelopperMode(bool bForDevelopper) {
	m_bForDevelopper = bForDevelopper;
}
