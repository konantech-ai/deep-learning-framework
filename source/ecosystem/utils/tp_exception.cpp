#include "../utils/tp_exception.h"

TpException::TpException() {
	m_core = new TpExceptionCore();
}

TpException::TpException(TpExceptionCore* core) {
	m_core = core->clone();
}

TpException::TpException(const TpException& src) {
	m_core = src.m_core->clone();
}

TpException::TpException(int nErrCode, VList messages) {
	m_core = new TpExceptionCore();

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack = messages;
}

TpException::TpException(int nErrCode, string sParam1, TpException ex, string file, int line) {
	m_core = ex.cloneCore();

	std::replace(file.begin(), file.end(), '\\', '/');

	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line, sParam1));
}

TpException::TpException(int nErrCode, string file, int line) {
	if (nErrCode == 0) return;

	m_core = new TpExceptionCore();

	std::replace(file.begin(), file.end(), '\\', '/');

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line));
}

TpException::TpException(int nErrCode, string sParam, string file, int line) {
	m_core = new TpExceptionCore();
	
	std::replace(file.begin(), file.end(), '\\', '/');

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line, sParam));
}

string TpException::GetDetailErrorMessage(int indent) {
	string prefix;
	for (int n = 0; n < indent; n++) prefix += " ";
	prefix += "=> ";

	string message;

	for (auto& it : m_core->m_msgStack) {
		message = prefix + (string)it + "" + message;
	}

	return message;
}

/*
TpException::TpException(int nErrCode, string sParam1, string sParam2, string file, int line) {
	m_core = new TpExceptionCore();

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line, sParam1, sParam2));
}

TpException::TpException(int nErrCode, const TpException& src, string file, int line) {
	m_core = src.m_core->clone();
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line));
}
*/

TpException::~TpException() {
	m_core->destroy();
}

TpException& TpException::operator =(const TpException& src) {
	if (&src != this) m_core = src.m_core->clone();
	return *this;
}

VRetCode TpException::GetErrorCode() {
	return m_core->m_nErrCode;
}

string TpException::m_createMessage(int nErrCode, string file, int line, string sParam1, string sParam2) {
	char buffer[1024];

	const char* p1 = (sParam1.length() > 0) ? sParam1.c_str() : "";
	const char* p2 = (sParam2.length() > 0) ? sParam2.c_str() : "";

	ms_createMessage(buffer, nErrCode, p1, p2);

	snprintf(buffer + strlen(buffer), 1024 - strlen(buffer), " in %s:%d", file.c_str(), line);

	return (string)buffer;
}

VList TpException::GetErrorMessageList() {
	if (m_core->m_nErrCode == 0) return VList();
	return m_core->m_msgStack;
}

VList TpException::GetErrorMessageList(VRetCode nErrorCode) {
	return VList({ "Sorry, error message for static code is not implemented yet..." });
}

void TpException::ms_createMessage(char* pBuf, int nErrCode, const char* p1, const char* p2) {
	switch (nErrCode) {
	case TERR_API_CALL:
		snprintf(pBuf, 1024, "API 함수 %s() 호출에서 예외가 발생하였습니다.", p1);
		break;
	case TERR_EXEC_TRANSACTION:
		snprintf(pBuf, 1024, "트랜잭션 %s 처리 과정에서 예외가 발생했습니다.", p1);
		break;
	case TERR_API_PYTHON_CALL:
		snprintf(pBuf, 1024, "파이썬 프론트엔드 인터페이스 함수 %s() 처리 과정에서 예외가 발생했습니다.", p1);
		break;
	case VERR_HOSTMEM_ALLOC_FAILURE:
		snprintf(pBuf, 1024, "용량 %s 바이트의 호스트 메모리 할당 요청이 거부되었습니다.", p1);
		break;
	default:
		snprintf(pBuf, 1024, "오류메시지가 준비되지 않았습니다.(code:%d,%s,%s)", nErrCode, p1, p2);
		break;
	}
}

/*
TpException::TpException() {
	m_nErrCode = 0;
}

TpException::TpException(int nErrCode, string file, int line) {
	m_nErrCode = nErrCode;
	if (nErrCode) {
		m_nErrCode = nErrCode;
		m_file = file;
		m_line = line;
	}
}

TpException::TpException(int nErrCode, string sParam, string file, int line) {
	m_nErrCode = nErrCode;
	m_sParam1 = sParam;
	m_file = file;
	m_line = line;
}

TpException::TpException(int nErrCode, string sParam1, string sParam2, string file, int line) {
	m_nErrCode = nErrCode;
	m_sParam1 = sParam1;
	m_sParam2 = sParam2;
	m_file = file;
	m_line = line;
}

TpException::TpException(const TpException& src) {
	m_nErrCode = src.m_nErrCode;
	m_sParam1 = src.m_sParam1;
	m_sParam2 = src.m_sParam2;
	m_file = src.m_file;
	m_line = src.m_line;
}

TpException::~TpException() {
}

TpException& TpException::operator =(const TpException& src) {
	if (&src != this) {
		m_nErrCode = src.m_nErrCode;
		m_sParam1 = src.m_sParam1;
		m_sParam2 = src.m_sParam2;
		m_file = src.m_file;
		m_line = src.m_line;
	}
	return *this;
}

VRetCode TpException::GetErrorCode() {
	return m_nErrCode;
}

string TpException::GetErrorMessage() {
	char buffer[1024];

	const char* p1 = (m_sParam1.length() > 0) ? m_sParam1.c_str() : "_";
	const char* p2 = (m_sParam2.length() > 0) ? m_sParam2.c_str() : "_";

	snprintf(buffer, 1024, "TpError(code:%d,%s,%s) in %s:%d", m_nErrCode, p1, p2, m_file.c_str(), m_line);

	return (string)buffer;
}

VList TpException::GetErrorMessages(VRetCode nErrorCode) {
	return VList({ "Sorry, error message is not implemented yet..." });
}
*/