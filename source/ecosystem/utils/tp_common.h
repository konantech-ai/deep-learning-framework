#pragma once

#define TERR_API_CALL			50000
#define TERR_EXEC_TRANSACTION	50001
#define TERR_API_PYTHON_CALL	60000

#define TP_THROW(x) {  throw TpException(x, __FILE__, __LINE__); }
#define TP_THROW2(x,arg) {  throw TpException(x, arg, __FILE__, __LINE__); }
#define TP_THROW3(x,a1, a2) {  throw TpException(x, a1, a2, __FILE__, __LINE__); }

#define TP_MEMO(x, str)

#define TP_CALL_TEST(fname, ...) { \
		VRetCode ret = fname( __VA_ARGS__); \
		if (ret != VERR_OK) { \
	        const VExBuf* pMsgsExBuf; \
			V_Session_getLastErrorMessageList(m_hSession, &pMsgsExBuf); \
			VList sErrMessages = VListWrapper::unwrap(pMsgsExBuf); \
			V_Session_freeExchangeBuffer(m_hSession, pMsgsExBuf); \
			TpException ex(ret, sErrMessages); \
			throw TpException(TERR_API_CALL, #fname, ex, __FILE__, __LINE__); \
		} \
	}

#define TP_CALL(call_exp) { \
		VRetCode ret = call_exp; \
		if (ret != VERR_OK) { \
	        const VExBuf* pMsgsExBuf; \
			V_Session_getLastErrorMessageList(m_hSession, &pMsgsExBuf); \
			VList sErrMessages = VListWrapper::unwrap(pMsgsExBuf); \
			V_Session_freeExchangeBuffer(m_hSession, pMsgsExBuf); \
			TpException ex(ret, sErrMessages); \
			throw TpException(TERR_API_CALL, __func__, ex, __FILE__, __LINE__); \
		} \
	}

#define TP_CALL_STATIC(call_exp) { \
		VRetCode ret = call_exp; \
		if (ret != VERR_OK) throw TpException(ret, __FILE__, __LINE__); \
	}

#define TP_CALL_EX(estimate_err,call_exp) { \
		VRetCode ret = call_exp; \
		if (ret != estimate_err) { \
			/* vp_report_error(ret, file, line, __FILE__, __LINE__); */ \
			throw TpException(ret, __FILE__, __LINE__); \
		} \
	}

#define TP_CALL_DUMP(call_exp) { \
		try { \
			call_exp; \
		} \
		catch (TpException ex) { \
			print("TpException(%d) in %s:%d", ex.GetErrorCode(), __FILE__, __LINE__); \
			print("%s", ex.GetDetailErrorMessage(4).c_str()); \
		} \
	}

#include <stdio.h>

#include <algorithm>
#include <string>
#include <thread>

using namespace std;

#include "../../engine/include/vapi.h"

const string sEcoVersion = "0.0.1";

//extern void tp_report_error(int errCode, const char* file, int line);															// in tp_eco_conn.cpp
//extern void vp_report_error(VHSession hSession, int errCode, string file, int line, string file_conn, int line_conn);	// in tp_eco_conn.cpp

//#include "../../../kai_engine/src/include/kai_api.h"
//#include "../../../kai_engine/src/include/kai_types.h"
//#include "../../../kai_engine/src/include/kai_errors.h"

inline string& ltrim(string& s, const char* t = " \t\r\f\v")
{
	s.erase(0, s.find_first_not_of(t));
	return s;
}

inline string& rtrim(string& s, const char* t = " \t\r\f\v")
{
	s.erase(s.find_last_not_of(t) + 1);
	return s;
}

inline string& trim(string& s, const char* t = " \t\r\f\v")
{
	return ltrim(rtrim(s, t), t);
}

inline void print(const char* fmt, ...) {
	va_list arg_ptr;

	va_start(arg_ptr, fmt);
	vprintf(fmt, arg_ptr);
	va_end(arg_ptr);

	printf("\n");
}

enum class EModuleType { layer, network, model, macro, custom };

class EcoObjCore : public VObjCore {
protected:
	friend class ENN;
	VHandle getEngineHandle() { return m_hEngineHandle; }

protected:
	EcoObjCore(VObjType type) : VObjCore(type) {
		m_hEngineHandle = 0;
	}
	VHandle m_hEngineHandle;
};

class ENN;

class EModule;
class EScalar;
class ETensor;
class ETensorData;
class EParam;
class EParameters;
class ELoss;
class EMetric;
class EMath;
class EOptimizer;
class EFunction;

typedef vector<EModule> EModuleList;
typedef vector<ETensor> ETensorList;

typedef map<string, EModule> EModuleDict;
typedef map<string, ETensor> ETensorDict;
typedef map<string, ELoss> ELossDict;
typedef map<string, EMetric> EMetricDict;

typedef map<string, ETensorDict> ETensorDicts;

typedef map<int, ETensor> ETensorMap;

typedef VDict TCbForwardCallback(VDict instInfo, VDict statusInfo, ETensorDicts tensors);
typedef VDict TCbBackwardCallback(VDict instInfo, VDict statusInfo, ETensorDicts tensors, ETensorDicts grads);
