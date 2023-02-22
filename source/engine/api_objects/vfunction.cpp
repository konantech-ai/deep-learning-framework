#include "../api_objects/vfunction.h"
#include "../api_objects/vfunction_core.h"
#include "../api_objects/vtensor.h"

int VFunctionCore::ms_nCheckCode = 41222015;

VStrList VFunction::ms_builtin = { "user_defined" };

//=========== API Object Common Part Start =======================

VFunction::VFunction() {
	m_core = NULL;
}

VFunction::VFunction(const VFunction& src) {
	m_core = src.m_core->clone();
}

VFunction::VFunction(VFunctionCore* core) {
	m_core = core->clone();
}

VFunction::VFunction(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VFunctionCore(session, sBuiltin, kwArgs);
}

VFunction::VFunction(VSession session, VHFunction handle) {
	m_core = NULL;
	VFunctionCore* core = (VFunctionCore*)handle;
	if (core == NULL) VP_THROW1(VERR_INVALID_CORE, "Function");
	if (core->m_nCheckCode != VFunctionCore::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, "Function");
	if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, "Function");
	m_core = (VFunctionCore*)core->clone_core();
}

VFunction::~VFunction() { m_core->destroy(); }

VFunction& VFunction::operator =(const VFunction& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VHFunction VFunction::cloneCore() {
	return (VHFunction)m_core->clone();
}

VHFunction VFunction::cloneHandle() {
	return (VHFunction)m_core->clone_handle();
}

VFunctionCore* VFunction::getClone() {
	return (VFunctionCore*)m_core->clone_core();
}

VFunctionCore* VFunction::getCore() {
	return m_core;
}

bool VFunction::isValid() {
	return m_core != NULL; 
}
void VFunction::closeHandle() {
	if (this) m_core->destroy_handle();
}

VSession VFunction::session() {
	return m_core->m_session;
}

int VFunction::getRefCnt() {
	return m_core->getRefCnt();
}

int VFunction::getNth() {
	return m_core->getNth();
}

void VFunction::incRefCount() {
	m_core->incRefCnt();
}

VFunctionCore::VFunctionCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Function) {
	m_nCheckCode = ms_nCheckCode;
	m_session = session;
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_onCreate();
}

VFunctionCore::~VFunctionCore() {
	m_onDelete();
	m_nCheckCode = 0;
}

//=========== API Object Common Part End =======================

VFunction::VFunction(VSession session, string sBuiltin, string sName, void* pCbAux, VDict kwArgs) {
	m_core = new VFunctionCore(session, sBuiltin, kwArgs);
	m_core->m_sName = sName;
	m_core->m_pCbAux = pCbAux;

	if (sBuiltin == "user_defined") {
		if (sName == "") VP_THROW(VERR_INVALID_FUNCTION_NAME);
		session.registUserDefinedFunction(sName, *this);
	}
}

VTensor VFunction::forward(int nInst, VTensorList operands, VDict opArgs) {
	VFuncCbHandlerInfo& cbInfo = session().getFunctionCbHandlerInfo();

	VCbForwardFunction* pForward = cbInfo.m_pFuncCbForward;
	VCbClose* pCbClose = cbInfo.m_pFuncCbClose;
	void* pHandlerCbAux = cbInfo.m_pFuncCbAux;

	if (pForward == NULL) VP_THROW(VERR_INVALID_FUNCTION_FORWARD);
	if (pCbClose == NULL) VP_THROW(VERR_INVALID_FUNCTION_CALLBACK_CLOSE);

	extern VTensor V_invokeFuncForwardCallback(VSession, void*, VHFunction, int, VCbForwardFunction*, VCbClose*, VTensorList, VDict);

	VTensor result = V_invokeFuncForwardCallback(session(), pHandlerCbAux, (VHandle)m_core, nInst, pForward, pCbClose, operands, opArgs);

	return result;
}

VTensor VFunction::backward(int nInst, VTensor ygrad, int nth, VTensorList operands, VDict opArgs) {
	VFuncCbHandlerInfo& cbInfo = session().getFunctionCbHandlerInfo();

	VCbBackwardFunction* pBackward = cbInfo.m_pFuncCbBackward;
	VCbClose* pCbClose = cbInfo.m_pFuncCbClose;
	void* pHandlerCbAux = cbInfo.m_pFuncCbAux;

	if (pBackward == NULL) VP_THROW(VERR_INVALID_FUNCTION_BACKWARD);
	if (pCbClose == NULL) VP_THROW(VERR_INVALID_FUNCTION_CALLBACK_CLOSE);

	extern VTensor V_invokeFuncBackwardCallback(VSession, void*, VHFunction, int, VCbBackwardFunction*, VCbClose*, VTensor, int, VTensorList, VDict);

	VTensor result = V_invokeFuncBackwardCallback(session(), pHandlerCbAux, (VHandle)m_core, nInst, pBackward, pCbClose, ygrad, nth, operands, opArgs);

	return result;
}

VList VFunction::GetBuiltinNames() {
	VList list;
	for (auto& it : ms_builtin) list.push_back(it);
	return list;
}

void VFunctionCore::m_onCreate() {
	m_pCbAux = NULL;
}

void VFunctionCore::m_onDelete() {}
