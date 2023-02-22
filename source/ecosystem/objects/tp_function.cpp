#include "../objects/tp_function.h"
#include "../objects/tp_function_core.h"
#include "../objects/tp_nn.h"
#include "../objects/tp_tensor.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

EFunction::EFunction() { m_core = NULL; }
EFunction::EFunction(ENN nn) { m_core = new EFunctionCore(nn, 0); }
EFunction::EFunction(ENN nn, VHFunction hFunction) { m_core = new EFunctionCore(nn, hFunction); }
EFunction::EFunction(const EFunction& src) { m_core = src.m_core->clone(); }
EFunction::EFunction(EFunctionCore* core) { m_core = core->clone(); }
EFunction::~EFunction() { m_core->destroy(); }
EFunction& EFunction::operator =(const EFunction& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone(); }
    return *this; }
EFunction::operator VHFunction() { return m_core->m_hEngineHandle; }
bool EFunction::isValid() { return m_core != NULL; }
void EFunction::close() { if (this) m_core->destroy(); }
ENN EFunction::nn() { return m_core ? m_core->m_nn : ENN(); }
EFunctionCore* EFunction::getCore() { return m_core; }
EFunctionCore* EFunction::cloneCore() { return (EFunctionCore*) m_core->clone(); }
int EFunction::meNth() { return m_core->getNth(); }
int EFunction::meRefCnt() { return m_core->getRefCnt(); }
EFunctionCore::EFunctionCore(ENN nn, VHFunction hFunction) : EcoObjCore(VObjType::custom) {
    m_nn = nn;
    m_hEngineHandle = hFunction;
    m_setup(); }
EFunctionCore::~EFunctionCore() {
    m_delete();
    m_nn.getApiConn()->Function_close(m_hEngineHandle, __FILE__, __LINE__);
 }
EFunctionCore* EFunction::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

EFunction::EFunction(ENN nn, string sBuiltin, string sName, VDict kwArgs) {
    sBuiltin = TpUtils::tolower(sBuiltin);

    if (!nn.isInBuiltinName("function", sBuiltin)) {
        TP_THROW(VERR_INVALID_BUILTIN_FUNCTION);
    }

    VHFunction hFunction = 0;

    m_core = new EFunctionCore(nn, 0);

    if (nn.isInBuiltinName("function", sBuiltin)) {
        void* pCbAux = (void*)(VObjCore*)m_core;
        hFunction = nn.getApiConn()->Function_create(sBuiltin, sName, pCbAux, kwArgs, __FILE__, __LINE__);
    }

    m_core->m_hEngineHandle = hFunction;
    m_core->m_name = sName;
}

string EFunction::getInstName() {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

void EFunction::registUserDefFunc(EFunction* pInst) {
    nn().registUserDefFunc(m_core->m_hEngineHandle, pInst);
}

ETensor EFunction::forward(int nInst, ETensor x, VDict opArgs) {
    return forward(nInst, ETensorList{ x }, opArgs);
}

ETensor EFunction::forward(int nInst, ETensorList operands, VDict opArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor EFunction::backward(int nInst, ETensor ygrad, ETensor x, VDict opArgs) {
    return backward(nInst, ygrad, 0, ETensorList{ x }, opArgs);
}

ETensor EFunction::backward(int nInst, ETensor ygrad, int nth, ETensorList operands, VDict opArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

/*
ETensor EFunction::evaluate(ETensor x, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);
    TP_THROW(VERR_UNDEFINED);
}
*/

//-----------------------------------------------------------------------------------------------------
// Core part

void EFunctionCore::m_setup() {
}

void EFunctionCore::m_delete() {
}
