#include "../objects/tp_parameters.h"
#include "../objects/tp_parameters_core.h"
#include "../objects/tp_nn.h"
#include "../objects/tp_tensor.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"

EParameters::EParameters() { m_core = NULL; }
EParameters::EParameters(ENN nn) { m_core = new EParametersCore(nn, 0); }
EParameters::EParameters(ENN nn, VHParameters hParameters) { m_core = new EParametersCore(nn, hParameters); }
EParameters::EParameters(const EParameters& src) { m_core = src.m_core->clone(); }
EParameters::EParameters(EParametersCore* core) { m_core = core->clone(); }
EParameters::~EParameters() { m_core->destroy(); }
EParameters& EParameters::operator =(const EParameters& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}
EParameters::operator VHParameters() { return m_core->m_hEngineHandle; }
bool EParameters::isValid() { return m_core != NULL; }
void EParameters::close() { if (this) m_core->destroy(); }
ENN EParameters::nn() { return m_core ? m_core->m_nn : ENN(); }
EParametersCore* EParameters::getCore() { return m_core; }
EParametersCore* EParameters::cloneCore() { return (EParametersCore*) m_core->clone(); }
int EParameters::meNth() { return m_core->getNth(); }
int EParameters::meRefCnt() { return m_core->getRefCnt(); }

EParametersCore::EParametersCore(ENN nn, VHParameters hParameters) : EcoObjCore(VObjType::custom) {
    m_nn = nn;
    m_hEngineHandle = hParameters;
    m_setup();
}
EParametersCore::~EParametersCore() {
    m_delete();
    m_nn.getApiConn()->Parameters_close(m_hEngineHandle, __FILE__, __LINE__);
}
EParametersCore* EParameters::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

/*
void EParameters::add(EParameters params) {
    //TP_THROW(KERR_UNIMPLEMENTED_YET);
}
*/

void EParameters::zero_grad() {
    nn().getApiConn()->Parameters_zeroGrad(m_core->m_hEngineHandle, __FILE__, __LINE__);
}

void EParameters::initWeights() {
    nn().getApiConn()->Parameters_initWeights(m_core->m_hEngineHandle, __FILE__, __LINE__);
}

VList EParameters::weightList(ETensorDict& tensors) {
    VList terms;
    VDict handles;

    nn().getApiConn()->Parameters_getWeights(m_core->m_hEngineHandle, false, terms, handles, __FILE__, __LINE__);
    tensors = TpUtils::DictToTensorDict(nn(), handles);

    return terms;
}

VList EParameters::gradientList(ETensorDict& tensors) {
    VList terms;
    VDict handles;

    nn().getApiConn()->Parameters_getWeights(m_core->m_hEngineHandle, true, terms, handles, __FILE__, __LINE__);
    tensors = TpUtils::DictToTensorDict(nn(), handles);

    return terms;
}

ETensorDict EParameters::weightDict() {
    VList terms;
    VDict handles;

    nn().getApiConn()->Parameters_getWeights(m_core->m_hEngineHandle, false, terms, handles, __FILE__, __LINE__);
    ETensorDict tensors = TpUtils::DictToTensorDict(nn(), handles);

    return tensors;
}

ETensorDict EParameters::gradientDict() {
    VList terms;
    VDict handles;

    nn().getApiConn()->Parameters_getWeights(m_core->m_hEngineHandle, true, terms, handles, __FILE__, __LINE__);
    ETensorDict tensors = TpUtils::DictToTensorDict(nn(), handles);

    return tensors;
}

//-----------------------------------------------------------------------------------------------------
// Core part

void EParametersCore::m_setup() {
}

void EParametersCore::m_delete() {
}

