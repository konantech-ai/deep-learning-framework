#include "../objects/tp_scalar.h"
#include "../objects/tp_scalar_core.h"
#include "../objects/tp_nn.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

EScalar::EScalar() { m_core = NULL; }
EScalar::EScalar(ENN nn) { m_core = new EScalarCore(nn); }
EScalar::EScalar(const EScalar& src) { m_core = src.m_core->clone(); }
EScalar::EScalar(EScalarCore* core) { m_core = core->clone(); }
EScalar::~EScalar() { m_core->destroy(); }
EScalar& EScalar::operator =(const EScalar& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}
bool EScalar::isValid() { return m_core != NULL; }
void EScalar::close() { if (this) m_core->destroy(); }
ENN EScalar::nn() { return m_core ? m_core->m_nn : ENN(); }
EScalarCore* EScalar::getCore() { return m_core; }
EScalarCore::EScalarCore(ENN nn) : VObjCore(VObjType::custom) { m_nn = nn; m_setup(); }
EScalarCore* EScalar::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

EScalar::EScalar(ENN nn, ETensor tensor, float value) {
    m_core = new EScalarCore(nn);
    m_core->m_value = value;
    m_core->m_srcTensor = tensor;
}

EScalar::operator float() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_value;
}

//-----------------------------------------------------------------------------------------------------
// Core part

void EScalarCore::m_setup() {
}
