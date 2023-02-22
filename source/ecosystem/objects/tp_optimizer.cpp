#include "../objects/tp_optimizer.h"
#include "../objects/tp_optimizer_core.h"
#include "../objects/tp_parameters.h"
#include "../objects/tp_nn.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

EOptimizer::EOptimizer() { m_core = NULL; }
EOptimizer::EOptimizer(ENN nn) { m_core = new EOptimizerCore(nn, 0); }
EOptimizer::EOptimizer(ENN nn, VHOptimizer hOptimizer) { m_core = new EOptimizerCore(nn, hOptimizer); }
EOptimizer::EOptimizer(const EOptimizer& src) { m_core = src.m_core->clone(); }
EOptimizer::EOptimizer(EOptimizerCore* core) { m_core = core->clone(); }
EOptimizer::~EOptimizer() { m_core->destroy(); }
EOptimizer& EOptimizer::operator =(const EOptimizer& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}
EOptimizer::operator VHOptimizer() { return (VHOptimizer)m_core->m_hEngineHandle; }
bool EOptimizer::isValid() { return m_core != NULL; }
void EOptimizer::close() { if (this) m_core->destroy(); }
ENN EOptimizer::nn() { return m_core ? m_core->m_nn : ENN(); }
EOptimizerCore* EOptimizer::getCore() { return m_core; }
EOptimizerCore* EOptimizer::cloneCore() { return (EOptimizerCore*) m_core->clone(); }
int EOptimizer::meNth() { return m_core->getNth(); }
int EOptimizer::meRefCnt() { return m_core->getRefCnt(); }

EOptimizerCore::EOptimizerCore(ENN nn, VHOptimizer hOptimizer) : EcoObjCore(VObjType::custom) {
	m_nn = nn;
	m_hEngineHandle = hOptimizer;
	m_setup();
}
EOptimizerCore::~EOptimizerCore() {
	m_delete();
	m_nn.getApiConn()->Optimizer_close(m_hEngineHandle, __FILE__, __LINE__);
}
EOptimizerCore* EOptimizer::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

void EOptimizer::setOption(VDict kwArgs) {
	if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
	nn().getApiConn()->Optimizer_set_option(m_core->m_hEngineHandle, kwArgs, __FILE__, __LINE__);
}

void EOptimizer::zero_grad() {
	if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
	m_core->m_parameters.zero_grad();
}

void EOptimizer::step() {
	if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
	nn().getApiConn()->Optimizer_step(m_core->m_hEngineHandle, __FILE__, __LINE__);
}

void EOptimizer::setup(string sBuiltin, VDict kwArgs, EParameters params) {
	m_core->m_sBuiltin = sBuiltin;
	m_core->m_kwArgs = kwArgs;
	m_core->m_parameters = params;
}
//-----------------------------------------------------------------------------------------------------
// Core part

void EOptimizerCore::m_setup() {
}

void EOptimizerCore::m_delete() {
}
