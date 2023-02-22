#include "../objects/tp_metric.h"
#include "../objects/tp_metric_core.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_nn.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

EMetric::EMetric() { m_core = NULL; }
EMetric::EMetric(ENN nn) { m_core = new EMetricCore(nn, 0); }
EMetric::EMetric(ENN nn, VHMetric hMetric) { m_core = new EMetricCore(nn, hMetric); }
EMetric::EMetric(const EMetric& src) { m_core = src.m_core->clone(); }
EMetric::EMetric(EMetricCore* core) { m_core = core->clone(); }
EMetric::~EMetric() { m_core->destroy(); }
EMetric& EMetric::operator =(const EMetric& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone(); }
    return *this; }
EMetric::operator VHMetric() { return m_core->m_hEngineHandle; }
bool EMetric::isValid() { return m_core != NULL; }
void EMetric::close() { if (this) m_core->destroy(); }
ENN EMetric::nn() { return m_core ? m_core->m_nn : ENN(); }
EMetricCore* EMetric::getCore() { return m_core; }
EMetricCore* EMetric::cloneCore() { return (EMetricCore*) m_core->clone(); }
int EMetric::meNth() { return m_core->getNth(); }
int EMetric::meRefCnt() { return m_core->getRefCnt(); }

EMetricCore::EMetricCore(ENN nn, VHMetric hMetric) : EcoObjCore(VObjType::custom) {
    m_nn = nn;
    m_hEngineHandle = hMetric;
    m_setup(); }
EMetricCore::~EMetricCore() {
    m_delete();
    m_nn.getApiConn()->Metric_close(m_hEngineHandle, __FILE__, __LINE__);
 }
EMetricCore* EMetric::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

EMetric::EMetric(ENN nn, VHMetric hMetric, EMetricDict inferences) {
    m_core = new EMetricCore(nn, hMetric);
    m_core->m_inferences = inferences;
}

ETensor EMetric::__call__(ETensor pred) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    ETensorDict preds = { {"pred", pred} };

    ETensorDict tensors = evaluate(preds);

    return tensors.begin()->second;
}

ETensorDict EMetric::__call__(ETensorDict preds) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return evaluate(preds);
}

void EMetric::m_setExpressions(VDict inferenceTerms, VDict subTerms) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDict kwArgs;
    kwArgs["terms"] = inferenceTerms;
    kwArgs["subTerms"] = subTerms;
    VHMetric hMetric = nn().getApiConn()->Metric_create("custom", kwArgs, __FILE__, __LINE__);
    m_core = new EMetricCore(nn(), hMetric);
}

ETensorDict EMetric::evaluate(ETensorDict preds) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDict pHandles = TpUtils::TensorDictToDict(preds, false);

    VDict infHandles = nn().getApiConn()->Metric_evaluate(m_core->m_hEngineHandle, pHandles, __FILE__, __LINE__);

    ETensorDict infTensors;

    for (auto& it : infHandles) {
        ETensor tensor(nn(), it.second, true, false);
        tensor.downloadData();
        infTensors[it.first] = tensor;
    }

    //m_core->m_result = inferenceTensors;

    return infTensors;
}

//-----------------------------------------------------------------------------------------------------
// Core part

void EMetricCore::m_setup() {
}

void EMetricCore::m_delete() {
}

