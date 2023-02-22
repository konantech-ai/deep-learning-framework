#include "../objects/tp_loss.h"
#include "../objects/tp_loss_core.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_nn.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

ELoss::ELoss() { m_core = NULL; }
ELoss::ELoss(ENN nn) { m_core = new ELossCore(nn, 0); }
ELoss::ELoss(ENN nn, VHLoss hELoss) { m_core = new ELossCore(nn, hELoss); }
ELoss::ELoss(const ELoss& src) { m_core = src.m_core->clone(); }
ELoss::ELoss(ELossCore* core) { m_core = core->clone(); }
ELoss::~ELoss() { m_core->destroy(); }
ELoss& ELoss::operator =(const ELoss& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}
ELoss::operator VHLoss() { return m_core->m_hEngineHandle; }
bool ELoss::isValid() { return m_core != NULL; }
void ELoss::close() { if (this) m_core->destroy(); }
ENN ELoss::nn() { return m_core ? m_core->m_nn : ENN(); }
ELossCore* ELoss::getCore() { return m_core; }
ELossCore* ELoss::cloneCore() { return (ELossCore*) m_core->clone(); }
int ELoss::meNth() { return m_core->getNth(); }
int ELoss::meRefCnt() { return m_core->getRefCnt(); }

ELossCore::ELossCore(ENN nn, VHLoss hELoss) : EcoObjCore(VObjType::custom) {
    m_nn = nn;
    m_hEngineHandle = hELoss;
    m_setup();
}
ELossCore::~ELossCore() {
    m_delete();
    m_nn.getApiConn()->Loss_close(m_hEngineHandle, __FILE__, __LINE__);
}
ELossCore* ELoss::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

ELoss::ELoss(ENN nn, VHLoss hLoss, ELossDict losses) {
    m_core = new ELossCore(nn, hLoss);
    m_core->m_losses = losses;
}

ETensor ELoss::__call__(ETensor pred, ETensor y, bool download_all) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    ETensorDict preds = { {"pred", pred} };
    ETensorDict ys = { {"y", y} };

    ETensorDict tensors = evaluate(preds, ys, download_all);

    return tensors.begin()->second;
}

ETensorDict ELoss::__call__(ETensorDict preds, ETensorDict ys, bool download_all) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return evaluate(preds, ys, download_all);
}

void ELoss::m_setExpressions(VDict lossTerms, VDict subTerms) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDict kwArgs;
    kwArgs["terms"] = lossTerms;
    kwArgs["subTerms"] = subTerms;
    VHLoss hLoss = nn().getApiConn()->Loss_create("custom", kwArgs, __FILE__, __LINE__);
    m_core = new ELossCore(nn(), hLoss);
}

ETensorDict ELoss::evaluate(ETensorDict preds, ETensorDict ys, bool download_all) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDict pHandles = TpUtils::TensorDictToDict(preds, false);
    VDict yHandles = TpUtils::TensorDictToDict(ys, true);

    VDict lossHandles = nn().getApiConn()->Loss_evaluate(m_core->m_hEngineHandle, download_all, pHandles, yHandles, __FILE__, __LINE__);

    ETensorDict lossTensors;

    for (auto& it : lossHandles) {
        lossTensors[it.first] = ETensor(nn(), it.second, true, false);
    }

    //m_core->m_result = lossTensors;

    return lossTensors;
}

ETensor ELoss::eval_accuracy(ETensor pred, ETensor y, bool download_all) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    ETensorDict preds = { {"pred", pred} };
    ETensorDict ys = { {"y", y} };

    ETensorDict tensors = eval_accuracy(preds, ys);

    return tensors.begin()->second;
}

ETensorDict ELoss::eval_accuracy(ETensorDict preds, ETensorDict ys, bool download_all) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDict pHandles = TpUtils::TensorDictToDict(preds, false);
    VDict yHandles = TpUtils::TensorDictToDict(ys, true);

    VDict accHandles = nn().getApiConn()->Loss_eval_accuracy(m_core->m_hEngineHandle, download_all, pHandles, yHandles, __FILE__, __LINE__);

    ETensorDict accTensors;

    for (auto& it : accHandles) {
        accTensors[it.first] = ETensor(nn(), it.second, true, false);
    }

    return accTensors;
}

void ELoss::backward() {
    //VDict lossHandles = TpUtils::TensorDictToDict(m_core->m_result, false);
    //nn().getApiConn()->Loss_backward(m_core->m_hEngineHandle, lossHandles, __FILE__, __LINE__);
    nn().getApiConn()->Loss_backward(m_core->m_hEngineHandle, __FILE__, __LINE__);
}

//-----------------------------------------------------------------------------------------------------
// Core part

void ELossCore::m_setup() {
}

void ELossCore::m_delete() {
}

