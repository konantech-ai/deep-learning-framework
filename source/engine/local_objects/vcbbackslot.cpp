#include "../local_objects/vcbbackslot.h"
#include "../local_objects/vcbbackslot_core.h"
#include "../local_objects/vcbitem.h"
#include "../local_objects/vcbitem_core.h"
#include "../api_objects/vmodule_core.h"
#include "../api_objects/vtensor.h"
#include "../utils/vutils.h"

VCbBackSlot::VCbBackSlot() {
    m_core = NULL;
}

VCbBackSlot::VCbBackSlot(VSession session, string sBuiltin, VDict kwArgs) {
    m_core = new VCbBackSlotCore(session, sBuiltin, kwArgs);
}

VCbBackSlot::VCbBackSlot(const VCbBackSlot& src) {
    m_core = src.m_core->clone();
}

VCbBackSlot::VCbBackSlot(VCbBackSlotCore* core) {
    m_core = core->clone();
}

VCbBackSlot::~VCbBackSlot() {
    m_core->destroy();
}

VCbBackSlot& VCbBackSlot::operator =(const VCbBackSlot& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VCbBackSlotCore* VCbBackSlot::getClone() {
    return (VCbBackSlotCore*)m_core->clone_core();
}

VCbBackSlotCore* VCbBackSlot::getCore() {
    return m_core;
}

void VCbBackSlot::destroyCore() {
    if (m_core->getRefCnt() > 1) m_core->destroy();
    else {
        m_core->destroy();
        m_core = NULL;
    }
}

VSession VCbBackSlot::session() const {
    return m_core->m_session;
}

bool VCbBackSlot::isValid() {
    return m_core != NULL;
}

int VCbBackSlot::getRefCnt() {
    return m_core->getRefCnt();
}

int VCbBackSlot::getNth() {
    return m_core->getNth();
}

VCbBackSlotCore::VCbBackSlotCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::CbBackSlot) {
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;
    m_session = session,
        m_setup();
}

void VCbBackSlot::addCbRequestSlot(
    VModuleCore* pStarter, VCbItem item, string name, int nDevice,
    VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VTensorMap devConvParams) {
    m_core->m_addCbRequestSlot(pStarter, item, name, nDevice, xs, ys, sideTerms, params, devConvParams);
}

void VCbBackSlot::close() {
    if (m_core == NULL) VP_THROW(VERR_INVALID_CORE);
    m_core->m_freeDependentTensors();
    //m_core->destroy();
}

void VCbBackSlot::fillAndInvokeOnFull(VTensor tensor, VTensor grad, VExecTracer tracer) {
    if (m_core == NULL) VP_THROW(VERR_INVALID_CORE);
    m_core->m_fillAndInvokeOnFull(tensor, grad, tracer);
}

void VCbBackSlot::replace(int nOldId, VTensor newTensor) {
    m_core->m_replace(nOldId, newTensor);
}

void VCbBackSlotCore::m_setup() {
}

void VCbBackSlotCore::m_freeDependentTensors() {
    for (int section = 0; section < 4; section++) {
        for (auto& it : m_tensors[section]) {
            if (it.second.needGrad()) {
                it.second.resetCbBackSlot(getNth());
            }
        }
        m_tensors[section].clear();
    }

}

void VCbBackSlotCore::m_fillAndInvokeOnFull(VTensor tensor, VTensor grad, VExecTracer tracer) {
    int tensorId = tensor.getNth();

    if (m_tensorMap.find(tensorId) == m_tensorMap.end()) VP_THROW(VERR_INVALID_MAP_KEY);

    int section = m_tensorMap[tensorId];

    if (section < 0) VP_THROW(VERR_UNDEFINED);

    bool found = false;

    for (auto& it : m_tensors[section]) {
        if (it.second.getNth() == tensor.getNth()) {
            m_gradients[section][it.first] = grad;
            found = true;
            break;
        }
    }

    if (!found) VP_THROW(VERR_UNDEFINED);

    tracer.addTensor(tensor);
    tracer.addTensor(grad);

    m_tensorMap[tensorId] = -1;

    if (section == 1) {
        if (--m_nUnresolvedOutputCount == 0) {
            m_invokeCallback(true, tracer);
        }

        if (m_nUnresolvedInputCount == 0) {
            m_invokeCallback(false, tracer);
        }
    }
    else if (section == 0) {
        if (--m_nUnresolvedInputCount == 0) {
            m_invokeCallback(false, tracer);
        }
    }
}

void VCbBackSlotCore::m_replace(int nOldId, VTensor newTensor) {
    for (int section = 0; section < 4; section++) {
        for (auto& it : m_tensors[section]) {
            if (it.second.getNth() == nOldId) {
                m_tensors[section][it.first] = newTensor;
                break;
            }
        }
    }

    m_tensorMap[newTensor.getNth()] = m_tensorMap[nOldId];
    m_tensorMap.erase(nOldId);
}

void VCbBackSlotCore::m_invokeCallback(bool bPre, VExecTracer tracer) {
    VDict instInfo = m_instInfo;

    VList names = { "input", "output", "sideterm", "param" };

    VDict tensorDict = vutils.toDictInternal(m_tensors, names);
    VDict gradDict = vutils.toDictInternal(m_gradients, names);

    VDict statusInfo = m_statusInfo;

    string phase = bPre ? "pre" : "post";

    if (m_cbFilters.find("phase") != m_cbFilters.end()) {
        VList phaseFilter = m_cbFilters["phase"];
        if (!phaseFilter.find_string(phase)) {
            return;
        }
    }

    statusInfo["phase"] = phase;
    statusInfo["direction"] = "backward";

    tracer.addInvokeBackwardCallback(m_pCbFunc, m_pCbClose, instInfo, statusInfo, tensorDict, gradDict);

    extern VDict V_invokeModuleBackwardCallback(VHSession hSession, VCbBackwardModule * pCbFunc, VCbClose * pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict paramDict);

    // 콜백 반값 result는 아직 특별한 용도가 없지만 차후 확장에 대비해 전달받을 수 있게 한다.
    VDict result = V_invokeModuleBackwardCallback(m_session, m_pCbFunc, m_pCbClose, instInfo, statusInfo, tensorDict, gradDict);

    // 억지로 일일이 삭제하지 않아도 Dict 반납시 최수 처리됨, 이중 회수로 문제 발생한 듯 하여 주석화함
    /*
    printf("PP3\n");
    vutils.freeDictInternal(tensorDict);
    printf("PP4\n");
    vutils.freeDictInternal(gradDict);
    printf("PP5\n");
    */
}

void VCbBackSlotCore::m_addCbRequestSlot(
        VModuleCore* pStarter, VCbItem item, string name, int nDevice,
        VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VTensorMap devConvParams) {
    VCbItemCore* pItemCore = item.getCore();

    m_pCbFunc = (VCbBackwardModule*)pItemCore->m_pCbFunc;
    m_pCbClose = (VCbClose*)pItemCore->m_pCbClose;

    m_instInfo = pItemCore->m_instInfo;
    m_instInfo["#data_idx"] = pStarter->getDataIdx(nDevice);

    m_statusInfo["name"] = name;
    m_statusInfo["device"] = nDevice;

    m_cbFilters = pItemCore->m_filters;  // pre, post 지점 각각에서 콜백 호출이 필요한 지 여부를 알려줌

    m_tensors[0] = xs;
    m_tensors[1] = ys;
    m_tensors[2] = sideTerms;

    VList names;
    VTensorDict paramDict;
    
    params.getWeights(names, paramDict, false);

    for (auto& it : paramDict) {
        int nTensorId = it.second.getNth();
        if (devConvParams.find(nTensorId) != devConvParams.end()) {
            m_tensors[3][it.first] = devConvParams[nTensorId];
        }
        else {
            m_tensors[3][it.first] = it.second;
        }
    }

    //m_nUnresolvedCount = 0;
    m_nUnresolvedInputCount = 0;
    m_nUnresolvedOutputCount = 0;

    VCbBackSlot slot(this);

    //for (int section = 0; section < 4; section++) {
    for (int section = 0; section < 2; section++) {
        for (auto& it : m_tensors[section]) {
            if (it.second.needGrad()) {
                it.second.setCbBackSlot(slot);

                int nTensorId = it.second.getNth();

                m_tensorMap[nTensorId] = section;
                
                if (section == 0) m_nUnresolvedInputCount++;
                else m_nUnresolvedOutputCount++;
            }
        }
    }
}

