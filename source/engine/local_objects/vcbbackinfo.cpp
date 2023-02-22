#include "../local_objects/vcbbackinfo.h"
#include "../local_objects/vcbbackinfo_core.h"
#include "../local_objects/vcbbackslot.h"
#include "../local_objects/vcbbackslot_core.h"
#include "../api_objects/vmodule_core.h"
#include "../api_objects/vtensor.h"
#include "../utils/vutils.h"

VCbBackInfo::VCbBackInfo() {
    m_core = NULL;
}

VCbBackInfo::VCbBackInfo(VSession session, string sBuiltin, VDict kwArgs) {
    m_core = new VCbBackInfoCore(session, sBuiltin, kwArgs);
}

VCbBackInfo::VCbBackInfo(const VCbBackInfo& src) {
    m_core = src.m_core->clone();
}

VCbBackInfo::VCbBackInfo(VCbBackInfoCore* core) {
    m_core = core->clone();
}

VCbBackInfo::~VCbBackInfo() {
    m_core->destroy();
}

VCbBackInfo& VCbBackInfo::operator =(const VCbBackInfo& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VCbBackInfoCore* VCbBackInfo::getClone() {
    return (VCbBackInfoCore*)m_core->clone_core();
}

VCbBackInfoCore* VCbBackInfo::getCore() {
    return m_core;
}

void VCbBackInfo::destroyCore() {
    if (m_core->getRefCnt() > 1) m_core->destroy();
    else {
        m_core->destroy();
        m_core = NULL;
    }
}

VSession VCbBackInfo::session() const {
    return m_core->m_session;
}

bool VCbBackInfo::isValid() {
    return m_core != NULL;
}

int VCbBackInfo::getRefCnt() {
    return m_core->getRefCnt();
}

int VCbBackInfo::getNth() {
    return m_core->getNth();
}

VCbBackInfoCore::VCbBackInfoCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::CbBackInfo) {
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;
    m_session = session,
    m_setup();
}

void VCbBackInfo::addDeviceConvInfo(int nHostId, VTensor tensor) {
    m_core->m_devConvParams[nHostId] = tensor;
}

void VCbBackInfo::addCbRequestSlot(
        VModuleCore* pStarter, VCbItem item, string name, int nDevice, bool bPre,
        VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params) {
	if (m_core == NULL) VP_THROW(VERR_INVALID_CORE);
    if (!bPre) m_core->m_addCbRequestSlot(pStarter, item, name, nDevice, xs, ys, sideTerms, params);
}

void VCbBackInfoCore::m_setup() {
}

void VCbBackInfoCore::m_addCbRequestSlot(
        VModuleCore* pStarter, VCbItem item, string name, int nDevice,
        VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params) {
    VCbBackSlot slot(m_session, "");

    slot.addCbRequestSlot(pStarter, item, name, nDevice, xs, ys, sideTerms, params, m_devConvParams);

    m_session.registCallbackSlot(slot);
}
