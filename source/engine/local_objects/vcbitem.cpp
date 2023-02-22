#include "../local_objects/vcbitem.h"
#include "../local_objects/vcbitem_core.h"

VCbItem::VCbItem() {
    m_core = NULL;
}

VCbItem::VCbItem(VSession session, string sBuiltin, VDict kwArgs) {
    m_core = new VCbItemCore(session, sBuiltin, kwArgs);
}

VCbItem::VCbItem(const VCbItem& src) {
    m_core = src.m_core->clone();
}

VCbItem::VCbItem(VCbItemCore* core) {
    m_core = core->clone();
}

VCbItem::~VCbItem() {
    m_core->destroy();
}

VCbItem& VCbItem::operator =(const VCbItem& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VCbItemCore* VCbItem::getClone() {
    return (VCbItemCore*)m_core->clone_core();
}

VCbItemCore* VCbItem::getCore() {
    return m_core;
}

void VCbItem::destroyCore() {
    if (m_core->getRefCnt() > 1) m_core->destroy();
    else {
        m_core->destroy();
        m_core = NULL;
    }
}

VSession VCbItem::session() const {
    return m_core->m_session;
}

bool VCbItem::isValid() {
    return m_core != NULL;
}

int VCbItem::getRefCnt() {
    return m_core->getRefCnt();
}

int VCbItem::getNth() {
    return m_core->getNth();
}

VCbItemCore::VCbItemCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::CbItem) {
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;
    m_session = session,
        m_setup();
}

VCbItem::VCbItem(VSession session, void* pCbFunc, void* pCbClose, VDict filters, VDict instInfo) {
    m_core = new VCbItemCore(session, "");

    m_core->m_pCbFunc = pCbFunc;
    m_core->m_pCbClose = pCbClose;
    m_core->m_filters = filters;
    m_core->m_instInfo = instInfo;
}

VCbItem::VCbItem(VSession session, void* pCbFunc, void* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict) {
    m_core = new VCbItemCore(session, "");

    m_core->m_pCbFunc = pCbFunc;
    m_core->m_pCbClose = pCbClose;
    m_core->m_instInfo = instInfo;
    m_core->m_statusInfo = statusInfo;
    m_core->m_tensorDict = tensorDict;
    m_core->m_gradDict = gradDict;
}

void* VCbItem::getCbFunc() {
    return m_core->m_pCbFunc;
}

void* VCbItem::getCbClose() {
    return m_core->m_pCbClose;
}

VDict VCbItem::getFilters() {
    return m_core->m_filters;
}

VDict VCbItem::getInstInfo() {
    return m_core->m_instInfo;
}

VDict VCbItem::getStatusInfo() {
    return m_core->m_statusInfo;
}

VDict VCbItem::getTensorDict() {
    return m_core->m_tensorDict;
}
VDict VCbItem::getGradDict() {
    return m_core->m_gradDict;
}

void VCbItemCore::m_setup() {
}
