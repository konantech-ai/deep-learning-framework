#include "../local_objects/vudfitem.h"
#include "../local_objects/vudfitem_core.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vfunction.h"

VUDFItem::VUDFItem() {
    m_core = NULL;
}

VUDFItem::VUDFItem(VSession session, string sBuiltin, VDict kwArgs) {
    m_core = new VUDFItemCore(session, sBuiltin, kwArgs);
}

VUDFItem::VUDFItem(const VUDFItem& src) {
    m_core = src.m_core->clone();
}

VUDFItem::VUDFItem(VUDFItemCore* core) {
    m_core = core->clone();
}

VUDFItem::~VUDFItem() {
    m_core->destroy();
}

VUDFItem& VUDFItem::operator =(const VUDFItem& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VUDFItemCore* VUDFItem::getClone() {
    return (VUDFItemCore*)m_core->clone_core();
}

VUDFItemCore* VUDFItem::getCore() {
    return m_core;
}

void VUDFItem::destroyCore() {
    if (m_core->getRefCnt() > 1) m_core->destroy();
    else {
        m_core->destroy();
        m_core = NULL;
    }
}

VSession VUDFItem::session() const {
    return m_core->m_session;
}

bool VUDFItem::isValid() {
    return m_core != NULL;
}

int VUDFItem::getRefCnt() {
    return m_core->getRefCnt();
}

int VUDFItem::getNth() {
    return m_core->getNth();
}

VUDFItemCore::VUDFItemCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::UDFItem) {
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;
    m_session = session,
        m_setup();
}

VUDFItem::VUDFItem(VSession session, VFunctionCore* functor, VTensor y, VTensorList operands, VDict opArgs) {
    m_core = new VUDFItemCore(session, "");

    m_core->m_functor = functor;
    m_core->m_y = y;
    m_core->m_xs = operands;
    m_core->m_opArgs = opArgs;

    for (auto& it : operands) {
        m_core->m_xgrads.push_back(VTensor());
    }
}

void VUDFItem::setGrad(int nth, VTensor ygrad, VTensor xgrad) {
    if (!m_core->m_ygrad.isValid()) {
        m_core->m_ygrad = ygrad;
    }
    else if (ygrad.getNth() != m_core->m_ygrad.getNth()) {
        VP_THROW(VERR_UNDEFINED);
    }
    
    if (nth >= m_core->m_xgrads.size()) {
        VP_THROW(VERR_UNDEFINED);
    }

    if (m_core->m_xgrads[nth].isValid()) {
        VP_THROW(VERR_UNDEFINED);
    }

    m_core->m_xgrads[nth] = xgrad;
}

VFunctionCore* VUDFItem::getFunctor() {
    return m_core->m_functor;
}

VTensorList VUDFItem::getXs() {
    return m_core->m_xs;
}

VDict VUDFItem::getOpArgs() {
    return m_core->m_opArgs;
}

VTensor VUDFItem::getY() {
    return m_core->m_y;
}

VTensor VUDFItem::getYGrad() {
    return m_core->m_ygrad;
}

VTensorList VUDFItem::getXGrads() {
    return m_core->m_xgrads;
}

void VUDFItem::dump(string title) {
    return;

    printf("[VUDFItem: %s] #%d\n", title.c_str(), getNth());

    printf("    x#%d(ref:%d), y#%d(ref:%d)\n", m_core->m_xs[0].getNth(), m_core->m_xs[0].getRefCnt(), m_core->m_y.getNth(), m_core->m_y.getRefCnt());

    if (m_core->m_ygrad.isValid()) {
        printf("    ygrad#%d(ref:%d), xgrad#%d(ref:%d)\n", m_core->m_ygrad.getNth(), m_core->m_ygrad.getRefCnt(), m_core->m_xgrads[0].getNth(), m_core->m_xgrads[0].getRefCnt());
    }

    int nnn = 0;
}

int VUDFItemCore::ms_intCnt = 0;

void VUDFItemCore::m_setup() {
    if (0) {
        ms_intCnt++;
        printf("VUDFItemCore[%d] created: %d instances\n", getNth(), ms_intCnt);
    }
}

VUDFItemCore::~VUDFItemCore() {
    if (0) {
        ms_intCnt--;
        printf("VUDFItemCore[%d] destroyed: %d instances\n", getNth(), ms_intCnt);
    }
}
