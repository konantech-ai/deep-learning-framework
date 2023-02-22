#include "../local_objects/vexectracerpool.h"
#include "../local_objects/vexectracerpool_core.h"
#include "../local_objects/vexectracer.h"
#include "../api_objects/vtensor.h"

VExecTracerPool::VExecTracerPool() {
    m_core = NULL;
}

VExecTracerPool::VExecTracerPool(VSession session, string sBuiltin, VDict kwArgs) {
    m_core = new VExecTracerPoolCore(session, sBuiltin, kwArgs);
}

VExecTracerPool::VExecTracerPool(const VExecTracerPool& src) {
    m_core = src.m_core->clone();
}

VExecTracerPool::VExecTracerPool(VExecTracerPoolCore* core) {
    m_core = core->clone();
}

VExecTracerPool::~VExecTracerPool() {
    m_core->destroy();
}

VExecTracerPool& VExecTracerPool::operator =(const VExecTracerPool& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VExecTracerPoolCore* VExecTracerPool::getClone() {
    return (VExecTracerPoolCore*)m_core->clone_core();
}

VExecTracerPoolCore* VExecTracerPool::getCore() {
    return m_core;
}

void VExecTracerPool::destroyCore() {
    if (m_core->getRefCnt() > 1) m_core->destroy();
    else {
        m_core->destroy();
        m_core = NULL;
    }
}

VSession VExecTracerPool::session() const {
    return m_core->m_session;
}

bool VExecTracerPool::isValid() {
    return m_core != NULL;
}

int VExecTracerPool::getRefCnt() {
    return m_core->getRefCnt();
}

int VExecTracerPool::getNth() {
    return m_core->getNth();
}

VExecTracerPoolCore::VExecTracerPoolCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::ExecTracerPool) {
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;
    m_session = session,
        m_setup();
}

bool VExecTracerPool::openAndExecute(VTensorDict xs, int nMaxTracer, VTensorDict& preds, VExecTracer& tracer) {
	if (!session().getNoTracer()) {
		for (auto& it : m_core->m_tracers) {
			if (it.hasValidHistory(xs)) {
				if (session().isTracerDirty()) {
					it.reset();
					tracer = it;
					break;
				}
				else {
					preds = it.executeHistory();
					return true;
				}
			}
		}

		if (!tracer.isValid()) {
            if (m_core->m_tracers.size() >= nMaxTracer) {
                m_core->m_tracers.erase(m_core->m_tracers.begin());
			}
			tracer = VExecTracer(session(), "module_forward", {});
			m_core->m_tracers.push_back(tracer);
		}
	}

	tracer.setInput(xs);

	return false;
}

void VExecTracerPoolCore::m_setup() {
}
