#include <cuda_runtime.h>

#include "../api_objects/vmetric_core.h"
#include "../api_objects/vmetric.h"
#include "../local_objects/vgraph.h"
#include "../api_objects/vtensor.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vcbbackinfo.h"
#include "../support/vmath.h"
#include "../support/vback_queue.h"
#include "../utils/vutils.h"

int VMetricCore::ms_nCheckCode = 18920011;

VStrList VMetric::ms_builtin = { "formula", "custom", "multiple" };

//=========== API Object Common Part Start =======================

VMetric::VMetric() {
	m_core = NULL;
}

VMetric::VMetric(const VMetric& src) {
	m_core = src.m_core->clone();
}

VMetric::VMetric(VMetricCore* core) {
	m_core = core->clone();
}

VMetric::VMetric(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VMetricCore(session, sBuiltin, kwArgs);
}

VMetric::VMetric(VSession session, VHMetric handle) {
	m_core = NULL;
	VMetricCore* core = (VMetricCore*)handle;
	if (core == NULL) VP_THROW1(VERR_INVALID_CORE, "Metric");
	if (core->m_nCheckCode != VMetricCore::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, "Metric");
	if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, "Metric");
	m_core = (VMetricCore*)core->clone_core();
}

VMetric::~VMetric() { m_core->destroy(); }

VMetric& VMetric::operator =(const VMetric& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VHMetric VMetric::cloneCore() {
	return (VHMetric)m_core->clone();
}

VHMetric VMetric::cloneHandle() {
	return (VHMetric)m_core->clone_handle();
}

VMetricCore* VMetric::getClone() {
	return (VMetricCore*)m_core->clone_core();
}

VMetricCore* VMetric::getCore() {
	return m_core;
}

bool VMetric::isValid() {
	return m_core != NULL;
}
void VMetric::closeHandle() {
	if (this) m_core->destroy_handle();
}

VSession VMetric::session() {
	return m_core->m_session;
}

int VMetric::getRefCnt() {
	return m_core->getRefCnt();
}

int VMetric::getNth() {
	return m_core->getNth();
}

void VMetric::incRefCount() {
	m_core->incRefCnt();
}

VMetricCore::VMetricCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Metric) {
	m_nCheckCode = ms_nCheckCode;
	m_session = session;
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_onCreate();
}

VMetricCore::~VMetricCore() {
	m_onDelete();
	m_nCheckCode = 0;
}

//=========== API Object Common Part End =======================

VTensorDict VMetric::evaluate(VTensorDict preds) {
	m_core->m_preds = preds;

	int nDevice = preds.begin()->second.device();
	
	VTensorDict xs = vutils.mergeTensorDict(preds, {}, "pred");

	VExecTracer tracer = m_core->m_execTracer;

	if (!tracer.isValid() && !session().getNoTracer()) {
		tracer = VExecTracer(session(), "metric", {});
		m_core->m_execTracer = tracer;
	}
	else if (tracer.isValid() && session().getNoTracer()) {
		tracer = VExecTracer();
		m_core->m_execTracer = tracer;
	}

	if (tracer.hasValidHistory(xs)) {
		if (session().isTracerDirty()) {
			tracer.reset();
		}
		else {
			return tracer.executeHistory();
		}
	}

	tracer.setInput(xs);

	VTensorDict metric;

	if (m_core->m_sBuiltin == "multiple") {
		for (auto& it : m_core->m_propDict) {
			VMetric child(session(), (VHMetric)it.second);
			VTensorDict childMetric = child.evaluate(preds);
			for (auto& it2 : childMetric) {
				if (it2.first == "#") metric[it.first] = it2.second;
				else metric[it2.first] = it2.second;
			}
		}
	}
	else if (m_core->m_sBuiltin == "custom") {
		m_core->m_staticTensors = vutils.toDevice(m_core->m_staticTensors, nDevice, tracer);

		VTensorDict input = vutils.mergeTensorDict(preds, {}, "#pred", "#y");
		VTensorDict vars = vutils.mergeTensorDict(input, m_core->m_staticTensors);

		//printf("VMetric::evaluate(nDevice:%d)\n", nDevice);
		//vutils.toDevice(m_core->m_staticTensors, nDevice, tracer);

		//VTensorDict input = vutils.dictToDevice(xs, nDevice, tracer);
		//VTensorDict vars = vutils.mergeTensorDict(input, m_core->m_staticTensors);

		/*
		for (auto& it : xs) {
			VTensor tensor = it.second;
			if (tensor.device() != nDevice) tensor = tensor.toDevice(nDevice, tracer);
			if (it.first == "#") xs["#pred"] = tensor;
			else if (xs.find(it.first) == xs.end()) xs[it.first] = tensor;
			else VP_THROW(VERR_UNDEFINED);
		}

		for (auto& it : statics) {
			VTensor tensor = it.second;
			if (tensor.device() != nDevice) tensor = tensor.toDevice(nDevice);
			if (it.first == "#") xs["#y"] = tensor;8
			else if (xs.find(it.first) == xs.end()) xs[it.first] = tensor;
			else VP_THROW(VERR_UNDEFINED);
		}
		*/

		bool noGrad = true;

		VTensorDict terms;
		VTensorDict resultTerms;

		// 순전파 처리 과정에서의 콜백도 지원하여야 함
		// 손실함수 처리에 대한 역전파 설정 정보를 session으로부터 얻어와야 함, 임시로 cbInfo = NULL로 처리함
		VCbBackInfo cbInfo;

		for (auto& it : m_core->m_graphs) {
			if (terms.find(it.first) == terms.end()) {
				terms[it.first] = VTensor();	// 재귀적 호출 탐지를 위해 일단 더미값을 설정해둔다.
				it.second.evaluateGraph(vars, &terms, it.first, m_core->m_graphs, true, noGrad, nDevice, cbInfo, tracer);
			}
		}

		VDict metricTerms = m_core->m_propDict["terms"];

		for (auto& it : metricTerms) {
			resultTerms[it.first] = terms[it.first];
		}

		return resultTerms;
	}
	else {
		VTensorDict xs;

		for (auto& it : m_core->m_propDict) {
			if (it.first[0] == '#' && it.second.is_string()) {
				string name = it.second;
				VTensor tensor;
				if (name.length() > 7 && name.substr(0, 7) == "preds::") tensor = preds[name];
				/*
				else if (name.length() > 4 && name.substr(0, 4) == "ys::") tensor = statics[name];
				else if (preds.find(name) != preds.end()) {
					if (statics.find(name) != statics.end()) VP_THROW(VERR_UNDEFINED);
					tensor = preds[name];
				}
				else if (statics.find(name) != statics.end()) tensor = statics[name];
				else VP_THROW(VERR_CONDITIONAL_STATEMENT);
				*/

				if (tensor.device() != nDevice) {
					tensor = tensor.toDevice(nDevice, tracer);
				}
				xs[it.first] = tensor;
			}
		}

		bool noGrad = session().getNoGrad();

		// 순전파 처리 과정에서의 콜백도 지원하여야 함
		// 손실함수 처리에 대한 역전파 설정 정보를 session으로부터 얻어와야 함, 임시로 cbInfo = NULL로 처리함
		VCbBackInfo cbInfo;

		metric = m_core->m_graph.evaluateGraph(xs, true, noGrad, nDevice, cbInfo, tracer);
	}

	tracer.closeRecording(metric);

	return metric;
}

VList VMetric::GetBuiltinNames() {
	VList list;
	for (auto& it : ms_builtin) list.push_back(it);
	return list;
}

void VMetricCore::m_onCreate() {
	if (m_sBuiltin == "multiple") {
		return;
	}
	else if (m_sBuiltin == "custom") {
		VDict terms = m_propDict["terms"];
		VDict sHandles = m_propDict["statistics"];

		m_staticTensors = vutils.toTensorDict(m_session, sHandles);

		for (auto& it : terms) {
			m_graphs[it.first] = VGraph(m_session, NULL, GraphInit::term, m_session.device_man(), it.second, VList(), m_propDict);
		}

		VDict kwArgs = m_propDict["kwArgs"];

		for (auto& it : kwArgs) {
			if (it.first != "download") {
				VDict subTerms = it.second;

				for (auto& it : subTerms) {
					m_graphs[it.first] = VGraph(m_session, NULL, GraphInit::term, m_session.device_man(), it.second, VList(), m_propDict);
				}
			}
		}

		if (kwArgs.find("download") != kwArgs.end()) {
			if (kwArgs["download"].is_bool()) {
				if ((bool)kwArgs["download"]) {
					for (auto& it : m_graphs) {
						string key = it.first;
						if (terms.find(key) != terms.end()) continue; // 굳이 필요 없는 처리인 듯, 삭제 여부 검토할 것
						m_downloadTerms.push_back(key);
					}
				}
			}
			else if (kwArgs["download"].is_list()) {
				for (auto& it : (VList)kwArgs["download"]) {
					string term_name = it;

					if (term_name[0] == '%') {
						m_downloadTerms.push_back(term_name.substr(1));
					}
					else {
						VDict subTerms = kwArgs[term_name];
						for (auto& it2 : subTerms) {
							m_downloadTerms.push_back(it2.first);
						}
					}
				}
			}
			else {
				VP_THROW(VERR_UNDEFINED);
			}
		}
	}
	else {
		m_graph = VGraph(m_session, NULL, GraphInit::metric, m_session.device_man(), m_sBuiltin, VList(), m_propDict);
	}
}

void VMetricCore::m_onDelete() {}
