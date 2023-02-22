#include <cuda_runtime.h>

#include "../api_objects/vloss_core.h"
#include "../api_objects/vloss.h"
#include "../local_objects/vgraph.h"
#include "../api_objects/vtensor.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vcbbackinfo.h"
#include "../support/vmath.h"
#include "../support/vback_queue.h"
#include "../utils/vutils.h"

int VLossCore::ms_nCheckCode = 62351613;

VStrList VLoss::ms_builtin = { "custom", "mse", "crossentropy", "crossentropy_pos_idx", "crossentropy_softmax", "crossentropy_softmax_idx", "binary_crossentropy", "crossentropy_sigmoid"};

//=========== API Object Common Part Start =======================

VLoss::VLoss() {
	m_core = NULL;
}

VLoss::VLoss(const VLoss& src) {
	m_core = src.m_core->clone();
}

VLoss::VLoss(VLossCore* core) {
	m_core = core->clone();
}

VLoss::VLoss(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VLossCore(session, sBuiltin, kwArgs);
}

VLoss::VLoss(VSession session, VHLoss handle) {
	m_core = NULL;
	VLossCore* core = (VLossCore*)handle;
	if (core == NULL) VP_THROW1(VERR_INVALID_CORE, "Loss");
	if (core->m_nCheckCode != VLossCore::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, "Loss");
	if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, "Loss");
	m_core = (VLossCore*)core->clone_core();
}

VLoss::~VLoss() { m_core->destroy(); }

VLoss& VLoss::operator =(const VLoss& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VHLoss VLoss::cloneCore() {
	return (VHLoss)m_core->clone();
}

VHLoss VLoss::cloneHandle() {
	return (VHLoss)m_core->clone_handle();
}

VLossCore* VLoss::getClone() {
	return (VLossCore*)m_core->clone_core();
}

VLossCore* VLoss::getCore() {
	return m_core;
}

bool VLoss::isValid() {
	return m_core != NULL;
}
void VLoss::closeHandle() {
	if (this) m_core->destroy_handle();
}

VSession VLoss::session() {
	return m_core->m_session;
}

int VLoss::getRefCnt() {
	return m_core->getRefCnt();
}

int VLoss::getNth() {
	return m_core->getNth();
}

void VLoss::incRefCount() {
	m_core->incRefCnt();
}

VLossCore::VLossCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Loss) {
	m_nCheckCode = ms_nCheckCode;
	m_session = session;
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_onCreate();
}

VLossCore::~VLossCore() {
	m_onDelete();
	m_nCheckCode = 0;
}

//=========== API Object Common Part End =======================

VExecTracer VLoss::m_getTracer(VTensorDict xs, int mission) {
	bool noGrad = session().getNoGrad();
	int gradMode = noGrad ? 0 : 1;

	int nAvail = -1;

	if (0) {
		for (auto& it : xs) {
			printf("VLoss::m_getTracer(xs[%s], gradMode=%d, mission=%d) = %d\n", it.first.c_str(), gradMode, mission, it.second.getNth());
		}
	}

	//VExecTracer tracer = m_core->m_execTracer[gradMode][mission];
	VExecTracer tracer;

	for (int n = 0; n < 3; n++) {
		tracer = m_core->m_execTracer[n][gradMode][mission];

		if (tracer.hasValidHistory(xs)) {
			if (session().getNoTracer()) {
				tracer = VExecTracer();
				m_core->m_execTracer[n][gradMode][mission] = tracer;
			}

			return tracer;
		}
		else if (nAvail < 0 && !tracer.isValid()){
			nAvail = n;
		}
	}

	if (nAvail < 0) nAvail = 0;

	tracer = m_core->m_execTracer[nAvail][gradMode][mission];

	if (!tracer.isValid() && !session().getNoTracer()) {
		static string sMode[2] = { "test", "train" };
		static string sMission[3] = { "forward", "backward", "accuracy" };

		string name = "loss " + sMission[mission] + " in " + sMode[gradMode] + " mode";

		tracer = VExecTracer(session(), name, {});

		m_core->m_execTracer[nAvail][gradMode][mission] = tracer;
	}
	else if (tracer.isValid() && session().getNoTracer()) {
		tracer = VExecTracer();
		m_core->m_execTracer[nAvail][gradMode][mission] = tracer;
	}

	return tracer;
}

VTensorDict VLoss::m_evaluateMultipleLoss(VTensorDict preds, VTensorDict ys, bool download_all) {
	VTensorDict loss;

	for (auto& it : m_core->m_propDict) {
		VLoss child(session(), (VHLoss)it.second);
		VTensorDict childLoss = child.evaluate(preds, ys, download_all);
		for (auto& it2 : childLoss) {
			if (it2.first == "#") loss[it.first] = it2.second;
			else loss[it2.first] = it2.second;
		}
	}

	return loss;
}

VTensorDict VLoss::m_evaluateCustomLoss(VTensorDict xs, int nDevice, VExecTracer tracer, bool download_all) {
	VTensorDict vars = vutils.dictToDevice(xs, nDevice, tracer);

	bool noGrad = session().getNoGrad();

	VTensorDict terms;
	VCbBackInfo cbInfo;

	for (auto& it : m_core->m_graphs) {
		if (terms.find(it.first) == terms.end()) {
			terms[it.first] = VTensor();	// 재귀적 호출 탐지를 위해 일단 더미값을 설정해둔다.
			it.second.evaluateGraph(vars, &terms, it.first, m_core->m_graphs, true, noGrad, nDevice, cbInfo, tracer);
		}
	}

	VTensorDict downloadTerms;
	VList downloadNames = m_core->m_downloadTerms;

	for (auto& it : downloadNames) {
		downloadTerms[it] = terms[it];
	}

	VTensorDict resultTerms;

	VDict lossTerms = m_core->m_propDict["terms"];

	for (auto& it : lossTerms) {
		resultTerms[it.first] = terms[it.first];
		downloadTerms[it.first] = terms[it.first];
	}

	m_core->m_losses = resultTerms;

	VTensorDict result = downloadTerms; // download_all ? terms : resultTerms;

	return result;
}

VTensorDict VLoss::m_preprocLossOpnd(VTensorDict preds, VTensorDict ys) {
	VTensorDict xs;

	for (auto& it : m_core->m_propDict) {
		if (it.first[0] == '#' && it.second.is_string()) {
			string name = it.second;

			VTensor tensor;
			if (preds.find(name) != preds.end()) {
				if (ys.find(name) != ys.end()) VP_THROW(VERR_UNDEFINED);
				tensor = preds[name];
			}
			else if (ys.find(name) != ys.end()) tensor = ys[name];
			else if (name.length() > 7 && name.substr(0, 7) == "preds::") {
				tensor = preds[name.substr(7)];
			}
			else if (name.length() > 4 && name.substr(0, 4) == "ys::") {
				tensor = ys[name.substr(4)];
			}
			else if (name == "pred" && preds.find("#") != preds.end()) {
				tensor = preds["#"];
			}
			else if (name == "y" && ys.find("#") != ys.end()) {
				tensor = ys["#"];
			}
			else {
				VP_THROW1(VERR_LOSS_TERM_NOT_FOUND, name);
			}

			xs[it.first] = tensor;
		}
	}

	return xs;
}

VTensorDict VLoss::m_evaluateLoss(VTensorDict xs, int nDevice, VExecTracer tracer) {
	// 순전파 처리 과정에서의 콜백도 지원하여야 함
	// 손실함수 처리에 대한 역전파 설정 정보를 session으로부터 얻어와야 함, 임시로 cbInfo = NULL로 처리함
	VCbBackInfo cbInfo;

	VTensorDict loss;

	VTensorDict vars = vutils.dictToDevice(xs, nDevice, tracer);

	bool noGrad = session().getNoGrad();

	loss = m_core->m_graph.evaluateGraph(vars, true, noGrad, nDevice, cbInfo, tracer);

	return loss;
}

VTensorDict VLoss::evaluate(VTensorDict preds, VTensorDict ys, bool download_all) {
	m_core->m_preds = preds;

	int nDevice = preds.begin()->second.device();

	VTensorDict loss;

	VExecTracer tracer;

	if (m_core->m_sBuiltin == "multiple") {
		loss = m_evaluateMultipleLoss(preds, ys, download_all);
	}

	if (m_core->m_sBuiltin == "custom") {
		m_core->m_staticTensors = vutils.toDevice(m_core->m_staticTensors, nDevice, tracer);

		VTensorDict input = vutils.mergeTensorDict(preds, ys, "#pred", "#y");
		VTensorDict vars = vutils.mergeTensorDict(input, m_core->m_staticTensors);

		tracer = m_getTracer(vars, 0);

		if (tracer.hasValidHistory(vars)) {
			if (session().isTracerDirty()) {
				tracer.reset();
			}
			else {
				loss = tracer.executeHistory();
				m_core->m_losses = loss;
				return loss;
			}
		}

		tracer.setInput(vars);

		VTensorDict result = m_evaluateCustomLoss(vars, nDevice, tracer, download_all);
		
		tracer.closeRecording(result);

		loss = result;
	}
	else if (m_core->m_sBuiltin != "multiple") {
		VTensorDict xs = m_preprocLossOpnd(preds, ys);

		tracer = m_getTracer(xs, 0);

		if (tracer.hasValidHistory(xs)) {
			if (session().isTracerDirty()) {
				tracer.reset();
			}
			else {
				loss = tracer.executeHistory();
				m_core->m_losses = loss;
				return loss;
			}
		}

		tracer.setInput(xs);

		loss = m_evaluateLoss(xs, nDevice, tracer);

		tracer.closeRecording(loss);
	}

	m_core->m_losses = loss;

	for (auto& it : loss) {
		it.second.setLossFunc(m_core);
	}

	return loss;
}

VTensorDict VLoss::eval_accuracy(VTensorDict preds, VTensorDict ys, bool download_all) {
	VTensorDict accurs;

	if (m_core->m_sBuiltin == "multiple") {
		for (auto& it : m_core->m_propDict) {
			VLoss child(session(), (VHLoss)it.second);
			VTensorDict childAccs = child.eval_accuracy(preds, ys, download_all);
			for (auto& it2 : childAccs) {
				if (it2.first == "#") accurs[it.first] = it2.second;
				else accurs[it2.first] = it2.second;
			}
		}
	}

	if (m_core->m_sBuiltin == "custom") {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
	else if (m_core->m_sBuiltin != "multiple") {
		VTensorDict xs;

		// 231행 이하와 같은 내용 복사해 옴: 이상 없으면 서브루틴으로 독립시킬 것
		for (auto& it : m_core->m_propDict) {
			if (it.first[0] == '#' && it.second.is_string()) {
				string name = it.second;

				VTensor tensor;
				if (preds.find(name) != preds.end()) {
					if (ys.find(name) != ys.end()) VP_THROW(VERR_UNDEFINED);
					tensor = preds[name];
				}
				else if (ys.find(name) != ys.end()) tensor = ys[name];
				else if (name.length() > 7 && name.substr(0, 7) == "preds::") {
					tensor = preds[name.substr(7)];
				}
				else if (name.length() > 4 && name.substr(0, 4) == "ys::") {
					tensor = ys[name.substr(4)];
				}
				else if (name == "pred" && preds.find("#") != preds.end()) {
					tensor = preds["#"];
				}
				else if (name == "y" && ys.find("#") != ys.end()) {
					tensor = ys["#"];
				}
				else {
					VP_THROW1(VERR_LOSS_TERM_NOT_FOUND, name);
				}

				xs[it.first] = tensor;
			}
		}

		VExecTracer tracer = m_getTracer(xs, 2);

		if (tracer.hasValidHistory(xs)) {
			if (session().isTracerDirty()) {
				tracer.reset();
			}
			else {
				return tracer.executeHistory();
			}
		}

		tracer.setInput(xs);

		string pred_name = (m_core->m_sBuiltin == "mse") ? "#estimate" : "#logit";
		string y_name = (m_core->m_sBuiltin == "mse") ? "#answer" : "#label";

		VTensor pred = xs[pred_name];
		VTensor y = xs[y_name];

		VShape pshape = pred.shape();
		VShape yshape = y.shape();

		y = y.toDevice(pred.device(), tracer);

		string reduction = vutils.seek_dict(m_core->m_propDict, "reduction", "sum");

		if (m_core->m_sBuiltin == "mse") {
			if (pshape == yshape) {
				string metric = vutils.seek_dict(m_core->m_propDict, "metric", "mae");
				if (metric == "mae") {
					VTensor MAE = pred.subtract(y, tracer).abs(tracer);
					accurs["#"] = (reduction == "mean") ? MAE.mean(tracer) : MAE.sum(tracer);
				}
				else if (metric == "mse") {
					VTensor MSE = pred.subtract(y, tracer).square(tracer).mean(tracer);
					accurs["#"] = (reduction == "mean") ? MSE.mean(tracer) : MSE.sum(tracer);
				}
				else if (metric == "rmse") {
					VTensor RMSE = pred.subtract(y, tracer).square(tracer).mean(tracer).sqrt(tracer);
					accurs["#"] = (reduction == "mean") ? RMSE.mean(tracer) : RMSE.sum(tracer);
				}
				/*
				RMSE.dump("RMSE");
				VTensor a = RMSE.complement_1(tracer);
				a.dump("complement_1");
				accurs["#"] = a.mean(tracer);
				*/
			}
			else {
				VP_THROW(VERR_UNKNOWN);
			}
		}
		else if (m_core->m_sBuiltin == "crossentropy") {
			if (pshape == yshape) {
				VTensor answer = y.argmax(tracer);
				VTensor estimate = pred.argmax(tracer);
				VTensor correct = answer.equal(estimate, tracer);
				accurs["#"] = (reduction == "mean") ? correct.mean(tracer) : correct.sum(tracer);
			}
			else if (pshape.remove_end() == yshape.remove_end() && yshape[-1] == 1 || pshape.remove_end() == yshape) {
				if (y.type() != VDataType::int32) VP_THROW(VERR_UNKNOWN);
				VTensor estimate = pred.argmax(tracer);
				VTensor correct = y.equal(estimate, tracer);
				accurs["#"] = (reduction == "mean") ? correct.mean(tracer) : correct.sum(tracer);
			}
			else {
				VP_THROW(VERR_UNKNOWN);
			}
		}
		else if (m_core->m_sBuiltin == "crossentropy_pos_idx") {
			if ((pshape.remove_end() != yshape.remove_end() || yshape[-1] != 1) && pshape.remove_end() != yshape) VP_THROW(VERR_UNDEFINED);

			VTensor estimate = pred.argmax(tracer);
			VTensor correct = y.equal(estimate, tracer);
			y.setOpArg("0", 0);
			VTensor mask = y.greater_than(tracer);
			VTensor correct_masked = correct.mult(mask, tracer);

			if (0) {
				VTensor mean = correct_masked.mean(tracer);
				VTensor sum = correct_masked.sum(tracer);

				estimate.dump("estimate");
				y.dump("y");
				correct.dump("correct");
				mask.dump("mask");
				correct_masked.dump("correct_masked");
				mean.dump("mean");
				sum.dump("sum");
			}

			accurs["#"] = (reduction == "mean") ? correct_masked.mean(tracer) : correct_masked.sum(tracer);
		}
		else if (m_core->m_sBuiltin == "crossentropy_softmax") {
			if (pshape != yshape) VP_THROW(VERR_UNDEFINED);

			VTensor answer = y.argmax(tracer);
			VTensor estimate = pred.argmax(tracer);
			VTensor correct = answer.equal(estimate, tracer);
			accurs["#"] = (reduction == "mean") ? correct.mean(tracer) : correct.sum(tracer);
		}
		else if (m_core->m_sBuiltin == "crossentropy_softmax_idx") {
			if ((pshape.remove_end() != yshape.remove_end() || yshape[-1] != 1) && pshape.remove_end() != yshape) VP_THROW(VERR_UNDEFINED);

			VTensor estimate = pred.argmax(tracer);
			VTensor correct = y.equal(estimate, tracer);
			accurs["#"] = (reduction == "mean") ? correct.mean(tracer) : correct.sum(tracer);
		}
		else if (m_core->m_sBuiltin == "crossentropy_sigmoid" || m_core->m_sBuiltin == "binary_crossentropy") {
			if (pshape == yshape) {
				pred.setOpArgs({ {"0", 0.0f} });
				y.setOpArgs({ {"0", 0.5f} });

				VTensor estimate = pred.greater_than(tracer);
				VTensor answer = y.greater_than(tracer);

				VTensor correct = y.equal(estimate, tracer);
				accurs["#"] = (reduction == "mean") ? correct.mean(tracer) : correct.sum(tracer);
			}
			else {
				VP_THROW(VERR_UNKNOWN);
			}
		}
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}

		tracer.closeRecording(accurs);
	}

	return accurs;
}

void VLoss::backward() {
	VTensorDict lossTensors = m_core->m_losses;
	VTensorDict predTensors = m_core->m_preds;

	VTensorDict xs = vutils.mergeTensorDict(lossTensors, predTensors, "#loss", "#pred");

	VExecTracer tracer = m_getTracer(xs, 1);

	if (tracer.hasValidHistory(xs)) {
		if (session().isTracerDirty()) {
			tracer.reset();
		}
		else {
			tracer.executeHistory();
			return;
		}
	}

	tracer.setInput(xs);

	VBackQueue queue(session(), tracer);

	for (auto& it : lossTensors) {
		queue.regist(it.second, VTensor());
	}

	queue.registCrossNodes(m_core->m_preds);

	while (!queue.isEnd()) {
		queue.step();
	}

	tracer.closeRecording({});
}

VList VLoss::GetBuiltinNames() {
	VList list;
    for (auto& it : ms_builtin) list.push_back(it);
	return list;
}

void VLossCore::m_onCreate() {
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

		// terms gkdemfdp eogks ㄱㄷ옃샤ㅐㅜ cjfl alc gkqtks cjfl, rmflrh qorxmfozld wldnjs 
	}
	else {
		m_graph = VGraph(m_session, NULL, GraphInit::loss, m_session.device_man(), m_sBuiltin, VList(), m_propDict);
	}
}

void VLossCore::m_onDelete() {
}
