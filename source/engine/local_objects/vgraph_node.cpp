#include "../local_objects/vgraph_node_core.h"
#include "../local_objects/vgraph_node.h"
#include "../local_objects/vgraph_core.h"
#include "../local_objects/vdevicemanager.h"
#include "../api_objects/vfunction.h"
#include "../support/vmath.h"
#include "../api/vconst.h"
#include "../utils/vutils.h"

VGraphNode::VGraphNode() {
	m_core = NULL;
}

VGraphNode::VGraphNode(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VGraphNodeCore(session, sBuiltin, kwArgs);
}

VGraphNode::VGraphNode(const VGraphNode& src) {
	m_core = src.m_core->clone();
}

VGraphNode::VGraphNode(VGraphNodeCore* core) {
	m_core = core->clone();
}

VGraphNode::~VGraphNode() {
	m_core->destroy();
}

VGraphNode& VGraphNode::operator =(const VGraphNode& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VGraphNodeCore* VGraphNode::getClone() {
	return (VGraphNodeCore*)m_core->clone_core();
}

VGraphNodeCore* VGraphNode::getCore() {
	return m_core;
}

void VGraphNode::destroyCore() {
	if (m_core->getRefCnt() > 1) m_core->destroy();
	else {
		m_core->destroy();
		m_core = NULL;
	}
}

VSession VGraphNode::session() const {
	return m_core->m_session;
}

bool VGraphNode::isValid() {
	return m_core != NULL;
}

int VGraphNode::getRefCnt() {
	return m_core->getRefCnt();
}

int VGraphNode::getNth() {
	return m_core->getNth();
}

VGraphNodeCore::VGraphNodeCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::GraphNode) {
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_session = session,
	m_setup();
}

bool VGraphNode::ms_bExprTrace = false;

VGraphNode::VGraphNode(VSession session, string sExp, VList params, int& pos, VDeviceManager devman, VGraphCore* pGraphCore) {
	m_core = new VGraphNodeCore(session);
	m_core->m_deviceMan = devman;
	m_core->m_pGraphCore = pGraphCore;
	m_core->m_setup(sExp, params, pos);

	if (m_core->m_pGraphCore == NULL) {
		VP_THROW(VERR_INVALID_CORE);
	}
}

VGraphNode::VGraphNode(bool bDeepCopy, const VGraphNode& src, VDeviceManager devman, VGraphCore* pGraphCore) {
	m_core = new VGraphNodeCore(src.session());
	m_core->m_deviceMan = devman;
	m_core->m_pGraphCore = pGraphCore;
	m_core->m_setup(src.m_core);

	if (m_core->m_pGraphCore == NULL) {
		VP_THROW(VERR_INVALID_CORE);
	}
}

VGraphNode::VGraphNode(VGraphNodeCore* pSrc, VGraphOpCode opCode, vector<VGraphNode> children) {
	m_core = new VGraphNodeCore(pSrc->m_session);
	m_core->m_deviceMan = pSrc->m_deviceMan;
	m_core->m_pGraphCore = pSrc->m_pGraphCore;
	m_core->m_opCode = opCode;
	m_core->m_children = children;
}

VTensor VGraphNode::evaluate(VDict args, VCbBackInfo cbInfo, VExecTracer tracer) {
	VTensorList operands;

	bool train = m_core->m_pGraphCore->m_train;
	bool no_grad = m_core->m_pGraphCore->m_noGrad;
	bool needGrad = false;

	if (0) {
		printf("VGraphNode::evaluate(train = %d, no_grad = %d)\n", train, no_grad);
	}

	VDict subargs;

	for (auto& it : m_core->m_children) {
		VTensor operand = it.evaluate(subargs, cbInfo, tracer);

		if (operand.isValid()) {	// valid하지 않은 그래프 노드는 ctx 리스트에 각종 인자 정보를 제공한다.
			bool opndNeedGrad = operand.needGrad();
			if (opndNeedGrad) {
				needGrad = true;
			}
			operands.push_back(operand);
		}
	}

	int nOpCnt = (int)operands.size();

	if (nOpCnt > 0) operands[0].setOpArgs(subargs);
	if (no_grad) needGrad = false;

	VTensor result;

	if (0) {
		printf("[FORWARD VGraphOpCode::%s] \n", GraphOpCodeName(m_core->m_opCode).c_str());
	}

	switch (m_core->m_opCode) {
	case VGraphOpCode::input:
		result = VTensor(m_core->m_pGraphCore->seekInput(m_core->m_varName));
		return result;
	case VGraphOpCode::term:
		result = VTensor(m_core->m_pGraphCore->m_seekTerm(m_core->m_varName, cbInfo, tracer));
		return result;
	case VGraphOpCode::side:
		args[m_core->m_varName] = m_core->m_aux;
		if (m_core->m_aux == "") {
			result = VTensor(m_core->m_pGraphCore->seekInput("#"));
		}
		else if (m_core->m_pGraphCore->m_sideTerms.find(m_core->m_aux) != m_core->m_pGraphCore->m_sideTerms.end()) {
			result = m_core->m_pGraphCore->m_sideTerms[m_core->m_aux];
		}
		else if (m_core->m_pGraphCore->m_xs.find(m_core->m_aux) != m_core->m_pGraphCore->m_xs.end()) {
			result = m_core->m_pGraphCore->m_xs[m_core->m_aux];
		}
		else {
			VP_THROW(VERR_INVALID_DICT_KEY);
		}
		return result;
	case VGraphOpCode::pm:
		if (nOpCnt != 0) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		if (m_core->m_pm.hasNoData()) {
			return m_core->m_pm;
		}
		else if (m_core->m_deviceMan.getCurDevice() < 0) {
			result = m_core->m_pm;
		}
		else {
			// m_core->m_pm 텐서가 result 텐서로 바뀌는데 이 변환 정보를 역전파 콜백 슬롯 생성시 파악할 수 있어야 함!!!
			result = m_core->m_pm.toDevice(m_core->m_deviceMan.getCurDevice(), tracer);
			cbInfo.addDeviceConvInfo(m_core->m_pm.getNth(), result);

			if (!m_core->m_grad.isValid()) {	
				result.setOpArg("moving_stat_src", (int64)(VObjCore*)m_core->m_pm.getCore());
			}
		}

		result = m_core->m_pGraphCore->optimizerPreproc(result, m_core->m_nesterovPmSet, tracer); // nesterov Optimizer 등의 경우 parameter 이용 전에 옵티마이저 나름의 전처리 과정을 거친다.

		if (!no_grad) {
			// 내부 처리 과정에서 moving_stat 등 기울기 필요없는 파라미터를 강제로 기울기 제공 대상으로 변경하여 이를 차단함, 부작용 여부에 유의할 것
			result.keepBackpropParamInfo(m_core->m_opCode, m_core->m_grad);
		}
		tracer.addTensor(result);
		return result;
	case VGraphOpCode::pmset:
	{
		VDict pmSet = m_core->m_pmsetToCurDevice(cbInfo, tracer);

		if (m_core->m_varName != "") {
			args[m_core->m_varName] = pmSet;
		}
		else {
			args[to_string((int)args.size())] = pmSet;
		}

		result = tracer.createTensor(session(), VShape({ 1 }), VDataType::float32, m_core->m_deviceMan.getCurDevice());

		if (!no_grad) {
			result.keepBackpropParamInfo(m_core->m_opCode, m_core->m_grad);
			result.setNeedGrad(true);
		}
		return result;
	}
	case VGraphOpCode::_user_defined_function:
	{
		VDict opArgs = operands[0].getOpArgs();
		VFunctionCore* functor = (VFunctionCore*)(int64)m_core->m_aux;
		VFunction function(functor);
		int nInst = getNth();
		result = function.forward(nInst, operands, opArgs);
		opArgs["__functor__"] = m_core->m_aux;
		opArgs["__funcinst__"] = nInst;
		tracer.addCallForwardUDF(functor, nInst, result, operands, opArgs);
	}
		break;
	case VGraphOpCode::_int:
	case VGraphOpCode::_bool:
	case VGraphOpCode::str:
	case VGraphOpCode::_float:
		if (m_core->m_varName != "") {
			args[m_core->m_varName] = m_core->m_aux;
		}
		else {
			args[to_string((int)args.size())] = m_core->m_aux;
		}
		return result;
	case VGraphOpCode::shape:
	{
		VShape shape = m_core->m_aux;
		shape = shape.copy();
		if (m_core->m_varName != "") {
			args[m_core->m_varName] = shape;
		}
		else {
			args[to_string((int)args.size())] = shape;
		}
		return result;
	}
	case VGraphOpCode::list:
	{
		VList list = m_core->m_aux;
		list = vutils.copy(list);
		if (m_core->m_varName != "") {
			args[m_core->m_varName] = list;
		}
		else {
			args[to_string((int)args.size())] = list;
		}
		return result;
	}
	case VGraphOpCode::matmul:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].matmul(operands[1], tracer);
		break;
	case VGraphOpCode::conv2d:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].conv2d(operands[1], tracer);
		break;
	case VGraphOpCode::conv2d_transposed:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].conv2d_transposed(operands[1], tracer);
		break;
	case VGraphOpCode::conv2d_dilated:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].conv2d_dilated(operands[1], tracer);
		break;
	case VGraphOpCode::rnn:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].rnn(train, operands[1], tracer);
		break;
	case VGraphOpCode::lstm:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].lstm(train, operands[1], tracer);
		break;
	case VGraphOpCode::gru:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].gru(train, operands[1], tracer);
		break;
	case VGraphOpCode::flatten:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].flatten(tracer);
		break;
	case VGraphOpCode::add:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].add(operands[1], tracer);
		break;
	case VGraphOpCode::add_2d_bias:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].add_2d_bias(operands[1], tracer);
		break;
	case VGraphOpCode::mult:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].mult(operands[1], tracer);
		break;
	case VGraphOpCode::se_mult:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].se_mult(operands[1], tracer);
		break;
	case VGraphOpCode::subtract:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].subtract(operands[1], tracer);
		break;
	case VGraphOpCode::div:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].div(operands[1], tracer);
		break;
	case VGraphOpCode::_not:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0]._not(tracer);
		needGrad = false;
		break;
	case VGraphOpCode::_and:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0]._and(operands[1], tracer);
		needGrad = false;
		break;
	case VGraphOpCode::_or:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0]._or(operands[1], tracer);
		needGrad = false;
		break;
	case VGraphOpCode::_eq:
		if (nOpCnt == 2) result = operands[0].equal(operands[1], tracer);
		else if (nOpCnt == 1) result = operands[0].equal(tracer);	// 상수와의 비교
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::_gt:
		if (nOpCnt == 2) result = operands[0].greater_than(operands[1], tracer);
		else if (nOpCnt == 1) result = operands[0].greater_than(tracer);	// 상수와의 비교
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::_lt:
		if (nOpCnt == 2) result = operands[0].less_than(operands[1], tracer);
		else if (nOpCnt == 1) result = operands[0].less_than(tracer);	// 상수와의 비교
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::_ge:
		if (nOpCnt == 2) result = operands[0].greater_equal(operands[1], tracer);
		else if (nOpCnt == 1) result = operands[0].greater_equal(tracer);	// 상수와의 비교
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::_le:
		if (nOpCnt == 2) result = operands[0].less_equal(operands[1], tracer);
		else if (nOpCnt == 1) result = operands[0].less_equal(tracer);	// 상수와의 비교
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::gt_cross:
		if (nOpCnt == 2) result = operands[0].greater_than_cross(operands[1], tracer);
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::lt_cross:
		if (nOpCnt == 2) result = operands[0].less_than_cross(operands[1], tracer);
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::ge_cross:
		if (nOpCnt == 2) result = operands[0].greater_equal_cross(operands[1], tracer);
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::le_cross:
		if (nOpCnt == 2) result = operands[0].less_equal_cross(operands[1], tracer);
		else VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		needGrad = false;
		break;
	case VGraphOpCode::pickup:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].pickup(operands[1], tracer);
		break;
	case VGraphOpCode::pickup_static:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].pickup_static(operands[1], tracer);
		needGrad = false;
		break;
	case VGraphOpCode::abs:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].abs(tracer);
		break;
	case VGraphOpCode::square:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].square(tracer);
		break;
	case VGraphOpCode::upsample:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].upsample(tracer);
		break;
	case VGraphOpCode::pass:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].pass(tracer);
		break;
	case VGraphOpCode::activate:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].activate(tracer);
		break;
	case VGraphOpCode::normal_noise:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].normal_noise(tracer);
		break;
	case VGraphOpCode::uniform_noise:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].uniform_noise(tracer);
		break;
	case VGraphOpCode::normal_random:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].normal_random(tracer);
		break;
	case VGraphOpCode::uniform_random:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].uniform_random(tracer);
		break;
	case VGraphOpCode::round:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].round(tracer);
		break;
	case VGraphOpCode::codeconv:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].codeconv(tracer);
		break;
	case VGraphOpCode::cosinesim:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].cosinesim(operands[1], tracer);
		break;
	case VGraphOpCode::selectntop:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].selectntop(tracer);
		break;
	case VGraphOpCode::selectntoparg:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].selectntoparg(tracer);
		break;
	case VGraphOpCode::mean:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].mean(tracer);
		break;
	case VGraphOpCode::sum:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].sum(tracer);
		break;
	case VGraphOpCode::reshape:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].reshape(tracer);
		break;
	case VGraphOpCode::transpose:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].transpose(tracer);
		break;
	case VGraphOpCode::extract:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].extract(tracer);
		break;
	case VGraphOpCode::stride:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].stride(tracer);
		break;
	case VGraphOpCode::maxpool:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].maxpool(tracer);
		break;
	case VGraphOpCode::avgpool:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].avgpool(tracer);
		break;
	case VGraphOpCode::max:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].max(tracer);
		break;
	case VGraphOpCode::min:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].min(tracer);
		break;
	case VGraphOpCode::argmax:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].argmax(tracer);
		needGrad = false;
		break;
	case VGraphOpCode::argmin:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].argmin(tracer);
		needGrad = false;
		break;
	case VGraphOpCode::maximum:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].maximum(operands[1], tracer);
		break;
	case VGraphOpCode::minimum:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].minimum(operands[1], tracer);
		break;
	case VGraphOpCode::globalavg:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].globalavg(tracer);
		break;
	case VGraphOpCode::adaptiveavg:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].adaptiveavg(tracer);
		break;
	case VGraphOpCode::embed:
		if (nOpCnt != 2) {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
		result = operands[0].embed(operands[1], tracer);
		break;
	case VGraphOpCode::batchnorm:
		if (nOpCnt != 5) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].batchnorm(operands[1], operands[2], operands[3], operands[4], train, tracer);
		break;
	case VGraphOpCode::layernorm:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].layernorm(tracer);
		break;
	case VGraphOpCode::mh_attention:
		if (nOpCnt != 12) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].mh_attention(operands[1], operands[2], operands[3], operands[4], operands[5], operands[6], operands[7], operands[8], operands[9], operands[10], operands[11], tracer);
		break;
	case VGraphOpCode::dropout:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].dropout(train, tracer);
		break;
	case VGraphOpCode::crossentropy:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].crossentropy(operands[1], tracer);
		break;
	case VGraphOpCode::crossentropy_sigmoid:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].crossentropy_sigmoid(operands[1], tracer);
		break;
	case VGraphOpCode::crossentropy_pos_idx:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].crossentropy_pos_idx(operands[1], tracer);
		break;
	case VGraphOpCode::iou_cross_xywh:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].iou_cross_xywh(operands[1], tracer);
		break;
	case VGraphOpCode::iou_cross_lrtb:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].iou_cross_lrtb(operands[1], tracer);
		break;
	case VGraphOpCode::ciou_loss:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].iou_loss(operands[1], m_core->m_opCode, tracer);
		break;
	case VGraphOpCode::sigmoid_crossentropy_with_logits:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].sigmoid_crossentropy_with_logits(tracer);
		break;
	case VGraphOpCode::sigmoid_crossentropy_with_logits_idx:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].sigmoid_crossentropy_with_logits_idx(operands[1], tracer);
		break;
	case VGraphOpCode::sigmoid:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].sigmoid(tracer);
		break;
	case VGraphOpCode::concat:
	case VGraphOpCode::hstack:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].concat(operands[1], tracer);
		break;
	case VGraphOpCode::subvector:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].subvector(tracer);
		break;
	case VGraphOpCode::exp:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].exp(tracer);
		break;
	case VGraphOpCode::log:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].log(tracer);
		break;
	case VGraphOpCode::to_tensor:
		if (nOpCnt != 0) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = m_createTensor(subargs["0"], tracer);
		needGrad = false;
		break;
	case VGraphOpCode::to_filter:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].to_filter(tracer);
		needGrad = false;
		break;
	case VGraphOpCode::to_boxes:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].to_boxes(operands[1], tracer);
		break;
	case VGraphOpCode::complement_1:
		if (nOpCnt != 1) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].complement_1(tracer);
		break;
	case VGraphOpCode::add_residual:
		if (nOpCnt != 2) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		result = operands[0].add_residual(operands[1], tracer);
		break;
	case VGraphOpCode::parallel_all:
	{
		bool reverse = vutils.seek_dict(m_core->m_pGraphCore->m_propDict, "reverse", false);
		VTensorDict xs = m_core->m_pGraphCore->m_xs;
		if (!needGrad) needGrad = m_lookupNeedGrad(xs);
		int64 n = 0;
		for (auto& it : xs) {
			if (n++ == 0) result = it.second;
			else {
				VTensor new_result = reverse ? it.second.parallel_concat(result, tracer) : result.parallel_concat(it.second, tracer);
				new_result.keepBackpropOperandsInfo(needGrad, VGraphOpCode::parallel_concat, { result, it.second });
				/*
				if (needGrad) {
					new_result.keepBackpropOperandsInfo(VGraphOpCode::parallel_concat, { result, it.second });
				}
				*/
				result = new_result;
			}
		}
		break;
	}
	case VGraphOpCode::add_all:
	{
		VTensorDict xs = m_core->m_pGraphCore->m_xs;
		if (!needGrad) needGrad = m_lookupNeedGrad(xs);
		int64 n = 0;
		for (auto& it : xs) {
			if (n++ == 0) result = it.second;
			else {
				VTensor new_result = result.add(it.second, tracer);
				new_result.keepBackpropOperandsInfo(needGrad, VGraphOpCode::add, { result, it.second });
				/*
				if (needGrad) {
					new_result.keepBackpropOperandsInfo(VGraphOpCode::add, { result, it.second });
				}
				*/
				result = new_result;
			}
		}
		break;
	}
	case VGraphOpCode::stack_all :
	{
		int64 tail_size = m_core->m_pGraphCore->m_propDict["tsize"];

		VTensorDict xs = m_core->m_pGraphCore->m_xs;
		if (!needGrad) needGrad = m_lookupNeedGrad(xs);
		result = VTensor::stack_all(session(), xs, tail_size, tracer);
		VTensorList operands;
		for (auto& it : xs) {
			operands.push_back(it.second);
		}
		result.keepBackpropOperandsInfo(needGrad, VGraphOpCode::stack_all, operands);
		/*
		for (auto& it : xs) {
			//it.second.keepBackpropOperandsInfo(needGrad, VGraphOpCode::add, { it.second });
			continue;
			// 아래 문제 해결 방안을 찾는 일환으로  result를 오퍼랜드에서 제거하고 실험해 보는 중
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
			// 아래 처리에서 3인자로 {result, it.second} 주는 탓에
			// result가 xs 루프 처리 중 매번 오퍼랜드로 삽입되는 바람에
			// 3회 중복 등록됨, 현재 3이 고스란이 남는 점을 볼 때 result 해소는 1회가 아니라 전혀 안되는 듯
			// 중복 등록 방지 혹은 등록 방지 혹은 해소 누락 찾아 해결 중 하나가 필요할 듯 
			it.second.keepBackpropOperandsInfo(needGrad, VGraphOpCode::add, { result, it.second });
		}
		*/
		break;
	}
	default:
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	if (0) {
		if (result.isValid()) {
			static int nth = 0;
			//float rf = result.square(tracer).mean(tracer).getElement({ 0 }, tracer);
			float rf = 0;
			if (result.type() == VDataType::float32 && result.shape().size() > 1 && result.shape()[1] > 1) {
				VList index;
				for (int64 n = 0; n < result.shape().size(); n++) index.push_back(0);
				index[(result.shape().size() == 4) ? 2 : 1] = 1;
				rf = result.getElement(index, tracer);
			}
			printf("[%d FORWARD VGraphOpCode::%s] processed => %s[avg:%f, dev%d]\n", nth++, GraphOpCodeName(m_core->m_opCode).c_str(), result.shape().desc().c_str(), rf, result.device());
		}
	}

	if (m_core->m_opCode != VGraphOpCode::add_all &&
		m_core->m_opCode != VGraphOpCode::parallel_all &&
		m_core->m_opCode != VGraphOpCode::stack_all) {

		result.keepBackpropOperandsInfo(needGrad, m_core->m_opCode, operands);
	}

	if (operands.size() > 0) {
		VDict args = operands[0].detachOpArgs();
		result.setMyArgs(args);
	}

	if (0) {
		static int nth = 0;
		result.dump("VGraphNode::" + GraphOpCodeName(m_core->m_opCode));
	}

	// 여기에서 역전파 콜백 설정 정보 가운데 출력 관계항을 텐서에 표시한다 pre
	//cbInfo.addOutputTensor(result, operands, needGrad);

	return result;
}

bool VGraphNode::m_lookupNeedGrad(VTensorDict xs) {
	for (auto& it : xs) {
		if (it.second.needGrad()) return true;
	}

	return false;
}

VTensor VGraphNode::m_createTensor(VValue value, VExecTracer tracer) {
	int nDevice = m_core->m_pGraphCore->m_device;

	if (value.is_float()) {
		VTensor tensor(session(), VShape{ 1 }, VDataType::float32, nDevice);
		tensor.fill_float((float)value, tracer);
		return tensor;
	}
	else if (value.is_int()) {
		VTensor tensor(session(), VShape{ 1 }, VDataType::int32, nDevice);
		tensor.fill_int((int)value, tracer);
		return tensor;
	}
	else {
		VP_THROW(VERR_CONDITIONAL_STATEMENT);
	}
}

string VGraphNode::GraphOpCodeName(VGraphOpCode opcode) {
	static const char* names[] = {
		"none", "input", "term", "pm", "pmset", "shape", "list", "_int", "_float", "_bool", "_user_defined_function", "str", "side",
		"flatten", "activate", "normal_noise", "uniform_noise", "normal_random", "uniform_random",
		"round", "codeconv", "cosinesim", "selectntop", "selectntoparg",
		"maxpool", "avgpool", "globalavg", "adaptiveavg", "stride", "upsample", "pass",
		"max", "min", "argmax", "argmin", "maximum", "minimum",
		"matmul", "conv2d", "conv2d_transposed", "conv2d_dilated",
		"add", "add_2d_bias", "add_residual", "subtract", "abs", "square", "reshape", "transpose", "embed",
		"rnn", "lstm", "gru",
		"batchnorm", "layernorm", "mh_attention", "dropout", "extract", "concat",
		"crossentropy", "crossentropy_sigmoid", "crossentropy_pos_idx", "mean", "sum",
		"merge_parallel_result",
		"parallel_all", "add_all", "stack_all", "parallel_concat",
		"mult", "se_mult", "div", "exp", "log", "and", "or", "not", "eq", "ne",
		"gt", "lt", "ge", "le", "gt_cross", "lt_cross", "ge_cross", "le_cross",
		"subvector", "pickup", "pickup_static", "to_filter", "to_int", "to_float", "to_tensor", "get_column", "hstack",
		//"iou_cross_best", "ciou_loss",
		"select_best_with_idx", "iou_cross_xywh", "iou_cross_lrtb", "ciou_loss", "diou_loss", "giou_loss", "iou_loss", "to_boxes",
		"complement_1",
		"sigmoid", "sigmoid_crossentropy_with_logits", "sigmoid_crossentropy_with_logits_idx",
	};

	if (sizeof(names) / sizeof(names[0]) != (int)VGraphOpCode::__end__) {
		VP_THROW(VERR_GRAPH_CODE_NAME);
	}

	return names[(int)opcode];
}

void VGraphNodeCore::m_setup() {
	m_pGraphCore = NULL;
}

void VGraphNodeCore::m_setup(string sExp, VList params, int& pos) {
	int nFrom = ms_skipSpaces(sExp, pos);

	for (; pos < sExp.size(); pos++) {
		if (strchr("() \t\r\n,:", sExp[pos])) break;
	}

	string opCode = sExp.substr(nFrom, pos - nFrom);

	if (opCode[0] == '#') m_opCode = VGraphOpCode::input;
	else if (opCode[0] == '%') m_opCode = VGraphOpCode::term;
	else if (vutils.isInt(opCode)) {
		m_opCode = VGraphOpCode::_int;
		m_aux = (int64)strtoll(opCode.c_str(), NULL, 0);
		m_varName = "";
	}
	else if (vutils.isFloat(opCode)) {
		m_opCode = VGraphOpCode::_float;
		m_aux = strtof(opCode.c_str(), NULL);
		m_varName = "";
	}
	else if (m_session.lookupUserDefinedFunctions(opCode, &m_aux)) {
		m_opCode = VGraphOpCode::_user_defined_function;
	}
	else m_opCode = VConsts::convToOpCode(opCode);

	ms_skipSpaces(sExp, pos);

	/*
	if (m_opCode == VGraphOpCode::none) {
		if (pos < sExp.size() && strchr(":(", sExp[pos])) VP_THROW(VERR_UNDEFINED);
		m_opCode = VGraphOpCode::input;
	}
	*/

	if (m_opCode == VGraphOpCode::input) m_varName = opCode;
	else if (m_opCode == VGraphOpCode::term) m_varName = opCode.substr(1);	// 선두의 '%' 표시는 떼어낸다.

	if (pos >= sExp.size()) return;

	if (sExp[pos] == ':') {
		pos++;
		nFrom = ms_skipSpaces(sExp, pos);
		for (; pos < sExp.size(); pos++) {
			if (strchr("() \t\r\n,:", sExp[pos])) break;
		}
		
		string sub_name = sExp.substr(nFrom, pos - nFrom);
		
		if (params.size() == 0) VP_THROW(VERR_SIZE_PARAMETER);

		VDict pmset, layer_params;

		switch (m_opCode) {
		case VGraphOpCode::pm:
			layer_params = params[0];
			if (layer_params.find(sub_name) != layer_params.end()) {
				m_nesterovPmSet = layer_params[sub_name];
				m_pm = VTensor(m_session, (VHTensor)m_nesterovPmSet["pm"]);
				if (sub_name != "mavg" && sub_name != "mvar") {
					m_grad = VTensor(m_session, (VHTensor)m_nesterovPmSet["grad"]);
				}
			}
			else {	// 무슨 이유로 예외 처리가 없었는지 확인 위해 추가
				// const.cpp 파일에 지정된 수식에 나타난 pm 항목이 실제 param에 없을 경우 발생
				if (m_pGraphCore->isParamAllocated()) VP_THROW(VERR_INVALID_DICT_KEY);
			}
			break;
		case VGraphOpCode::pmset:
			m_pmSet = params[0];
			m_varName = sub_name;
			break;
		case VGraphOpCode::shape:
			pmset = params[0];
			//hs.cho
			m_aux = pmset[sub_name];
			m_varName = sub_name;
			break;
		case VGraphOpCode::list:
			pmset = params[0];
			m_aux = (VList)pmset[sub_name];
			m_varName = sub_name;
			break;
		case VGraphOpCode::_int:
			pmset = params[0];
			m_aux = (int64)0;
			if (pmset.find(sub_name) != pmset.end()) {
				m_aux = (int64)pmset[sub_name];
			}
			m_varName = sub_name;
			break;
		case VGraphOpCode::_float:
			pmset = params[0];
			m_aux = (float)0;
			if (pmset.find(sub_name) != pmset.end()) {
				m_aux = (float)pmset[sub_name];
			}
			m_varName = sub_name;
			break;
		case VGraphOpCode::_bool:
			pmset = params[0];
			m_aux = (bool)pmset[sub_name];
			m_varName = sub_name;
			break;
		case VGraphOpCode::str:
			pmset = params[0];
			m_aux = (string)pmset[sub_name];
			m_varName = sub_name;
			break;
		case VGraphOpCode::side:
			pmset = params[0];
			m_aux = (string)pmset[sub_name];
			m_varName = sub_name;
			break;
		default:
			VP_THROW(VERR_CONDITIONAL_STATEMENT);
		}
	}
	else if (sExp[pos] == '(') {
		m_seekOperands(sExp, params, pos);
	}
}

void VGraphNodeCore::m_setup(VGraphNodeCore* src_core) {
	m_opCode = src_core->m_opCode;
	m_varName = src_core->m_varName;

	int nDevice = m_deviceMan.getCurDevice();

	m_grad = src_core->m_grad;					// 합산 대상이므로 공유하되 mutex를 이용해 동시 접근을 방지하기로 한다.
	m_pm = src_core->m_pm;
	m_pmSet = src_core->m_pmSet;

	//m_pGraphCore = src_core->m_pGraphCore;
	m_aux = src_core->m_aux;

	/*
	if (src_core->m_pm && src_core->m_pm.device() != nDevice) {
		m_pm = src_core->m_pm.toDevice(nDevice);	// 병렬로 수행되는 연산에 이용되므로 각 스레드 디바이스 메모리에 별도로 존재해야 한다.
	}
	else {
		m_pm = src_core->m_pm;	// 0번 스레드의 경우 어차피 혼자 쓰므로 복사 불필요
	}
	*/

	for (auto& it : src_core->m_children) {
		VGraphNode childClone(true, it, m_deviceMan, m_pGraphCore);
		m_children.push_back(childClone);	// 일단 리스트는 따로 구성하되 모듈까지는 복사하지 않기로 한다.
	}
}

void VGraphNodeCore::m_seekOperands(string layerExp, VList params, int& pos) {
	while (layerExp[pos] != ')') {
		if (!strchr("(,", layerExp[pos++])) VP_THROW(VERR_GRAPH_SEEK);
		ms_skipSpaces(layerExp, pos);
		VGraphNode child = VGraphNode(m_session, layerExp, params, pos, m_deviceMan, m_pGraphCore);
		m_children.push_back(child);
		ms_skipSpaces(layerExp, pos);
	}
	pos++;

	if (m_opCode == VGraphOpCode::add || m_opCode == VGraphOpCode::mult) {
		if (m_children.size() > 2) {
			VGraphNode lastChild = m_children.back();
			m_children.pop_back();
			VGraphNode newChild = m_split_multi_operands(this, m_opCode, m_children);
			m_children.clear();
			m_children.push_back(newChild);
			m_children.push_back(lastChild);
		}
	}
}

VGraphNode VGraphNodeCore::m_split_multi_operands(VGraphNodeCore* pSrc, VGraphOpCode opCode, vector<VGraphNode> children) {
	if (m_children.size() > 2) {
		VGraphNode lastChild = children.back();
		children.pop_back();
		VGraphNode newChild = m_split_multi_operands(pSrc, opCode, children);
		children.clear();
		children.push_back(newChild);
		children.push_back(lastChild);
	}

	VGraphNode node(pSrc, opCode, children);

	return node;
}

int VGraphNodeCore::ms_skipSpaces(string sExp, int& pos) {
	for (; pos < sExp.size(); pos++) {
		if (!strchr(" \t\r\n", sExp[pos])) break;
	}

	return pos;
}

VDict VGraphNodeCore::m_pmsetToCurDevice(VCbBackInfo cbInfo, VExecTracer tracer) {
	int curDevice = m_deviceMan.getCurDevice();
	
	if (curDevice < 0) return m_pmSet;;

	VDict pmset;

	for (auto& it : m_pmSet) {
		//pmset[it.first] = (VHTensor) 0; // it.second;

		if (!it.second.is_dict()) continue;

		VDict pmInfo = it.second;
		VTensor pm = VTensor(session(), (VHTensor)pmInfo["pm"]);

		if (pm.hasNoData()) {
			pmset[it.first] = pm.cloneCore();
			continue;
		}
		if (pm.device() == curDevice) return m_pmSet;

		// m_pm 텐서가 result 텐서로 바뀌는데 이 변환 정보를 역전파 콜백 슬롯 생성시 파악할 수 있어야 함!!!
		VTensor devTensor = pm.toDevice(curDevice, tracer);
		cbInfo.addDeviceConvInfo(pm.getNth(), devTensor);

		VTensor grad = VTensor(session(), (VHTensor)pmInfo["grad"]);
		devTensor.keepBackpropParamInfo(VGraphOpCode::pm, grad);

		devTensor = m_pGraphCore->optimizerPreproc(devTensor, m_nesterovPmSet, tracer); // nesterov Optimizer 등의 경우 parameter 이용 전에 옵티마이저 나름의 전처리 과정을 거친다.

		//((VDict)pmset[it.first])["pm"] = devTensor.cloneCore();
		pmset[it.first] = devTensor.cloneCore();
	}

	// m_pmSet = pmset;
	return pmset;
}