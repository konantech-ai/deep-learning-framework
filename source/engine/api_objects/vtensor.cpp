#include "../api/vconst.h"
#include "../api_objects/vtensor_core.h"
#include "../api_objects/vloss_core.h"
#include "../api_objects/vmodule_core.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vloss.h"
#include "../api_objects/vfunction.h"
#include "../local_objects/vexectracer_core.h"
#include "../local_objects/vgraph_node.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vcbbackslot.h"
#include "../support/vmath.h"
#include "../support/vback_queue.h"

int VTensorCore::ms_nCheckCode = 12104232;

//=========== API Object Common Part Start =======================

VTensor::VTensor() {
	m_core = NULL;
}

VTensor::VTensor(const VTensor& src) {
	m_core = src.m_core->clone();
}

VTensor::VTensor(VTensorCore* core) {
	m_core = core->clone();
}

VTensor::VTensor(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VTensorCore(session, sBuiltin, kwArgs);
}

VTensor::VTensor(VSession session, VHTensor handle) {
	m_core = NULL;
	VTensorCore* core = (VTensorCore*)handle;
	if (core == NULL) VP_THROW1(VERR_INVALID_CORE, "Tensor");
	if (core->m_nCheckCode != VTensorCore::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, "Tensor");
	if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, "Tensor");
	m_core = (VTensorCore*)core->clone_core();
}

VTensor::~VTensor() { m_core->destroy(); }

VTensor& VTensor::operator =(const VTensor& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VHTensor VTensor::cloneCore() {
	return (VHTensor)m_core->clone();
}

VHTensor VTensor::cloneHandle() {
	return (VHTensor)m_core->clone_handle();
}

VTensorCore* VTensor::getClone() {
	return (VTensorCore*)m_core->clone_core();
}

VTensorCore* VTensor::getCore() {
	return m_core;
}

bool VTensor::isValid() {
	return m_core != NULL;
}
void VTensor::closeHandle() {
	if (this) m_core->destroy_handle();
}

VSession VTensor::session() {
	return m_core->m_session;
}

int VTensor::getRefCnt() {
	return m_core->getRefCnt();
}

int VTensor::getNth() {
	return m_core->getNth();
}

void VTensor::incRefCount() {
	m_core->incRefCnt();
}

VTensorCore::VTensorCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Tensor) {
	m_nCheckCode = ms_nCheckCode;
	m_session = session;
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_onCreate();
}

VTensorCore::~VTensorCore() {
	m_onDelete();
	m_nCheckCode = 0;
}

//=========== API Object Common Part End =======================

//-----------------------------------------------------------------------------------------------------
// 캡슐 영역 확장 코드

bool VTensor::ms_bBlockDump = false;

mutex VTensor::ms_grad_mutex;
mutex VTensor::ms_dump_mutex;

VTensor::VTensor(VSession session) {	// 사용되지 않는 pm, grad 표현을 위한 빈 텐서
	m_core = new VTensorCore(session);
	m_core->m_data = NULL;
}

VTensor::VTensor(VSession session, VShape shape, VDataType type, int nDevice) {
	int64 size = shape.total_size() * VDataTypeSize(type);

	m_core = new VTensorCore(session);
	m_core->m_shape = shape;
	m_core->m_dataType = type;
	if (size > 0) m_core->m_data = VTensorData(session, size, nDevice);
}

VTensor::VTensor(VTensor src, VShape shape, TensorCloneInit init) {
	m_core = new VTensorCore(src.session());

	if (shape.total_size() != src.shape().total_size()) VP_THROW(VERR_SIZE_TENSOR);

	m_core->m_shape = shape;
	m_core->m_dataType = src.type();

	switch (init) {
	case TensorCloneInit::share:
		m_core->m_data = src.m_core->m_data;
		break;
	case TensorCloneInit::alloc:
		m_core->m_data = VTensorData(session(), src.byte_size(), src.device());
		break;
	case TensorCloneInit::zeros:
		m_core->m_data = VTensorData(session(), src.byte_size(), src.device());
		//m_core->m_data.memset(0);
		break;
		/*
	case TensorCloneInit::copy:
		m_core->m_data = VTensorData(session(), src.byte_size(), src.device());
		m_core->m_data.copyFrom(src.data());
		break;
		*/
	case TensorCloneInit::empty:
		break;
	}
}

VShape VTensor::shape() {
	return m_core->m_shape;
}

VDataType VTensor::type() {
	return m_core->m_dataType;
}

bool VTensor::needGrad() {
	return m_core && m_core->m_bNeedGrad && m_core->m_data.isValid();
}

void VTensor::setNeedGrad(bool needGrad) {
	if (m_core->m_data.isValid()) m_core->m_bNeedGrad = needGrad;
}

bool VTensor::hasData() {
	return m_core->m_data.isValid();
}

bool VTensor::hasNoData() {
	return !m_core->m_data.isValid();
}

VTensorData VTensor::data() {
	return m_core->m_data;
}

int VTensor::device() {
	return m_core->m_data.device();
}

int64 VTensor::byte_size() {
	return m_core->m_data.byte_size();
}

void* VTensor::void_ptr() {
	return m_core->m_data.void_ptr();
}

int* VTensor::int_ptr() {
	if (m_core->m_dataType != VDataType::int32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(m_core->m_dataType));
	}
	return m_core->m_data.int_ptr();
}

int64* VTensor::int64_ptr() {
	if (m_core->m_dataType != VDataType::int64) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int64", VDataTypeName(m_core->m_dataType));
	}
	return m_core->m_data.int64_ptr();
}

float* VTensor::float_ptr() {
	if (m_core->m_dataType != VDataType::float32) {
		printf("VP1: shape = %s\n", shape().desc().c_str());
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}
	return m_core->m_data.float_ptr();
}

unsigned char* VTensor::uchar_ptr() {
	if (m_core->m_dataType != VDataType::uint8) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "uint8", VDataTypeName(m_core->m_dataType));
	}
	return m_core->m_data.uchar_ptr();
}

void VTensor::setFeature(VShape shape, VDataType type, int nDevice) {
	if (shape.size() == 0 && nDevice != -2) {
		// nDevice == -2: data 없는 보조정보용 텐서를 의미

		m_core->m_shape.copyInto(shape);
		m_core->m_dataType = type;

		return;
	}

	bool diff = (shape != m_core->m_shape) || (type != m_core->m_dataType) ||
				(!m_core->m_data.isValid()) || (nDevice != m_core->m_data.device());

    if (!diff) return;

    int64 size = shape.total_size() * VDataTypeSize(type);

	if (size == 0) VP_THROW(VERR_SIZE_TENSOR);

    m_core->m_shape.copyInto(shape);
    m_core->m_dataType = type;

    if (!m_core->m_data.isValid() || size != m_core->m_data.byte_size()) {
        m_core->m_data = VTensorData(session(), size, nDevice);
    }
}

void VTensor::allocData(int nDevice) {
	int64 size = shape().total_size() * VDataTypeSize(type());

	if (m_core->m_data.isValid() || size == 0) return;

	m_core->m_data = VTensorData(session(), size, nDevice);
}

void VTensor::setZero(VExecTracer tracer) {
	if (m_core->m_data.isValid()) {
		void* pData = void_ptr();
		int64 size = byte_size();

		CALL_MATH(set_zero, device(), pData, size);
		//m_core->m_data.setZero(tracer);
	}
}

void VTensor::initParam(TensorInitMethod init_op, float mean, float init_arg, bool adaptive) {
	m_core->m_initParam(init_op, mean, init_arg, adaptive);
	m_core->m_initArgs = VDict({ {"init_op", (int)init_op}, {"mean", mean}, {"init_arg", init_arg}, {"adaptive", adaptive} });
}

void VTensor::initParam() {
	if (m_core->m_initArgs.size() == 0) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);

	TensorInitMethod init_op = (TensorInitMethod)(int)m_core->m_initArgs["init_op"];

	float mean = m_core->m_initArgs["mean"];
	float init_arg = m_core->m_initArgs["init_arg"];
	bool adaptive = m_core->m_initArgs["adaptive"];

	m_core->m_initParam(init_op, mean, init_arg, adaptive);
}

void VTensor::getFeature(VShape* pshape, VDataType* ptype, int* pnDevice) {
	if (pshape) pshape->copyInto(m_core->m_shape);
	if (ptype) *ptype = m_core->m_dataType;
	if (pnDevice) *pnDevice = m_core->m_data.device();
}

void VTensor::uploadData(void* pData, int64 nByteSize) {
	m_core->m_data.uploadData(pData, nByteSize);
}

void VTensor::downloadData(void* pData, int64 nByteSize) {
	m_core->m_data.downloadData(pData, nByteSize);
}

void VTensor::setLossFunc(VLossCore* pLossCore) {
	// clone_core() 호출은 순환참조에 의한 메모리 릭 발생의 원인이 되는 것으로 보임
	// 이에 따라 clone_core() 대신 ref# 증가 없이 코어를 직접 저장하도록 수정해 봄
	// 에 때 dict 삭제 과정에서의 ref# 감소로 인한 오류 발생을 막기 위해 int64 형변환을 이용함
	// 이에 따라 삭제된 loss function에 접근하려는 로직 오류가 발생할 가능성이 있으므로 요주의
	// 
	//m_core->m_propDict["__loss_fn__"] = (VObjCore*)pLossCore->clone_core();
	m_core->m_propDict["__loss_fn__"] = (int64)(VObjCore*)pLossCore;
}

void VTensor::setCbBackSlot(VCbBackSlot slot) {
	//m_core->m_cbBackSlots.push_back(slot.getCore());
	m_core->m_cbBackSlots.push_back(slot);
}

void VTensor::resetCbBackSlot(int sid) {
	m_core->m_resetCbBackSlot(sid);
}

void VTensor::backward() {
	if (m_core->m_propDict.find("__loss_fn__") == m_core->m_propDict.end()) VP_THROW(VERR_INVALID_DICT_KEY);
	VLossCore* pLossCore = (VLossCore*)(int64)m_core->m_propDict["__loss_fn__"];
	VLoss loss_fn(pLossCore);
	loss_fn.backward();

	/*
	VBackQueue queue(session(), VExecTracer());
	
	queue.regist(*this, grad);

	while (!queue.isEnd()) {
		queue.step();
	}
	*/
}

void VTensor::backwardWithGradient(VTensor grad) {
	// VLoss::backward()에서 복사해 온 아래 코드가 주어진 grad 갖고 작동하도록 수정 필요
	
	// 트레이서 설정은 일단 보류
	/*
	VExecTracer tracer = m_core->m_>m_execTracer[1];

	if (!tracer.isValid() && !session().getNoTracer()) {
		tracer = VExecTracer(session(), "loss backward", {});
		m_core->m_execTracer[1] = tracer;
	}
	else if (tracer.isValid() && session().getNoTracer()) {
		tracer = VExecTracer();
		m_core->m_execTracer[1] = tracer;
	}

	VTensorDict lossTensors = m_core->m_losses;
	VTensorDict predTensors = m_core->m_preds;

	VTensorDict xs = vutils.mergeTensorDict(lossTensors, predTensors);

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
	*/

	VExecTracer tracer;

	VTensor pred = *this;

	//pred.dump("pred");
	//printf("pred: device = %d\n", pred.device());
	//grad = grad.toDevice(pred.device(), tracer);
	//grad = grad.toDevice(0, tracer);

	VBackQueue queue(session(), tracer);

	/*
	for (auto& it : lossTensors) {
		queue.regist(it.second, VTensor());
	}

	queue.registCrossNodes(m_core->m_preds);
	*/

	queue.regist(pred, grad);

	queue.registCrossNodes({ {"#default", pred} });

	while (!queue.isEnd()) {
		queue.step();
	}

	tracer.closeRecording({});
}

bool VTensor::is_pm() {
	return m_core->m_opCode == VGraphOpCode::pm;
}

string VTensor::getOpName() {
	return VGraphNode::GraphOpCodeName(m_core->m_opCode);
}

void VTensor::pm_backprop(VTensor grad, VExecTracer tracer) {
	VTensor pmGrad = m_core->m_operands[0];

	int64 nrow = pmGrad.shape().total_size();

	tracer.addTensor(grad);
	tracer.addTensor(pmGrad);

	float* pgy = grad.float_ptr();
	float* pgm = pmGrad.float_ptr();

	int device1 = pmGrad.device();
	int device2 = grad.device();

	VMath::cudaCheck((cudaError_t)0, "WP1", __FILE__, __LINE__);

	VMath::cudaCheck((cudaError_t)0, "WP2", __FILE__, __LINE__);

	if (device1 < 0 && device2 < 0) {
		CALL_MATH(accumulate_grad, -1, pgm, pgy, nrow);
	}
	else if (device1 == device2) {
		ms_grad_mutex.lock();
		CALL_MATH(accumulate_grad, device1, pgm, pgy, nrow);
		ms_grad_mutex.unlock();
	}
	else if (device1 < 0 || device2 < 0) {
		VP_THROW(VERR_UNKNOWN);
	}
	else {
		//ms_grad_mutex.lock();
		VTensor ygrad_0 = grad.toDevice(device1, tracer);
		pgy = ygrad_0.float_ptr();
		ms_grad_mutex.lock();
		int nOldDevice = session().device_man().setCurDevice(device1, tracer);
		CALL_MATH(accumulate_grad, device1, pgm, pgy, nrow);
		ms_grad_mutex.unlock();
		session().device_man().setCurDevice(nOldDevice, tracer);
	}
}

void VTensor::accGrad(VTensor grad, VExecTracer tracer) {
	VTensor me = *this;

	int64 nrow = me.shape().total_size();

	if (grad.shape().total_size() != nrow) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	if (me.type() != VDataType::float32 || grad.type() != VDataType::float32) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	if (me.device() != grad.device()) VP_THROW(VERR_NOT_IMPLEMENTED_YET);

	tracer.addTensor(me);
	tracer.addTensor(grad);

	float* pm = me.float_ptr();
	float* pg = grad.float_ptr();

	if (me.device() < 0) {
		CALL_MATH(accumulate_grad, -1, pm, pg, nrow);
	}
	else {
		ms_grad_mutex.lock();
		CALL_MATH(accumulate_grad, me.device(), pm, pg, nrow);
		ms_grad_mutex.unlock();
	}
}

void VTensor::backprop_noGgrad(VBackQueue* pQueue) {
	for (auto& it : m_core->m_operands) {
		pQueue->registNoGrad(it);
	}
}

void VTensor::invokeBackpropCallback(VTensor grad, VExecTracer tracer) {
	if (m_core->m_cbBackSlots.size() == 0) return;

	for (auto& slot : m_core->m_cbBackSlots) {
		slot.fillAndInvokeOnFull(*this, grad, tracer);
	}
}

void VTensor::backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	if (0) {
		printf("backprop: %s called\n", VGraphNode::GraphOpCodeName(m_core->m_opCode).c_str());
	}

	switch (m_core->m_opCode) {
	case VGraphOpCode::none:
		break;
	case VGraphOpCode::_user_defined_function:
		user_defined_func_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::mean:
		mean_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::sum:
		sum_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::merge_parallel_result:
	{
		m_core->m_pOwningModule->parallel_backprop(ygrad, m_core->m_operands, pQueue, tracer);
	}
		break;
	case VGraphOpCode::add:
		add_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::add_2d_bias:
		add_2d_bias_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::add_residual:
		add_residual_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::subtract:
		subtract_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::mult:
		mult_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::se_mult:
		se_mult_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::div:
		div_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::maximum:
		maximum_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::minimum:
		minimum_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::matmul:
		matmul_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::conv2d:
		conv2d_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::conv2d_transposed:
		conv2d_transposed_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::conv2d_dilated:
		conv2d_dilated_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::activate:
		activate_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::normal_noise:
		normal_noise_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::uniform_noise:
		uniform_noise_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::normal_random:
		normal_random_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::uniform_random:
		uniform_random_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::round:
		round_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::codeconv:
		codeconv_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::maxpool:
		maxpool_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::avgpool:
		avgpool_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::globalavg:
		globalavg_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::adaptiveavg:
		adaptiveavg_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::stride:
		stride_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::rnn:
		rnn_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::lstm:
		lstm_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::gru:
		gru_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::abs:
		abs_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::square:
		square_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::exp:
		exp_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::log:
		log_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::sigmoid:
		sigmoid_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::extract:
		extract_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::upsample:
		upsample_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::dropout:
		dropout_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::batchnorm:
		batchnorm_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::layernorm:
		layernorm_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::mh_attention:
		multihead_attention_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::embed:
		embed_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::subvector:
		subvector_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::pickup:
		pickup_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::concat:
	case VGraphOpCode::hstack:
		concat_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::transpose:
		transpose_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::reshape:
		reshape_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::flatten:
		flatten_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::pass:
		pass_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::crossentropy:
		crossentropy_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::crossentropy_sigmoid:
		crossentropy_sigmoid_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::crossentropy_pos_idx:
		crossentropy_pos_idx_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::ciou_loss:
		iou_loss_backprop(ygrad, pQueue, m_core->m_opCode, tracer);
		break;
	case VGraphOpCode::complement_1:
		complement_1_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::sigmoid_crossentropy_with_logits:
		sigmoid_crossentropy_with_logits_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::sigmoid_crossentropy_with_logits_idx:
		sigmoid_crossentropy_with_logits_idx_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::parallel_concat:
		parallel_concat_backprop(ygrad, pQueue, tracer);
		break;
	case VGraphOpCode::stack_all:
		stack_all_backprop(ygrad, pQueue, tracer);
		break;
	default:
		VP_THROW1(VERR_CONDITIONAL_STATEMENT, to_string((int)m_core->m_opCode));
		break;
	}

	//m_core->m_operands.clear();
}

VTensor VTensor::getNthOperand(int nth) {
	if (m_core == NULL || m_core->m_operands.size() <= 0) {
		return VTensor();
	}
	return m_core->m_operands[nth];
}

void VTensor::dump1(string title, bool full) {
	if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
	printf("T#%s%s:%s = [", title.c_str(), shape().desc().c_str(), VDataTypeName(type()).c_str());

	void* pHostBuf = void_ptr();

	if (device() >= 0) {
		pHostBuf = malloc(byte_size());
		cudaMemcpy(pHostBuf, void_ptr(), byte_size(), cudaMemcpyDeviceToHost);
	}

	if (shape().total_size() <= 120) {
		for (int64 n = 0; n < shape().total_size(); n++) {
			if (n % 10 == 0) printf("\n      ");
			m_dumpNthElement(pHostBuf, n, true);
		}
	}
	else {
		for (int64 n = 0; n < 50; n++) {
			if (n % 10 == 0) printf("\n      ");
			m_dumpNthElement(pHostBuf, n, true);
		}
		printf("\n      ...");
		for (int64 k = 0, n = shape().total_size() / 2 - 10; n < shape().total_size() / 2 + 10; n++, k++) {
			if (k % 10 == 0) printf("\n      ");
			m_dumpNthElement(pHostBuf, n, true);
		}
		printf("\n      ...");
		for (int64 n = (shape().total_size() - 50) / 10 * 10; n < shape().total_size(); n++) {
			if (n % 10 == 0) printf("\n      ");
			m_dumpNthElement(pHostBuf, n, true);
		}
	}
	printf("]\n");

	if (device() >= 0) {
		free(pHostBuf);
	}
}

void VTensor::dump(string title, bool full) {
	if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
	printf("T#%d:%s%s:%s(dev:%d)", getNth(), title.c_str(), shape().desc().c_str(), VDataTypeName(type()).c_str(), device());

	if (shape().total_size() > 0) {
		void* pHostBuf = void_ptr();

		if (device() >= 0) {
			pHostBuf = malloc(byte_size());
			cudaMemcpy(pHostBuf, void_ptr(), byte_size(), cudaMemcpyDeviceToHost);
		}

		if (shape().total_size() > 12) printf("\n    ");

		m_dump(shape(), pHostBuf, 0, 4, full);

		if (device() >= 0) {
			free(pHostBuf);
		}
	}

	printf("\n");
}

void VTensor::m_dump(VShape shape, void* pHostBuf, int64 nth, int indent, bool full) {
	VShape tshape = shape.remove_head();

	int64 ax_size = shape[0];
	int64 child_size = tshape.total_size();

	if (shape.size() == 0) return;
	else if (shape.size() == 1) {
		printf("[");
		if (full || ax_size <= 7) {
			for (int64 n = 0; n < ax_size; n++) m_dumpNthElement(pHostBuf, nth++, n > 0);
		}
		else {
			for (int64 n = 0; n < 3; n++) m_dumpNthElement(pHostBuf, nth++, n > 0);
			nth += ax_size - 6;
			printf(" ...");
			for (int64 n = 0; n < 3; n++) m_dumpNthElement(pHostBuf, nth++);
		}
		printf("]");
	}
	else {
		bool needNewLine = shape.total_size() > 12;
		printf("[");
		if (full || ax_size <= 5) {
			for (int64 n = 0; n < ax_size; n++) {
				if (n > 0 && needNewLine) {
					printf("\n%*s", indent, "");
				}
				m_dump(tshape, pHostBuf, nth, indent + 2, full);
				nth += child_size;
			}
		}
		else {
			m_dump(tshape, pHostBuf, nth, indent + 2, false);
			nth += child_size;
			if (needNewLine) printf("\n%*s", indent, "");
			m_dump(tshape, pHostBuf, nth, indent + 2, false);
			if (needNewLine) printf("\n%*s", indent, "");
			nth += child_size;

			nth += (ax_size - 4) * child_size;
			printf("...");

			if (needNewLine) printf("\n%*s", indent, "");
			m_dump(tshape, pHostBuf, nth, indent + 2, false);
			nth += child_size;
			if (needNewLine) printf("\n%*s", indent, "");
			m_dump(tshape, pHostBuf, nth, indent + 2, false);
		}
		printf("]");
	}
}

void VTensor::m_dumpNthElement(void* pHostBuf, int64 n, bool bSpaceOnLeft) {
	if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
	if (bSpaceOnLeft) printf(" ");
	printf("%s", m_getElementDesc(pHostBuf, n).c_str());
}

string VTensor::m_getElementDesc(void* pHostBuf, int64 n) {
	char buffer[128];

	switch (type()) {
	case VDataType::float32:
		snprintf(buffer, 128, "%f", ((float*)pHostBuf)[n]);
		break;
	case VDataType::int32:
		snprintf(buffer, 128, "%d", ((int*)pHostBuf)[n]);
		break;
	case VDataType::bool8:
		snprintf(buffer, 128, "%s", ((unsigned char*)pHostBuf)[n] ? "True" : "False");
		break;
	case VDataType::uint8:
		snprintf(buffer, 128, "%d", ((unsigned char*)pHostBuf)[n]);
		break;
	case VDataType::int64:
		snprintf(buffer, 128, "%lld", ((int64*)pHostBuf)[n]);
		break;
	default:
		TP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	return string(buffer);
}

/*
void VTensor::dump(string title) {
	if (ms_bBlockDump) return;

	ms_dump_mutex.lock();

	printf("[%s#%d] shape:%s, type:%d, device:%d", title.c_str(), getNth(), shape().desc().c_str(), (int)type(), device());

	void* pHostBuf = VMath::mem_alloc(-1, shape().total_size() * VDataTypeSize(type()));

	if (device() < 0) VMath::memcpy_host_to_host(pHostBuf, void_ptr(), m_core->m_data.byte_size());
	else VMath::memcpy_device_to_host(pHostBuf, void_ptr(), m_core->m_data.byte_size());

	if (shape().total_size() < 100) {
		for (int64 n = 0; n < shape().total_size(); n++) {
			if (n % 10 == 0) printf("\n    ");
			m_dumpElement(pHostBuf, n);
		}
		printf("\n");
	}
	else {
		for (int64 n = 0; n < 50; n++) {
			if (n % 10 == 0) printf("\n    ");
			m_dumpElement(pHostBuf, n);
		}
		printf("\n   ...");
		for (int64 n = 0; n < 50; n++) {
			if (n % 10 == 0) printf("\n    ");
			m_dumpElement(pHostBuf, shape().total_size() - 50 + n);
		}
		printf("\n");
	}

	VMath::mem_free(-1, pHostBuf);

	ms_dump_mutex.unlock();
}

void VTensor::m_dumpElement(void* pHostBuf, int64 n) {
	switch (type()) {
	case VDataType::float32:
		printf("%g ", ((float*)pHostBuf)[n]);
		break;
	case VDataType::int32:
		printf("%d ", ((int*)pHostBuf)[n]);
		break;
	case VDataType::int64:
		printf("%lld ", ((int64*)pHostBuf)[n]);
		break;
	case VDataType::bool8:
		printf("%s ", ((unsigned char*)pHostBuf)[n] ? "True" : "False");
		break;
	case VDataType::uint8:
		printf("%d ", ((unsigned int*)pHostBuf)[n]);
		break;
	}
}
*/

void VTensor::dump_arr_feat(int nth, string title) {
	VTensor x = *this;
	float sum = 0, dsqsum = 0;
	int64 minpos = 0, maxpos = 0;

	if (!x.isValid()) {
		printf("%d\t%s: empty\n", nth, title.c_str());
		return;
	}

	int64 size = x.shape().total_size();
	
	printf("%d\t%s[T#%d]\t%s\tdevice:%d\n",
		nth, title.c_str(), x.getNth(), x.shape().desc().c_str(), x.device());

	x = x.toDevice(-1, VExecTracer());

	if (x.type() == VDataType::float32) {
		float* pBuffer = x.float_ptr();
		float first, second, last, min, max, avg, std;

		first = min = max = pBuffer[0];
		second = (size > 1) ? pBuffer[1] : 0;
		last = pBuffer[size - 1];
		for (int64 n = 0; n < size; n++) {
			if (pBuffer[n] > max) {
				max = pBuffer[n];
				maxpos = n;
			}
			if (pBuffer[n] < min) {
				min = pBuffer[n];
				minpos = n;
			}
			sum += pBuffer[n];
		}
		avg = sum / (float)size;

		for (int64 n = 0; n < size; n++) {
			dsqsum += (pBuffer[n] - avg) * (pBuffer[n] - avg);
		}

		avg = sum / (float)size;
		std = ::sqrtf(dsqsum / (float)size + 1e-10f);

		printf("\tfirst:%g\tsecond:%g\tlast:%g\tmin:%g(%lld)\tmax:%g(%lld)\tavg:%g\tstd:%g\n",
			first, second, last, min, minpos, max, maxpos, avg, std);

		int histogram[11];
		int nout = 0;

		memset(histogram, 0, 11 * sizeof(int));

		if (min < max) {
			for (int64 n = 0; n < size; n++) {
				if (pBuffer[n] == min) histogram[0]++;
				else if (pBuffer[n] == max) histogram[10]++;
				else {
					int nth = (int)((pBuffer[n] - min) * 9 / (max - min));
					if (nth >= 0 && nth < 10) histogram[nth]++;
					else nout++;
				}
			}
		}
		else {
			histogram[0] = (int)size;
		}
		printf("\t\t\tHistogram:");
		for (int64 n = 0; n < 10; n++) {
			printf(" %d", histogram[n]);
		}
		printf(" [out: %d]\n", nout);
	}
	else if (x.type() == VDataType::int32) {
		int* pBuffer = x.int_ptr();
		int first, second, last, min, max;
		float avg, std;

		first = min = max = pBuffer[0];
		second = (size > 1) ? pBuffer[1] : 0;
		last = pBuffer[size - 1];
		for (int64 n = 0; n < size; n++) {
			if (pBuffer[n] > max) {
				max = pBuffer[n];
				maxpos = n;
			}
			if (pBuffer[n] < min) {
				min = pBuffer[n];
				minpos = n;
			}
			sum += pBuffer[n];
		}
		avg = sum / (float)size;
		for (int64 n = 0; n < size; n++) {
			dsqsum += (pBuffer[n] - avg) * (pBuffer[n] - avg);
		}
		std = ::sqrtf(dsqsum / size +1e-10f);
		printf("\tfirst:%d\tsecond:%d\tlast:%d\tmin:%d(%lld)\tmax:%d(%lld)\tavg:%g\tstd:%g\n",
			first, second, last, min, minpos, max, maxpos, avg, std);

		int histogram[11];
		int nout = 0;

		memset(histogram, 0, 11 * sizeof(int));

		if (min < max) {
			for (int64 n = 0; n < size; n++) {
				if (pBuffer[n] == min) histogram[0]++;
				else if (pBuffer[n] == max) histogram[10]++;
				else {
					int nth = (int)((pBuffer[n] - min) * 9 / (max - min));
					if (nth >= 0 && nth < 10) histogram[nth]++;
					else nout++;
				}
			}
		}
		else {
			histogram[0] = (int)size;
		}
		printf("\t\t\tHistogram:");
		for (int64 n = 0; n < 10; n++) {
			printf(" %d", histogram[n]);
		}
		printf(" [out: %d]\n", nout);
	}
}

VTensor VTensor::getNthSlice(int nDivisions, int nth, int nDevice, VExecTracer tracer) {
	tracer.addTensor(*this);

	VShape src_shape = shape();
	VDataType dat_type = type();

	int64 src_batch_size = src_shape[0];
	int64 max_slice_size = (src_batch_size + nDivisions - 1) / nDivisions;
	int64 slice_from = max_slice_size * nth;
	int64 slice_to = MIN(slice_from + max_slice_size, src_batch_size);
	int64 slice_size = slice_to - slice_from;

	if (slice_size <= 0) return VTensor();

	VShape slice_shape = src_shape.replace_nth(0, slice_size);

	VTensor slice = tracer.createTensor(session(), slice_shape, dat_type, nDevice);

	int64 data_size = src_shape.total_size() / src_batch_size;
	int64 skip_size = data_size * slice_from;

	void* pSrc = (unsigned char*)void_ptr() + skip_size * VDataTypeSize(dat_type);
	void* pDst = slice.void_ptr();

	CALL_MATH(get_slice, device(), pDst, pSrc, slice.byte_size());

	return slice;
}

VTensor VTensor::getSlicePiece(int64 slice_from, int64 slice_size, int nDevice, VExecTracer tracer) {
	int64 slice_to = slice_from + slice_size;

	if (slice_size <= 0) return VTensor();

	VShape sshape = shape().replace_nth(0, slice_size);

	VTensor slice = tracer.createTensor(session(), sshape, type(), nDevice);

	int64 skip_size = (shape().total_size() / shape()[0]) * slice_from;

	float* pSrc = float_ptr() + skip_size;
	float* pDst = slice.float_ptr();

	CALL_MATH(memcpy_to_device, device(), pDst, pSrc, slice.byte_size());

	/*
	if (device() < 0) {
		VMath::memcpy_host_to_device(slice.float_ptr(), pSrc, slice.byte_size());
	}
	else {
		VMath::memcpy_device_to_device(slice.float_ptr(), pSrc, slice.byte_size());
	}
	*/

	return slice;
}

void VTensor::setElement(VList pos, VValue value, VExecTracer tracer) {
	int64 index = 0;

	if (pos.size() > 0) {
		VShape xshape = shape();

		if (pos.size() != xshape.size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

		for (int n = 0; n < pos.size(); n++) {
			int64 npos = pos[n];
			if (npos < 0 || npos >= xshape[n]) VP_THROW(VERR_OUT_OF_RANGE);
			index = index * xshape[n] + npos;
		}
	}

	if (type() == VDataType::int32) {
		int nValue = value;
		int* pDst = int_ptr() + index;
		int* pSrc = &nValue;
		int64 size = sizeof(int);

		CALL_MATH(memcpy_from_host, device(), pDst, pSrc, size);
		/*
		if (device() < 0) {
			VMath::memcpy_host_to_host(pElem, &nValue, sizeof(int));
		}
		else {
			VMath::memcpy_host_to_device(pElem, &nValue, sizeof(int));
		}
		*/
	}
	else {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
}

VValue VTensor::getElement(VList pos, VExecTracer tracer) {
	int64 index = 0;

	if (pos.size() > 0) {
		VShape xshape = shape();

		if (pos.size() != xshape.size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

		for (int n = 0; n < pos.size(); n++) {
			int64 npos = pos[n];
			if (npos < 0 || npos >= xshape[n]) VP_THROW(VERR_OUT_OF_RANGE);
			index = index * xshape[n] + npos;
		}
	}

	if (type() == VDataType::int32) {
		int nValue;
		int* pDst = &nValue;
		int* pSrc = int_ptr() + index;
		int64 size = sizeof(int);

		CALL_MATH(memcpy_to_host, device(), pDst, pSrc, size);

		return nValue;
	}
	else if (type() == VDataType::float32) {
		float fValue;
		float* pDst = &fValue;
		float* pSrc = float_ptr() + index;
		int64 size = sizeof(float);

		CALL_MATH(memcpy_to_host, device(), pDst, pSrc, size);

		return fValue;
	}
	else {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
}

VTensor VTensor::toDevice(int nDevice, VExecTracer tracer) {
	if (device() == nDevice) return (*this);

	VShape tshape = shape().copy();

	VTensor clone = tracer.createTensor(session(), tshape, type(), nDevice);

	clone.m_core->m_copyDataFrom(m_core, tracer);
	clone.setNeedGrad(needGrad());

	return clone;
}

int64 VTensor::copySliceFrom(VTensor slice, int64 startRow, VExecTracer tracer) {
	int64 skip_size = (shape().total_size() / shape()[0]) * startRow;
	int64 copy_size = slice.byte_size();

	if (type() != slice.type()) VP_THROW(VERR_UNDEFINED);

	void* pDst = (char*)void_ptr() + skip_size * VDataTypeSize(type());
	void* pSrc = slice.void_ptr();

	CALL_MATH(get_slice, device(), pDst, pSrc, copy_size);

	return startRow + slice.shape()[0];
}

void VTensor::setOpArgs(VDict args) {
	m_core->m_opArgs = args;
}

void VTensor::setMyArgs(VDict args) {
	m_core->m_myArgs = args;
}

void VTensor::setOpArg(string name, VValue value) {
	m_core->m_opArgs[name] = value;
}

VDict VTensor::getOpArgs() {
	return m_core->m_opArgs;
}

VDict VTensor::getMyArgs() {
	return m_core->m_myArgs;
}

VDict VTensor::detachOpArgs() {
	VDict args = m_core->m_opArgs;
	m_core->m_opArgs = VDict();
	return args;
}

VValue VTensor::getOpArg(string name, VValue def) {
	if (m_core->m_opArgs.find(name) != m_core->m_opArgs.end()) return m_core->m_opArgs[name];
	if (def.is_none()) {
		printf("BP1: getOpArg(%s) failed\n", name.c_str());
		for (auto& it : m_core->m_opArgs) {
			printf("BP1: m_opArgs(%s) exists\n", it.first.c_str());
		}
		VP_THROW(VERR_TENSOR_DATATYPE);
	}
	return def;
}

VValue VTensor::getMyArg(string name) {
	return m_core->m_myArgs[name];
}

VValue VTensor::getOpArg(string name, int nth, VValue def) {
	if (m_core->m_opArgs.find(name) != m_core->m_opArgs.end()) return m_core->m_opArgs[name];
	if (m_core->m_opArgs.find(to_string(nth)) != m_core->m_opArgs.end()) return m_core->m_opArgs[to_string(nth)];
	if (def.is_none()) {
		printf("BP1: getOpArg(%s) failed\n", name.c_str());
		for (auto& it : m_core->m_opArgs) {
			printf("BP1: m_opArgs(%s) exists\n", it.first.c_str());
		}
		VP_THROW(VERR_TENSOR_DATATYPE);
	}
	return def;
}

VValue VTensor::getMyArg(string name, int nth) {
	if (m_core->m_myArgs.find(name) != m_core->m_myArgs.end()) return m_core->m_myArgs[name];
	return m_core->m_myArgs[to_string(nth)];
}

VTensor VTensor::flatten(VExecTracer tracer) {
	if (shape().size() <= 2) {
		return *this;
	}
	else {
		int64 mb_size = shape()[0];
		int64 dat_size = shape().total_size() / mb_size;
		VShape fshape = { mb_size, dat_size };

		return VTensor(*this, fshape, TensorCloneInit::share);
	}
}

void VTensor::fill_float(float value, VExecTracer tracer) {
	float* px = float_ptr();

	int64 size = shape().total_size();

	CALL_MATH(fill_float, device(), px, size, value);
}

void VTensor::fill_int(int value, VExecTracer tracer) {
	int* px = int_ptr();
	int64 size = shape().total_size();

	CALL_MATH(fill_int, device(), px, size, value);
}

VTensor VTensor::matmul(VTensor w, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape wshape = w.shape();

	if (xshape.size() < 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (wshape.size() != 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (device() != w.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 yvec = wshape[0];
	int64 xvec = wshape[1];
	int64 ndat = xshape.total_size() / xvec;

	//if (xshape.total_size() % xvec != 0) VP_THROW(VERR_UNDEFINED);

	VShape yshape = xshape;
	
	while (true) {
		yshape = yshape.remove_end();
		if (yshape.total_size() == ndat) break;
		if (yshape.total_size() < ndat) VP_THROW(VERR_UNDEFINED);
	}
	
	yshape = yshape.append(yvec);

	//printf("[AFTER] yvec:%lld, xvec:%lld, ndat:%lld, yshape:%s\n", yvec, xvec, ndat, yshape.desc().c_str());
	if (0) {	// 수정전
		int64 xvec = xshape[-1];
		int64 yvec = wshape[0];
		int64 ndat = xshape.total_size() / xvec;

	if (wshape[1] != xvec) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

		if (1) {
			printf("xshape: %s\n", xshape.desc().c_str());
			printf("wshape: %s\n", wshape.desc().c_str());
		}

		VShape yshape = xshape.replace_end(yvec);

		printf("[BEFORE] yvec:%lld, xvec:%lld, ndat:%lld, yshape:%s\n", yvec, xvec, ndat, yshape.desc().c_str());
	}

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	float* pw = w.float_ptr();

	CALL_MATH(matmul, device(), py, pw, px, yvec, ndat, xvec, false);

	return y;
}

void VTensor::matmul_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	VTensor w = m_core->m_operands[1];

	float* pgy = ygrad.float_ptr();

	int64 yvec = w.shape()[0];
	int64 xvec = w.shape()[1];

	int64 ndat = x.shape().total_size() / xvec;

	if (x.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pw = w.float_ptr();

		CALL_MATH(matmul_backward_x, device(), pgx, pgy, pw, yvec, ndat, xvec, false);

		pQueue->regist(x, xgrad);
	}
	else{
		pQueue->registNoGrad(x);
	}

	if (w.needGrad()) {
		VTensor wgrad = tracer.createTensor(session(), w.shape(), type(), device());

		float* pgw = wgrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(matmul_backward_w, device(), pgw, pgy, px, yvec, ndat, xvec, false);

		pQueue->regist(w, wgrad);
	}
	else {
		pQueue->registNoGrad(w);
	}
}

VTensor VTensor::conv2d(VTensor k, VExecTracer tracer) {
	VTensor x = *this;

	if (k.device() == -1) {
		printf("k.device() = %d\n", k.device());
		VP_THROW(VERR_TENSOR_DEVICE);
	}

	VShape xshape = x.shape();
	VShape kshape = k.shape();
	
	VShape pshape = x.getOpArg("padding");
	PaddingMode pmode = (PaddingMode)(int)x.getOpArg("padding_mode");

	if (pmode != PaddingMode::zeros) {
		VP_THROW1(VERR_WILL_BE_IMPLEMENTED, "non-zeros padding mode");
	}

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (kshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (device() != k.device()) {
		printf("x.device() = %d, k.device() = %d\n", x.device(), k.device());
		VP_THROW(VERR_TENSOR_DEVICE);
	}

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 ychn = kshape[0];
	int64 kh = kshape[2];
	int64 kw = kshape[3];

	int64 yh = xh + pshape[0] + pshape[1] + 1 - kshape[2];
	int64 yw = xw + pshape[2] + pshape[3] + 1 - kshape[3];

	if (xchn % kshape[1] != 0) VP_THROW(VERR_SHAPE_CONV2D);

	int64 group = xchn / kshape[1];

	VShape yshape{ ndat, ychn, yh, yw };

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	float* pk = k.float_ptr();

	CALL_MATH(conv2d, device(), py, px, pk, ndat, ychn, yh, yw, xchn, xh, xw, kh, kw, group, pshape[0], pshape[2], (int)pmode);

	if (0) {
		static int nth = 0;
		VValue x_axes = x.getOpArg("axes", VList{});
		VValue y_axes = y.getOpArg("axes", VList{});
		x.setOpArg("axes", VList{ 0,2,3,1 });
		y.setOpArg("axes", VList{ 0,2,3,1 });
		VTensor xt = x.transpose(tracer);
		VTensor yt = y.transpose(tracer);
		float xf = xt.square(tracer).mean(tracer).getElement({ 0 }, tracer);
		float kf = k.square(tracer).mean(tracer).getElement({ 0 }, tracer);
		float yf = yt.square(tracer).mean(tracer).getElement({ 0 }, tracer);
		printf("[%d] x: %f%s, k = %f%s, y = %f%s\n", nth++, xf, xt.shape().desc().c_str(), kf, kshape.desc().c_str(), yf, yt.shape().desc().c_str());
		//k.dump("k");
		//x.dump("x");
		//xt.dump("xt");
		//y.dump("y");
		//yt.dump("yt");
		x.setOpArg("axes", x_axes);
		y.setOpArg("axes", y_axes);
	}

	return y;
}

void VTensor::conv2d_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	VTensor k = m_core->m_operands[1];

	VShape xshape = x.shape();
	VShape yshape = ygrad.shape();
	VShape kshape = k.shape();

	VShape pshape = getMyArg("padding");

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 yh = yshape[2];
	int64 yw = yshape[3];

	int64 ychn = kshape[0];
	int64 kh = kshape[2];
	int64 kw = kshape[3];

	int64 group = xchn / kshape[1];

	float* pgy = ygrad.float_ptr();

	if (x.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), xshape, type(), device());

		float* pgx = xgrad.float_ptr();
		float* pk = k.float_ptr();

		CALL_MATH(conv2d_backward_x, device(), pgx, pgy, pk, ndat, xchn, xh, xw, ychn, yh, yw, kh, kw, group, pshape[0], pshape[2]);

		pQueue->regist(x, xgrad);
	}
	else {
		pQueue->registNoGrad(x);
	}

	if (k.needGrad()) {
		VTensor kgrad = tracer.createTensor(session(), kshape, type(), device());

		float* pgk = kgrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(conv2d_backward_k, device(), pgk, pgy, px, ndat, xchn, xh, xw, ychn, yh, yw, kh, kw, group, pshape[0], pshape[2]);

		pQueue->regist(k, kgrad);
	}
	else {
		pQueue->registNoGrad(k);
	}
}

VTensor VTensor::conv2d_transposed(VTensor k, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape kshape = k.shape();

	VShape pshape = x.getOpArg("padding");
	PaddingMode pmode = (PaddingMode)(int)x.getOpArg("padding_mode");

	if (pmode != PaddingMode::zeros) {
		VP_THROW1(VERR_WILL_BE_IMPLEMENTED, "non-zeros padding mode");
	}

	VShape stride = x.getOpArg("stride");

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (kshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (device() != k.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 ychn = kshape[0];

	int64 kh = kshape[2];
	int64 kw = kshape[3];

	int64 sh = stride[0];
	int64 sw = stride[1];

	int64 yh = xh * stride[0];
	int64 yw = xw * stride[1];

	if (kshape[1] != xchn) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	VTensor y = tracer.createTensor(session(), VShape{ ndat, ychn, yh, yw }, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	float* pk = k.float_ptr();

	CALL_MATH(conv2d_transposed, device(), py, px, pk, ndat, xchn, xh, xw, ychn, kh, kw, sh, sw);

	return y;
}

void VTensor::conv2d_transposed_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	VTensor k = m_core->m_operands[1];

	VShape xshape = x.shape();
	VShape kshape = k.shape();

	VShape stride = getMyArg("stride");

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 ychn = kshape[0];

	int64 kh = kshape[2];
	int64 kw = kshape[3];

	int64 sh = stride[0];
	int64 sw = stride[1];

	int64 yh = xh * stride[0];
	int64 yw = xw * stride[1];

	float* pgy = ygrad.float_ptr();

	if (x.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), xshape, type(), device());

		float* pgx = xgrad.float_ptr();
		float* pk = k.float_ptr();

		CALL_MATH(conv2d_transposed_backward_x, device(), pgx, pgy, pk, ndat, xchn, xh, xw, ychn, kh, kw, sh, sw);

		pQueue->regist(x, xgrad);
	}
	else {
		pQueue->registNoGrad(x);
	}

	if (k.needGrad()) {
		VTensor kgrad = tracer.createTensor(session(), kshape, type(), device());

		float* pgk = kgrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(conv2d_transposed_backward_k, device(), pgk, pgy, px, ndat, xchn, xh, xw, ychn, kh, kw, sh, sw);

		pQueue->regist(k, kgrad);
	}
	else {
		pQueue->registNoGrad(k);
	}
}

VTensor VTensor::conv2d_dilated(VTensor k, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape kshape = k.shape();
	VShape gshape = x.getOpArg("gap");

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (kshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (device() != k.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 ychn = kshape[0];

	int64 kh = kshape[2];
	int64 kw = kshape[3];
	
	int64 gh = gshape[0];
	int64 gw = gshape[1];

	if (kshape[1] != xchn) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	VTensor y = tracer.createTensor(session(), VShape{ ndat, ychn, xh, xw }, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	float* pk = k.float_ptr();

	CALL_MATH(conv2d_dilated, device(), py, px, pk, ndat, xchn, xh, xw, ychn, kh, kw, gh, gw);

	return y;
}

void VTensor::conv2d_dilated_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	VTensor k = m_core->m_operands[1];

	VShape xshape = x.shape();
	VShape kshape = k.shape();
	VShape gshape = getMyArg("gap");

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 ychn = kshape[0];

	int64 kh = kshape[2];
	int64 kw = kshape[3];

	int64 gh = gshape[0];
	int64 gw = gshape[1];

	float* pgy = ygrad.float_ptr();

	if (x.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), xshape, type(), device());

		float* pgx = xgrad.float_ptr();
		float* pk = k.float_ptr();

		CALL_MATH(conv2d_dilated_backward_x, device(), pgx, pgy, pk, ndat, xchn, xh, xw, ychn, kh, kw, gh, gw);

		pQueue->regist(x, xgrad);
	}
	else {
		pQueue->registNoGrad(x);
	}

	if (k.needGrad()) {
		VTensor kgrad = tracer.createTensor(session(), kshape, type(), device());

		float* pgk = kgrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(conv2d_dilated_backward_k, device(), pgk, pgy, px, ndat, xchn, xh, xw, ychn, kh, kw, gh, gw);

		pQueue->regist(k, kgrad);
	}
	else {
		pQueue->registNoGrad(k);
	}
}

VTensor VTensor::maxpool(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape kernel = x.getOpArg("kernel");

	if (kernel.total_size() == 1) {
		VTensor y = tracer.createTensor(x, xshape, TensorCloneInit::share);
		return y;
	}

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (kernel.size() != 2) VP_THROW(VERR_SHAPE_KERNEL);

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 kh = kernel[0];
	int64 kw = kernel[1];

	if (kh <= 0 || kh > xh) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (kw <= 0 || kw > xw) VP_THROW(VERR_SHAPE_KERNEL);

	VTensor y = tracer.createTensor(session(), xshape, type(), device());
	VTensor map = tracer.createTensor(session(), xshape, VDataType::int32, device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	int* pm = map.int_ptr();

	CALL_MATH(maxpool, device(), py, px, pm, ndat, xchn, xh, xw, kh, kw);

	m_core->m_opArgs["map"] = map.cloneCore();

	return y;
}

void VTensor::maxpool_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();
		VShape kernel = getMyArg("kernel");

		VTensor xgrad;

		if (kernel.total_size() == 1) {
			xgrad = ygrad;
		}
		else {
			VTensor map = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("map"));

			int64 ndat = xshape[0];
			int64 xchn = xshape[1];
			int64 xh = xshape[2];
			int64 xw = xshape[3];

			int64 kh = kernel[0];
			int64 kw = kernel[1];

			xgrad = VTensor(session(), xshape, x.type(), x.device());

			float* pgx = xgrad.float_ptr();
			float* pgy = ygrad.float_ptr();
			int* pm = map.int_ptr();

			CALL_MATH(maxpool_backward_x, device(), pgx, pgy, pm, ndat, xchn, xh, xw, kh, kw);
		}

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::avgpool(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape kernel = x.getOpArg("kernel");

	if (kernel.total_size() == 1) {
		VTensor y = tracer.createTensor(x, xshape, TensorCloneInit::share);
		return y;
	}

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (kernel.size() != 2) VP_THROW(VERR_SHAPE_KERNEL);

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 kh = kernel[0];
	int64 kw = kernel[1];

	if (kh <= 0 || kh > xh) {
		VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}
	if (kw <= 0 || kw > xw) VP_THROW(VERR_SHAPE_KERNEL);

	VTensor y = tracer.createTensor(session(), xshape, type(), device());
	VTensor map = tracer.createTensor(session(), xshape, VDataType::int32, device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	int* pm = map.int_ptr();

	CALL_MATH(avgpool, device(), py, px, pm, ndat, xchn, xh, xw, kh, kw);

	m_core->m_opArgs["map"] = map.cloneCore();

	return y;
}

void VTensor::avgpool_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();
		VShape kernel = getMyArg("kernel");

		VTensor xgrad;

		if (kernel.total_size() == 1) {
			xgrad = ygrad;
		}
		else {
			VTensor map = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("map"));

			int64 ndat = xshape[0];
			int64 xchn = xshape[1];
			int64 xh = xshape[2];
			int64 xw = xshape[3];

			int64 kh = kernel[0];
			int64 kw = kernel[1];

			xgrad = VTensor(session(), xshape, x.type(), x.device());

			float* pgx = xgrad.float_ptr();
			float* pgy = ygrad.float_ptr();
			int* pm = map.int_ptr();

			CALL_MATH(avgpool_backward_x, device(), pgx, pgy, pm, ndat, xchn, xh, xw, kh, kw);
		}

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::globalavg(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	VShape yshape{ ndat, xchn };

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(globalavg, device(), py, px, ndat, xchn, xh, xw);

	return y;
}

void VTensor::globalavg_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();

		int64 ndat = xshape[0];
		int64 xchn = xshape[1];
		int64 xh = xshape[2];
		int64 xw = xshape[3];

		VTensor xgrad = tracer.createTensor(session(), xshape, x.type(), x.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();

		CALL_MATH(globalavg_backward_x, device(), pgx, pgy, ndat, xchn, xh, xw);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::adaptiveavg(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape size_info = x.getOpArg("size");

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 yh = size_info[0];
	int64 yw = size_info[1];

	int64 hratio = xh / yh;
	int64 wratio = xw / yw;

	if (xh % yh != 0) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (xw % yw != 0) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	VShape yshape{ ndat, xchn, yh, yw };

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(adaptiveavg, device(), py, px, ndat, xchn, yh, yw, hratio, wratio);

	return y;
}

void VTensor::adaptiveavg_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();

		int64 ndat = xshape[0];
		int64 xchn = xshape[1];
		int64 xh = xshape[2];
		int64 xw = xshape[3];

		int64 yh = ygrad.shape()[2];
		int64 yw = ygrad.shape()[3];

		int64 hratio = xh / yh;
		int64 wratio = xw / yw;

		VTensor xgrad = tracer.createTensor(session(), xshape, x.type(), x.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();

		CALL_MATH(adaptiveavg_backward_x, device(), pgx, pgy, ndat, xchn, xh, xw, hratio, wratio);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::stride(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape stride = x.getOpArg("stride");

	if (stride.total_size() == 1) {
		VTensor y = tracer.createTensor(x, xshape, TensorCloneInit::share);
		return y;
	}

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (stride.size() != 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	int64 ndat = xshape[0];
	int64 xchn = xshape[1];
	int64 xh = xshape[2];
	int64 xw = xshape[3];

	int64 sh = stride[0];
	int64 sw = stride[1];

	if (sh <= 0 || sh > xh) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (sw <= 0 || sw > xw) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	int64 yh = xh / sh;
	int64 yw = xw / sw;

	VShape yshape{ ndat, xchn, yh, yw };

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(stride, device(), py, px, ndat, xchn, xh, xw, yh, yw, sh, sw);

	return y;
}

void VTensor::stride_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();
		VShape stride = getMyArg("stride");

		VTensor xgrad;

		if (stride.total_size() == 1) {
			xgrad = ygrad;
		}
		else {
			int64 ndat = xshape[0];
			int64 xchn = xshape[1];
			int64 xh = xshape[2];
			int64 xw = xshape[3];

			int64 sh = stride[0];
			int64 sw = stride[1];

			int64 yh = xh / sh;
			int64 yw = xw / sw;

			xgrad = VTensor(session(), xshape, x.type(), x.device());

			float* pgx = xgrad.float_ptr();		// [nrow, nvec]
			float* pgy = ygrad.float_ptr();		// [nrow]

			CALL_MATH(stride_backward_x, device(), pgx, pgy, ndat, xchn, xh, xw, yh, yw, sh, sw);
		}

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::rnn(bool train, VTensor pmset, VExecTracer tracer) {
	return m_rnn_base(train, "rnn", pmset, tracer);
}

void VTensor::rnn_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	return m_rnn_backprop_base(ygrad, pQueue, tracer);
}

VTensor VTensor::lstm(bool train, VTensor pmset, VExecTracer tracer) {
	return m_rnn_base(train, "lstm", pmset, tracer);
}

void VTensor::lstm_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	return m_rnn_backprop_base(ygrad, pQueue, tracer);
}

VTensor VTensor::gru(bool train, VTensor pmset, VExecTracer tracer) {
	return m_rnn_base(train, "gru", pmset, tracer);
}

void VTensor::gru_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	return m_rnn_backprop_base(ygrad, pQueue, tracer);
}

VTensor VTensor::m_rnn_base(bool train, string cell_type, VTensor pmset, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	VDict params = getOpArg("rnn");
	
	m_core->m_opArgs["cell_type"] = cell_type;

	bool batch_first = getOpArg("batch_first");
	bool bInSeq = getOpArg("in_seq");
	bool bOutSeq = getOpArg("out_seq");

	HYPER_KEY  drop_ratio = x.getOpArg("drop_ratio");

	if (!bInSeq && !bOutSeq) {
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}

	if (batch_first && bInSeq) {
		VShape xshape = x.shape();
		if (xshape.size() != 3) VP_THROW(VERR_SHAPE_RNN);
		VTensor x_trans = tracer.createTensor(session(), { xshape[1], xshape[0], xshape[2] }, type(), device());
		x_trans.transpose_on(x, {1, 0, 2}, tracer);
		x = x_trans;
		m_core->m_opArgs["x_trans"] = x_trans.cloneCore();
	}

	xshape = x.shape();

	int64 nLayers = getOpArg("num_layers");;
	bool bidirectional = getOpArg("bidirectional");

	int64 ntimes = bInSeq ? xshape[0] : (int64)getOpArg("timesteps");
	int64 ndat = bInSeq ? xshape[1] : xshape[0];
	int64 nrec = getOpArg("rec_size");
	int64 nstack = bidirectional ? 2 * nLayers : nLayers;
	int64 nout = bidirectional ? 2 * nrec : nrec;

	VShape yshape = VShape{ nstack, ndat, nrec };

	VTensor output = tracer.createTensor(session(), yshape, type(), device());
	VTensor for_recurs, rev_recurs;
	VTensor for_y, rev_y;

	for (int64 n = 0, nth_stack = 0; n < nLayers; n++) {
		string prefix = "L" + std::to_string(n) + "F";

		VTensor wi = m_pickupParam(params, prefix + "_iw");
		VTensor wr = m_pickupParam(params, prefix + "_rw");
		VTensor bi = m_pickupParam(params, prefix + "_ib");
		VTensor br = m_pickupParam(params, prefix + "_rb");

		for_y = m_rnn(for_recurs, cell_type, true, x, wi, wr, bi, br, prefix, tracer);

		//for_y.dump("for_y");
		//for_recurs.dump("for_recurs");

		float* psrc = for_y.float_ptr();
		float* pout = output.float_ptr() + nth_stack++ * ndat * nrec;

		CALL_MATH(copy, device(), pout, psrc, ndat * nrec);

		if (bidirectional) {
			string prefix = "L" + std::to_string(n) + "R";

			wi = m_pickupParam(params, prefix + "_iw");
			wr = m_pickupParam(params, prefix + "_rw");
			bi = m_pickupParam(params, prefix + "_ib");
			br = m_pickupParam(params, prefix + "_rb");

			rev_y = m_rnn(rev_recurs, cell_type, false, x, wi, wr, bi, br, prefix, tracer);

			//rev_y.dump("rev_y");
			//rev_recurs.dump("rev_recurs");

			float* psrc = rev_y.float_ptr();
			float* pout = output.float_ptr() + nth_stack++ * ndat * nrec;

			CALL_MATH(copy, device(), pout, psrc, ndat * nrec);

			//rev_y = m_mergeRnnBidirectPairFotOutput(y, rev_y, tracer);

			x = m_mergeRnnBidirectPairForInput(for_recurs, rev_recurs, tracer);
		}
		else {
			if (n < nLayers - 1) x = for_recurs;
		}
	}

	VTensor y = output;

	if (bOutSeq) y = x;

	if (batch_first) {
		VShape yshape = y.shape();
		if (yshape.size() != 3) VP_THROW(VERR_SHAPE_RNN);
		VTensor y_trans = tracer.createTensor(session(), { yshape[1], yshape[0], yshape[2] }, type(), device());
		y_trans.transpose_on(y, { 1, 0, 2 }, tracer);
		y = y_trans;
	}

	if (train && drop_ratio >= 0) {
		VTensor mask;
		y = y.m_dropout(drop_ratio, mask, tracer);
		m_core->m_opArgs["mask"] = mask.cloneCore();
	}

	return y;
}

void VTensor::m_rnn_backprop_base(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	string cell_type = getMyArg("cell_type");

	bool batch_first = getMyArg("batch_first");

	bool bInSeq = getMyArg("in_seq");
	bool bOutSeq = getMyArg("out_seq");

	HYPER_KEY  drop_ratio = getMyArg("drop_ratio");

	if (drop_ratio >= 0) {
		VTensor mask = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("mask"));
		ygrad = m_dropout_backprop(ygrad, mask, drop_ratio, tracer);
	}

	if (bOutSeq && batch_first) {
		VShape yshape = ygrad.shape();
		if (yshape.size() != 3) VP_THROW(VERR_SHAPE_RNN);
		VTensor ygrad_trans = tracer.createTensor(session(), { yshape[1], yshape[0], yshape[2] }, type(), device());
		ygrad_trans.transpose_on(ygrad, { 1, 0, 2 }, tracer);
		ygrad = ygrad_trans;
	}

	//ygrad.dump("ygrad after transpose");

	int64 nLayers = getMyArg("num_layers");;
	bool bidirectional = getMyArg("bidirectional");

	//VShape out_shape = getMyArg("out_shape");

	int64 nstack = bidirectional ? 2 * nLayers : nLayers;

	VDict params = getMyArg("rnn");

	VTensor x = m_core->m_operands[0];
	VShape xshape = x.shape();
	if (batch_first) xshape = VShape { xshape[1], xshape[0], xshape[2] };

	int64 ntimes = bInSeq ? xshape[0] : (int64)getMyArg("timesteps");
	int64 ndat = bInSeq ? xshape[1] : xshape[0];
	//int64 ninp = xshape[-1];
	int64 nrec = getMyArg("rec_size");

	VShape rshape { ndat, nrec };
	VShape rtshape { ntimes, ndat, nrec };

	VTensor rgrad = tracer.createTensor(session(), rshape, type(), device());
	VTensor rtgrad_for = tracer.createTensor(session(), rtshape, type(), device());
	VTensor rtgrad_rev = bidirectional ? tracer.createTensor(session(), rtshape, type(), device()) : VTensor();

	if (bOutSeq) {
		if (bidirectional) {
			if (ygrad.shape()[-1] != 2 * nrec) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
			if (ygrad.shape().replace_end(nrec) != rtshape) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
			
			ygrad.m_undo_concat(rtgrad_for, rtgrad_rev, tracer);
		}
		else {
			if (ygrad.shape() != rtshape) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
			CALL_MATH(copy, device(), rtgrad_for.float_ptr(), ygrad.float_ptr(), ygrad.shape().total_size());
		}
	}
	else {
		rtgrad_for.setZero(tracer);
		if (bidirectional) {
			rtgrad_rev.setZero(tracer);
		}
	}

	VTensor xgrad, xgrev;

	for (int64 n = nLayers - 1, nth_stack = nstack; n >= 0; n--) {
		if (bidirectional) {
			string prefix = "L" + std::to_string(n) + "R";

			VTensor wi = m_pickupParam(params, prefix + "_iw");
			VTensor wr = m_pickupParam(params, prefix + "_rw");
			VTensor bi = m_pickupParam(params, prefix + "_ib");
			VTensor br = m_pickupParam(params, prefix + "_rb");

			if (!bOutSeq) {
				extract_on(rgrad, ygrad, 0, --nth_stack, 1, true, tracer);
			}
			else {
				rgrad.setZero(tracer);
			}

			xgrev = m_rnn_backprop(cell_type, false, rgrad, rtgrad_rev, wi, wr, bi, br, prefix, pQueue, tracer);
		}

		string prefix = "L" + std::to_string(n) + "F";

		VTensor wi = m_pickupParam(params, prefix + "_iw");
		VTensor wr = m_pickupParam(params, prefix + "_rw");
		VTensor bi = m_pickupParam(params, prefix + "_ib");
		VTensor br = m_pickupParam(params, prefix + "_rb");

		if (!bOutSeq) {
			extract_on(rgrad, ygrad, 0, --nth_stack, 1, true, tracer);
		}
		else {
			rgrad.setZero(tracer);
		}

		xgrad = m_rnn_backprop(cell_type, true, rgrad, rtgrad_for, wi, wr, bi, br, prefix, pQueue, tracer);

		if (bidirectional) {
			float* pf = xgrad.float_ptr();
			float* pr = xgrev.float_ptr();

			CALL_MATH(add, device(), pf, pf, pr, xgrad.shape().total_size());
		}

		if (bOutSeq) {
			if (bidirectional) {
				xgrad.m_undo_concat(rtgrad_for, rtgrad_rev, tracer);
			}
			else {
				if (ygrad.shape() != rtshape) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
				CALL_MATH(copy, device(), rtgrad_for.float_ptr(), xgrad.float_ptr(), xgrad.shape().total_size());
			}
		}
		else {
			rtgrad_for.setZero(tracer);

			if (bidirectional) rtgrad_rev.setZero(tracer);
		}
	}

	if (batch_first && bInSeq) {
		VShape xshape = xgrad.shape();
		VTensor xgrad_trans = tracer.createTensor(session(), { xshape[1], xshape[0], xshape[2] }, xgrad.type(), xgrad.device());
		xgrad_trans.transpose_on(xgrad, { 1, 0, 2 }, tracer);
		xgrad = xgrad_trans;
	}

	if (x.needGrad()) pQueue->regist(x, xgrad);
	else pQueue->registNoGrad(x);
}

void VTensor::m_undo_concat(VTensor dst1, VTensor dst2, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape shape1 = dst1.shape();
	VShape shape2 = dst2.shape();

	if (xshape.remove_end() != shape1.remove_end()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (xshape.remove_end() != shape2.remove_end()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (xshape[-1] != shape1[-1] + shape2[-2]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}
	if (dst1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(dst1.type()));
	}
	if (dst2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(dst2.type()));
	}

	if (x.device() != dst1.device()) VP_THROW(VERR_TENSOR_DEVICE);
	if (x.device() != dst2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 ncol1 = shape1[-1];
	int64 ncol2 = shape2[-1];
	int64 nrow = shape1.total_size() / ncol1;

	float* px = x.float_ptr();
	float* py1 = dst1.float_ptr();
	float* py2 = dst2.float_ptr();

	CALL_MATH(undo_concat, device(), py1, py2, px, nrow, ncol1, ncol2);
}

VTensor VTensor::m_rnn(VTensor& recurs_internal, string cell_type, bool forward, VTensor x, VTensor wi, VTensor wr, VTensor bi, VTensor br, string prefix, VExecTracer tracer) {
	VShape xshape = x.shape();

	VShape wishape = wi.shape();
	VShape wrshape = wr.shape();
	VShape bishape = bi.shape();
	VShape brshape = br.shape();

	bool bInSeq = xshape.size() == 3;

	if (wishape.size() != 2) VP_THROW(VERR_SHAPE_RNN);
	if (wrshape.size() != 2) VP_THROW(VERR_SHAPE_RNN);

	if (x.device() != wi.device()) VP_THROW(VERR_TENSOR_DEVICE);
	if (x.device() != wr.device()) VP_THROW(VERR_TENSOR_DEVICE);

	bool use_bias = getOpArg("use_bias");

	if (use_bias) {
		if (bishape.size() != 1) VP_THROW(VERR_SHAPE_RNN);
		if (brshape.size() != 1) VP_THROW(VERR_SHAPE_RNN);

		if (x.device() != bi.device()) VP_THROW(VERR_TENSOR_DEVICE);
		if (x.device() != br.device()) VP_THROW(VERR_TENSOR_DEVICE);
	}
	else {
		if (bishape.size() != 0) VP_THROW(VERR_SHAPE_RNN);
		if (brshape.size() != 0) VP_THROW(VERR_SHAPE_RNN);
	}

	int64 ntimes = bInSeq ? xshape[0] : (int64)getOpArg("timesteps");
	int64 ndat = bInSeq ? xshape[1] : xshape[0];
	int64 ninp = xshape[-1];
	int64 nrec = getOpArg("rec_size");

	int64 ngates = 1;
	bool bUseState = false;
	int actFunc = (int)ActFunc::tanh;
	HYPER_KEY leaky_alpha = 0;

	if (cell_type == "rnn") {
		actFunc = getOpArg("actfunc", "tanh");
		leaky_alpha = getOpArg("leaky_alpha", 0);
	}
	else if (cell_type == "lstm") {
		bUseState = getOpArg("use_state");
		ngates = 4;
	}
	else if (cell_type == "gru") {
		ngates = 3;
	}

	if (wishape[0] != ngates * nrec) VP_THROW(VERR_SHAPE_RNN);
	if (wishape[1] != ninp) VP_THROW(VERR_SHAPE_RNN);
	if (wrshape[0] != ngates * nrec) VP_THROW(VERR_SHAPE_RNN);
	if (wrshape[1] != nrec) VP_THROW(VERR_SHAPE_RNN);

	if (use_bias) {
		if (bishape[0] != ngates * nrec) VP_THROW(VERR_SHAPE_RNN);
		if (bishape[1] != ngates * nrec) VP_THROW(VERR_SHAPE_RNN);
	}
	else {
	}

	VShape rshape = VShape{ ndat, nrec };
	VShape ashape = VShape{ ndat, ngates * nrec };
	VShape rtshape = VShape{ ntimes, ndat, nrec };
	VShape atshape = VShape{ ntimes, ndat, ngates * nrec };

	VTensor recurs = tracer.createTensor(session(), rtshape, type(), device());
	VTensor affines = tracer.createTensor(session(), atshape, type(), device());

	VTensor iaffines = affines;
	VTensor raffines = affines;

	recurs.setZero(tracer);
	iaffines.setZero(tracer);

	if (cell_type == "gru") {
		raffines = tracer.createTensor(session(), atshape, type(), device());   // gru에 한해 입력과 리커전을 별도로 처리
		raffines.setZero(tracer);
	}

	VTensor states;

	if (cell_type == "lstm") {
		states = tracer.createTensor(session(), rtshape, type(), device());
		states.setZero(tracer);
	}

	float* px = x.float_ptr();

	float* pwi = wi.float_ptr();
	float* pwr = wr.float_ptr();
	float* pbi = bi.float_ptr();
	float* pbr = br.float_ptr();

	float* prs = recurs.float_ptr();
	float* pss = (states.isValid()) ? states.float_ptr() : NULL;
	float* pais = iaffines.float_ptr();
	float* pars = raffines.float_ptr();

	int64 xsize = ndat * ninp;
	int64 ysize = ndat * nrec;
	int64 asize = ngates * ysize;

	if (!forward && bInSeq) px += (ntimes - 1) * xsize;

	for (int64 n = 0; n < ntimes; n++) {
		int64 nt = forward ? n : ntimes - n - 1;

		float* pr = prs + nt * ysize;
		float* pai = pais + nt * asize;
		float* par = pars + nt * asize;

		float* pr_prev = pr + (forward ? -ysize : ysize);

		if (n > 0) CALL_MATH(matmul, device(), par, pwr, pr_prev, ngates * nrec, ndat, nrec, false);

		CALL_MATH(matmul, device(), pai, pwi, px, ngates * nrec, ndat, ninp, true);

		if (use_bias) {
			CALL_MATH(add_bias, device(), par, par, pbr, ndat, ngates * nrec);
			CALL_MATH(add_bias, device(), pai, pai, pbi, ndat, ngates * nrec);
		}

		if (cell_type == "rnn") {
			CALL_MATH(copy, device(), pr, pai, ysize);
			CALL_MATH(add, device(), pr, pr, pai, ysize);
			CALL_MATH(activate, device(), pr, pr, ndat, nrec, actFunc, HYPER_FETCH(leaky_alpha));
		}
		else if (cell_type == "lstm") {
			float* ps = pss + nt * ysize;
			float* ps_prev = (n > 0) ? (ps + (forward ? -ysize : ysize)) : NULL;

			CALL_MATH(lstm_process, device(), pr, ps, ps_prev, pai, ndat, nrec, ninp);
		}
		else if (cell_type == "gru") {
			CALL_MATH(gru_process, device(), pr, pai, par, nt, ntimes, ndat, nrec);
		}
		else {
			VP_THROW(VERR_CONDITIONAL_STATEMENT);
		}

		if (bInSeq) px += forward ? xsize : -xsize;
	}

	m_core->m_opArgs[prefix + "x"] = x.cloneCore();
	m_core->m_opArgs[prefix + "ngates"] = ngates;
	m_core->m_opArgs[prefix + "use_state"] = bUseState;

	//iaffines.dump("iaffines");

	m_core->m_opArgs[prefix + "recurs"] = recurs.cloneCore();
	m_core->m_opArgs[prefix + "states"] = states.cloneCore();
	m_core->m_opArgs[prefix + "iaffines"] = iaffines.cloneCore();

	if (cell_type == "gru") {
		m_core->m_opArgs[prefix + "raffines"] = raffines.cloneCore();
	}

	VTensor y = tracer.createTensor(session(), rshape, type(), device());

	float* py = y.float_ptr();
	float* pds = bUseState ? pss : prs;
	float* psrc = pds + (forward ? ((ntimes - 1) * ysize) : 0);

	CALL_MATH(copy, device(), py, psrc, ysize);

	recurs_internal = bUseState ? states : recurs;
	return y;
}

VTensor VTensor::m_rnn_backprop(string cell_type, bool forward, VTensor ygrad, VTensor ytgrad, VTensor wi, VTensor wr, VTensor bi, VTensor br, string prefix, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg(prefix + "x"));

	VShape xshape = x.shape();
	VShape wishape = wi.shape();
	VShape wrshape = wr.shape();
	VShape bishape = bi.shape();
	VShape brshape = br.shape();
	VShape rshape = ygrad.shape();
	VShape rtshape = ytgrad.shape();

	bool bInSeq = xshape.size() == 3;

	bool use_bias = getMyArg("use_bias");

	int64 ngates = 1;
	bool bUseState = false;
	int actFunc = (int)ActFunc::tanh;
	HYPER_KEY leaky_alpha = 0;

	if (cell_type == "rnn") {
		actFunc = getMyArg("actfunc");
		leaky_alpha = getMyArg("leaky_alpha");
	}
	else if (cell_type == "lstm") {
		bUseState = getMyArg("use_state");
		ngates = 4;
	}
	else if (cell_type == "gru") {
		ngates = 3;
	}

	int64 ntimes = rtshape[0];
	int64 ndat = rtshape[1];
	int64 nrec = rtshape[2];

	int64 ninp = xshape[-1];
	
	VShape ashape = VShape{ ndat, ngates * nrec };

	VTensor recurs = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg(prefix + "recurs"));
	VTensor iaffines = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg(prefix + "iaffines"));

	VTensor xgrad = tracer.createTensor(session(), xshape, type(), device());
	VTensor wigrad = tracer.createTensor(session(), wishape, type(), device());
	VTensor wrgrad = tracer.createTensor(session(), wrshape, type(), device());
	VTensor aigrad = tracer.createTensor(session(), iaffines.shape().remove_head(), type(), device());
	VTensor bigrad;
	VTensor brgrad;

	xgrad.setZero(tracer);
	wigrad.setZero(tracer);
	wrgrad.setZero(tracer);

	VTensor rgrad = ygrad;

	float* pgx = xgrad.float_ptr();
	float* pgwi = wigrad.float_ptr();
	float* pgwr = wrgrad.float_ptr();
	float* pgbi = NULL;
	float* pgbr = NULL;

	float* pgai = aigrad.float_ptr();

	float* px = x.float_ptr();

	float* pwi = wi.float_ptr();
	float* pwr = wr.float_ptr();
	float* prs = recurs.float_ptr();
	float* pais = iaffines.float_ptr();

	VTensor states, sgrad, raffines, argrad;

	float* pss = NULL;
	float* pgs = NULL;
	float* pars = pais;
	float* pgar = pgai;

	if (use_bias) {
		bigrad = tracer.createTensor(session(), bishape, type(), device());
		brgrad = tracer.createTensor(session(), brshape, type(), device());
		bigrad.setZero(tracer);
		brgrad.setZero(tracer);
		pgbi = bigrad.float_ptr();
		pgbr = brgrad.float_ptr();
	}

	if (cell_type == "lstm") {
		states = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg(prefix + "states"));

		if (bUseState) {
			sgrad = ygrad;
			
			rgrad = tracer.createTensor(session(), rshape, type(), device());
			rgrad.setZero(tracer);
		}
		else {
			sgrad = tracer.createTensor(session(), rshape, type(), device());
			sgrad.setZero(tracer);
		}

		pss = states.float_ptr();
		pgs = sgrad.float_ptr();
	}
	else if (cell_type == "gru") {
		raffines = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg(prefix + "raffines"));

		argrad = tracer.createTensor(session(), ashape, type(), device());
		argrad.setZero(tracer);

		pars = raffines.float_ptr();
		pgar = argrad.float_ptr();
	}

	float* pgr = rgrad.float_ptr();
	float* pgyt = ytgrad.float_ptr();

	int64 xsize = ndat * ninp;
	int64 ysize = ndat * nrec;
	int64 asize = ngates * ysize;

	if (bInSeq) {
		px += (forward ? ntimes - 1 : 0) * xsize;
		pgx += (forward ? ntimes - 1 : 0) * xsize;
	}

	/*
	if (bOutSeq) {
		pgy += ntimes * ysize;	// y grad has timestep axis only bOutSeq is true
	}
	*/

	bool is_gru = cell_type == "gru";

	for (int64 n = 0; n < ntimes; n++) {
		int64 nt = forward ? ntimes - n - 1 : n;

		float* pr = prs + nt * ysize;
		float* pai = pais + nt * asize;
		float* par = pars + nt * asize;

		CALL_MATH(add, device(), bUseState ? pgs : pgr, bUseState ? pgs : pgr, pgyt + nt * ysize, ysize);

		if (cell_type == "rnn") {
			CALL_MATH(activate_backward_with_y, device(), pgai, pgr, pr, ndat, nrec, actFunc, HYPER_FETCH(leaky_alpha));
		}
		else if (cell_type == "lstm") {
			float* ps_prev = (n < ntimes - 1) ? pss + (forward ? nt - 1 : nt + 1) * ysize : NULL;
			CALL_MATH(lstm_process_backward, device(), pgr, pgs, pgai, ps_prev, pai, ndat, nrec, ninp);
		}
		else if (cell_type == "gru") {
			CALL_MATH(gru_process_backward, device(), pgr, pgai, pgar, pr, pai, par, nt, ntimes, ndat, nrec);
		}
		else {
			VP_THROW(VERR_CONDITIONAL_STATEMENT);
		}

		if (use_bias) {
			CALL_MATH(add_bias_backward_b, device(), pgbi, pgai, ndat, ngates * nrec, true);
			CALL_MATH(add_bias_backward_b, device(), pgbr, pgar, ndat, ngates * nrec, true);
		}

		CALL_MATH(matmul_backward_w, device(), pgwi, pgai, px, ngates * nrec, ndat, ninp, true);

		//if (nt > 0) {
		if (n < ntimes -1) {
			float* pr_prev = pr + (forward ? -ysize : ysize);
			CALL_MATH(matmul_backward_w, device(), pgwr, pgar, pr_prev, ngates * nrec, ndat, nrec, true);
		}

		CALL_MATH(matmul_backward_x, device(), pgx, pgai, pwi, ngates * nrec, ndat, ninp, false);
		CALL_MATH(matmul_backward_x, device(), pgr, pgar, pwr, ngates * nrec, ndat, nrec, is_gru);

		if (bInSeq) {
			px += forward ? -xsize : xsize;
			pgx += forward ? -xsize : xsize;
		}
	}

	if (wi.needGrad()) pQueue->regist(wi, wigrad);
	else pQueue->registNoGrad(wi);

	if (wr.needGrad()) pQueue->regist(wr, wrgrad);
	else pQueue->registNoGrad(wr);

	if (use_bias && bi.needGrad()) pQueue->regist(bi, bigrad);
	else pQueue->registNoGrad(bi);

	if (use_bias && br.needGrad()) pQueue->regist(br, brgrad);
	else pQueue->registNoGrad(br);

	return xgrad;
}

VTensor VTensor::m_pickupParam(VDict params, string name) {
	return VTensor(session(), (VHTensor)params[name]);
}

VTensor VTensor::m_mergeRnnBidirectPairForInput(VTensor for_y, VTensor rev_y, VExecTracer tracer) {
	if (for_y.shape() != rev_y.shape()) VP_THROW(VERR_SHAPE_RNN);

	VTensor concated = for_y.concat(rev_y, tracer);

	return concated;
}

VTensor VTensor::embed(VTensor w, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape wshape = w.shape();

	bool position = getOpArg("position");
	int ndim = getOpArg("ndim");

	if (!position && x.type() != VDataType::int32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(x.type()));
	}
	if (x.device() != w.device()) VP_THROW(VERR_TENSOR_DEVICE);

	if (wshape.size() != 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (w.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(w.type()));
	}

	if (position && ndim >= 0) {
		xshape = xshape.cut_tail(ndim);
	}

	int64 ndat = xshape.total_size();
	int64 nword = wshape[0];
	int64 nvec = wshape[1];

	VShape yshape = xshape.append(nvec);

	VTensor y = tracer.createTensor(session(), yshape, w.type(), x.device());

	float* py = y.float_ptr();
	int* px = position ? NULL : x.int_ptr();
	float* pw = w.float_ptr();

	CALL_MATH(embed, device(), py, px, pw, ndat, nword, nvec, position);

	return y;
}

void VTensor::embed_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	VTensor w = m_core->m_operands[1];

	pQueue->registNoGrad(x);

	if (w.needGrad()) {
		VShape xshape = x.shape();
		VShape wshape = w.shape();

		bool position = getMyArg("position");
		int ndim = getMyArg("ndim");

		if (position && ndim >= 0) {
			xshape = xshape.cut_tail(ndim);
		}

		int64 ndat = xshape.total_size();
		int64 nword = wshape[0];
		int64 nvec = wshape[1];

		VTensor wgrad = tracer.createTensor(session(), wshape, VDataType::float32, x.device());
		wgrad.setZero(tracer);

		float* pgw = wgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		int* px = position ? NULL : x.int_ptr();

		CALL_MATH(embed_backward_w, device(), pgw, pgy, px, ndat, nword, nvec, position);

		pQueue->regist(w, wgrad);
	}
	else pQueue->registNoGrad(w);
}

VTensor VTensor::batchnorm(VTensor mavg, VTensor mvar, VTensor rescale, VTensor shift, bool train, VExecTracer tracer) {
	VTensor x = *this;
	VShape xshape = x.shape();

	if (mavg.hasNoData()) {
		VTensor y = tracer.createTensor(x, xshape, TensorCloneInit::share);
		return y;
	}

	HYPER_KEY momentum = x.getOpArg("momentum", 0);
	HYPER_KEY epsilon = x.getOpArg("epsilon", 0);

	int64 axis = (xshape.size() == 4) ? 1 : xshape.size() - 1;
	int64 ndat = xshape.total_size();
	int64 ncol = xshape[axis];
	int64 nrest= xshape.tail(axis+1).total_size();

	if (mavg.shape().total_size() != ncol) VP_THROW(VERR_SIZE_TENSOR);
	if (mvar.shape().total_size() != ncol) VP_THROW(VERR_SIZE_TENSOR);
	if (rescale.hasData() && rescale.shape().total_size() != ncol) VP_THROW(VERR_SIZE_TENSOR);
	if (shift.hasData() && shift.shape().total_size() != ncol) VP_THROW(VERR_SIZE_TENSOR);

	VTensor y = tracer.createTensor(session(), xshape, type(), device());
	VTensor norm = tracer.createTensor(session(), xshape, type(), device());
	VTensor bmavg = tracer.createTensor(session(), mavg.shape(), type(), device());
	VTensor bmvar = tracer.createTensor(session(), mvar.shape(), type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	float* pn = norm.float_ptr();
	float* pma = mavg.float_ptr();
	float* pmv = mvar.float_ptr();
	float* pba = bmavg.float_ptr();
	float* pbv = bmvar.float_ptr();
	float* pscale = rescale.float_ptr();
	float* pshift = shift.float_ptr();

	CALL_MATH(batchnorm_norm, device(), pn, px, pma, pmv, pba, pbv, ndat, ncol, nrest, HYPER_FETCH(momentum), HYPER_FETCH(epsilon), train);
	CALL_MATH(batchnorm_scale, device(), py, pn, pscale, pshift, ndat, ncol, nrest);

	m_core->m_opArgs["norm"] = norm.cloneCore();
	m_core->m_opArgs["bmvar"] = bmvar.cloneCore();

	if (train) {
		VHTensor hAvg = (VHTensor)(int64)mavg.getOpArg("moving_stat_src");
		VHTensor hVar = (VHTensor)(int64)mvar.getOpArg("moving_stat_src");
		VTensor avgSrc(session(), hAvg);
		VTensor varSrc(session(), hVar);
		CALL_MATH(copy_data, avgSrc.device(), device(), avgSrc.float_ptr(), pma, avgSrc.byte_size());
		CALL_MATH(copy_data, varSrc.device(), device(), varSrc.float_ptr(), pmv, varSrc.byte_size());
	}

	return y;
}

void VTensor::batchnorm_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor mavg = m_core->m_operands[1];
	VTensor mvar = m_core->m_operands[2];

	if (mavg.hasNoData()) {
		VTensor x = m_core->m_operands[0];
		if (x.needGrad()) pQueue->regist(x, ygrad);
		else pQueue->registNoGrad(x);
		return;
	}

	VTensor x = m_core->m_operands[0];
	VTensor rescale = m_core->m_operands[3];
	VTensor shift = m_core->m_operands[4];

	VShape xshape = x.shape();
	VShape rshape = rescale.shape();
	VShape sshape = shift.shape();

	HYPER_KEY epsilon = getMyArg("epsilon");

	VTensor norm = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("norm"));
	VTensor bmvar = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("bmvar"));

	int64 axis = (xshape.size() == 4) ? 1 : xshape.size() - 1;
	int64 ndat = xshape.total_size();
	int64 ncol = xshape[axis];
	int64 nrest = xshape.tail(axis + 1).total_size();

	VTensor xgrad = tracer.createTensor(session(), xshape, x.type(), x.device());
	VTensor rgrad = tracer.createTensor(session(), rshape, x.type(), x.device());
	VTensor sgrad = tracer.createTensor(session(), sshape, x.type(), x.device());

	float* pgx = xgrad.float_ptr();
	float* pgy = ygrad.float_ptr();
	float* pgr = rgrad.float_ptr();
	float* pgs = sgrad.float_ptr();

	float* px = x.float_ptr();
	float* pn = norm.float_ptr();
	float* pbv = bmvar.float_ptr();
	float* pscale = rescale.float_ptr();
	float* pshift = shift.float_ptr();

	CALL_MATH(batchnorm_backward_x, device(), pgx, pgy, pscale, ndat, ncol, nrest);
	CALL_MATH(batchnorm_backward_scale, device(), pgr, pgy, px, ndat, ncol, nrest);
	CALL_MATH(batchnorm_backward_shift, device(), pgs, pgy, ndat, ncol, nrest);
	CALL_MATH(batchnorm_backward_norm, device(), pgx, pbv, ndat, ncol, nrest, HYPER_FETCH(epsilon));

	if (x.needGrad()) pQueue->regist(x, xgrad);
	else pQueue->registNoGrad(x);

	if (rescale.needGrad()) pQueue->regist(rescale, rgrad);
	else pQueue->registNoGrad(rescale);

	if (shift.needGrad()) pQueue->regist(shift, sgrad);
	else pQueue->registNoGrad(shift);
}

VTensor VTensor::layernorm(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 axis = x.getOpArg("axis");
	HYPER_KEY scale = x.getOpArg("scale", 0);

	if (axis != 0) VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 nrow = xshape[0];
	int64 ncol = xshape.total_size() / nrow;

	VTensor y = tracer.createTensor(session(), xshape, type(), device());
	VTensor stat = tracer.createTensor(session(), VShape{ nrow, 2 }, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	float* ps = stat.float_ptr();

	CALL_MATH(layernorm, device(), py, px, ps, nrow, ncol, HYPER_FETCH(scale));

	m_core->m_opArgs["stat"] = stat.cloneCore();

	return y;
}

void VTensor::layernorm_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();

		int64 axis = getMyArg("axis");
		HYPER_KEY scale = getMyArg("scale");

		if (axis != 0) VP_THROW(VERR_NOT_IMPLEMENTED_YET);

		VTensor stat = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("stat"));

		int64 nrow = xshape[0];
		int64 ncol = xshape.total_size() / nrow;

		VTensor xgrad = tracer.createTensor(session(), xshape, x.type(), x.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* ps = stat.float_ptr();

		CALL_MATH(layernorm_backward, device(), pgx, pgy, ps, nrow, ncol, HYPER_FETCH(scale));

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::mh_attention(VTensor key, VTensor query, VTensor value, VTensor Kw, VTensor Kb, VTensor Qw, VTensor Qb, VTensor Vw, VTensor Vb, VTensor Ow, VTensor Ob, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 nhead = x.getOpArg("head_cnt");
	bool mask = x.getOpArg("mask", false);

	HYPER_KEY coef = x.getOpArg("coef", 0);

	if (xshape.size() != 3 || xshape[2] % nhead != 0) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (xshape != key.shape() || xshape != query.shape() || xshape != value.shape()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	int64 nbat = xshape[0];
	int64 ntimes = xshape[1];
	int64 nvec = xshape[2];
	int64 npiece = nvec / nhead;

	if (Kw.shape()[0] != nvec || Kw.shape()[1] != nvec) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (Qw.shape()[0] != nvec || Qw.shape()[1] != nvec) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (Vw.shape()[0] != nvec || Vw.shape()[1] != nvec) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	VTensor y = tracer.createTensor(session(), xshape, type(), device());

	VTensor K = tracer.createTensor(session(), VShape{ nbat, ntimes, nvec }, VDataType::float32, device());
	VTensor Q = tracer.createTensor(session(), VShape{ nbat, ntimes, nvec }, VDataType::float32, device());
	VTensor V = tracer.createTensor(session(), VShape{ nbat, ntimes, nvec }, VDataType::float32, device());
	VTensor probs = tracer.createTensor(session(), VShape{ nbat, ntimes, ntimes, nhead }, VDataType::float32, device());
	VTensor mixed = tracer.createTensor(session(), xshape, type(), device());

	float* pk = key.float_ptr();
	float* pq = query.float_ptr();
	float* pv = value.float_ptr();
	float* py = y.float_ptr();
	float* pKw = Kw.float_ptr();
	float* pKb = Kb.float_ptr();
	float* pQw = Qw.float_ptr();
	float* pQb = Qb.float_ptr();
	float* pVw = Vw.float_ptr();
	float* pVb = Vb.float_ptr();
	float* pOw = Ow.float_ptr();
	float* pOb = Ob.float_ptr();
	float* pK = K.float_ptr();
	float* pQ = Q.float_ptr();
	float* pV = V.float_ptr();
	float* pp = probs.float_ptr();
	float* pm = mixed.float_ptr();

	CALL_MATH(matmul, device(), pK, pKw, pk, nvec, nbat * ntimes, nvec, false);
	CALL_MATH(matmul, device(), pQ, pQw, pq, nvec, nbat * ntimes, nvec, false);
	CALL_MATH(matmul, device(), pV, pVw, pv, nvec, nbat * ntimes, nvec, false);

	CALL_MATH(add_bias, device(), pK, pK, pKb, nbat * ntimes, nvec);
	CALL_MATH(add_bias, device(), pQ, pQ, pQb, nbat * ntimes, nvec);
	CALL_MATH(add_bias, device(), pV, pV, pVb, nbat * ntimes, nvec);

	CALL_MATH(mult_on_heads, device(), pp, pK, pQ, nbat, ntimes, nvec, nhead, npiece, HYPER_FETCH(coef));

	if (mask) {
		CALL_MATH(set_mh_attention_mask, device(), pp, nbat, ntimes, nhead, true);
	}

	CALL_MATH(softmax_direct_on_axis, device(), pp, nbat * ntimes, ntimes, nhead);
	CALL_MATH(mix_values, device(), pm, pp, pV, nbat, ntimes, nvec, nhead, npiece);
	CALL_MATH(matmul, device(), py, pOw, pm, nvec, nbat * ntimes, nvec, false);
	CALL_MATH(add_bias, device(), py, py, pOb, nbat * ntimes, nvec);

	m_core->m_opArgs["K"] = K.cloneCore();
	m_core->m_opArgs["Q"] = Q.cloneCore();
	m_core->m_opArgs["V"] = V.cloneCore();
	m_core->m_opArgs["probs"] = probs.cloneCore();
	m_core->m_opArgs["mixed"] = mixed.cloneCore();

	return y;
}

void VTensor::multihead_attention_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	VTensor key = m_core->m_operands[1];
	VTensor query = m_core->m_operands[2];
	VTensor value = m_core->m_operands[3];

	VShape xshape = x.shape();

	int64 nhead = getMyArg("head_cnt");
	bool mask = getMyArg("mask");

	HYPER_KEY coef = getMyArg("coef");

	int64 xsize = xshape.total_size();
	int64 nbat = xshape[0];
	int64 ntimes = xshape[1];
	int64 nvec = xshape[2];
	int64 npiece = nvec / nhead;

	VTensor Kw = m_core->m_operands[4];
	VTensor Kb = m_core->m_operands[5];
	VTensor Qw = m_core->m_operands[6];
	VTensor Qb = m_core->m_operands[7];
	VTensor Vw = m_core->m_operands[8];
	VTensor Vb = m_core->m_operands[9];
	VTensor Ow = m_core->m_operands[10];
	VTensor Ob = m_core->m_operands[11];

	VTensor xgrad = tracer.createTensor(session(), xshape, x.type(), x.device());
	VTensor kgrad = tracer.createTensor(session(), xshape, x.type(), x.device());
	VTensor qgrad = tracer.createTensor(session(), xshape, x.type(), x.device());
	VTensor vgrad = tracer.createTensor(session(), xshape, x.type(), x.device());

	VTensor Kwgrad = tracer.createTensor(session(), Kw.shape(), type(), device());
	VTensor Kbgrad = tracer.createTensor(session(), Kb.shape(), type(), device());
	VTensor Qwgrad = tracer.createTensor(session(), Qw.shape(), type(), device());
	VTensor Qbgrad = tracer.createTensor(session(), Qb.shape(), type(), device());
	VTensor Vwgrad = tracer.createTensor(session(), Vw.shape(), type(), device());
	VTensor Vbgrad = tracer.createTensor(session(), Vb.shape(), type(), device());
	VTensor Owgrad = tracer.createTensor(session(), Ow.shape(), type(), device());
	VTensor Obgrad = tracer.createTensor(session(), Ob.shape(), type(), device());

	VTensor K = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("K"));
	VTensor Q = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("Q"));
	VTensor V = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("V"));
	VTensor probs = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("probs"));
	VTensor mixed = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("mixed"));

	VTensor mixedgrad = tracer.createTensor(session(), xshape, type(), device()); // 다른 텐서와 겸용 가능성 검토
	VTensor Kgrad = tracer.createTensor(session(), VShape{ nbat, ntimes, nvec }, VDataType::float32, device());
	VTensor Qgrad = tracer.createTensor(session(), VShape{ nbat, ntimes, nvec }, VDataType::float32, device());
	VTensor Vgrad = tracer.createTensor(session(), VShape{ nbat, ntimes, nvec }, VDataType::float32, device());
	VTensor probsgrad = tracer.createTensor(session(), VShape{ nbat, ntimes, ntimes, nhead }, VDataType::float32, device());

	float* pk = key.float_ptr();
	float* pq = query.float_ptr();
	float* pv = value.float_ptr();
	float* pm = mixed.float_ptr();
	float* pKw = Kw.float_ptr();
	float* pQw = Qw.float_ptr();
	float* pVw = Vw.float_ptr();
	float* pOw = Ow.float_ptr();
	float* pK = K.float_ptr();
	float* pQ = Q.float_ptr();
	float* pV = V.float_ptr();
	float* pp = probs.float_ptr();

	float* pgx = xgrad.float_ptr();
	float* pgk = kgrad.float_ptr();
	float* pgq = qgrad.float_ptr();
	float* pgv = vgrad.float_ptr();
	float* pgy = ygrad.float_ptr();

	float* pgK = Kgrad.float_ptr();
	float* pgQ = Qgrad.float_ptr();
	float* pgV = Vgrad.float_ptr();
	float* pgp = probsgrad.float_ptr();
	float* pgm = mixedgrad.float_ptr();
	float* pgKw = Kwgrad.float_ptr();
	float* pgKb = Kbgrad.float_ptr();
	float* pgQw = Qwgrad.float_ptr();
	float* pgQb = Qbgrad.float_ptr();
	float* pgVw = Vwgrad.float_ptr();
	float* pgVb = Vbgrad.float_ptr();
	float* pgOw = Owgrad.float_ptr();
	float* pgOb = Obgrad.float_ptr();

	CALL_MATH(add_bias_backward_b, device(), pgOb, pgy, nbat * ntimes, nvec, false);
	CALL_MATH(matmul_backward_w, device(), pgOw, pgy, pm, nvec, nbat * ntimes, nvec, false);
	CALL_MATH(matmul_backward_x, device(), pgm, pgy, pOw, nvec, nbat * ntimes, nvec, false);
	CALL_MATH(mix_values_backward_prop, device(), pgp, pgm, pV, nbat, ntimes, nvec, nhead, npiece);
	CALL_MATH(mix_values_backward_value, device(), pgV, pgm, pp, nbat, ntimes, nvec, nhead, npiece);
	CALL_MATH(softmax_direct_on_axis_backward, device(), pgp, pp, nbat, ntimes, nhead, HYPER_FETCH(coef));

	if (mask) {
		CALL_MATH(set_mh_attention_mask, device(), pgp, nbat, ntimes, nhead, false);
	}

	CALL_MATH(mult_on_heads_backward, device(), pgK, pgp, pQ, nbat, ntimes, nvec, nhead, npiece);
	CALL_MATH(mult_on_heads_backward, device(), pgQ, pgp, pK, nbat, ntimes, nvec, nhead, npiece);

	CALL_MATH(add_bias_backward_b, device(), pgKb, pgK, nbat * ntimes, nvec, false);
	CALL_MATH(add_bias_backward_b, device(), pgQb, pgQ, nbat * ntimes, nvec, false);
	CALL_MATH(add_bias_backward_b, device(), pgVb, pgV, nbat * ntimes, nvec, false);

	CALL_MATH(matmul_backward_w, device(), pgKw, pgK, pk, nvec, nbat * ntimes, nvec, false);
	CALL_MATH(matmul_backward_w, device(), pgQw, pgQ, pq, nvec, nbat * ntimes, nvec, false);
	CALL_MATH(matmul_backward_w, device(), pgVw, pgV, pv, nvec, nbat* ntimes, nvec, false);
	
	CALL_MATH(matmul_backward_x, device(), pgk, pgK, pKw, nvec, nbat* ntimes, nvec, false);
	CALL_MATH(matmul_backward_x, device(), pgq, pgQ, pQw, nvec, nbat* ntimes, nvec, false);
	CALL_MATH(matmul_backward_x, device(), pgv, pgV, pVw, nvec, nbat* ntimes, nvec, false);

	CALL_MATH(set_zero, device(), pgx, xsize);

	//setMyArgs();	// QKV ... mixedgrad 등의 공간이 x, ..., QKVb 역전파 처리 때 남아있지 않게 하기 위한 청소

	if (key.needGrad()) {
		if (key.getNth() != x.getNth()) pQueue->regist(key, kgrad);
		else {
			CALL_MATH(add, device(), pgx, pgx, pgk, xsize);
			x.decOperandRefCount();
		}
	}
	else pQueue->registNoGrad(key);

	if (query.needGrad()) {
		if (query.getNth() != x.getNth()) pQueue->regist(query, qgrad);
		else {
			CALL_MATH(add, device(), pgx, pgx, pgq, xsize);
			x.decOperandRefCount();
		}
	}
	else pQueue->registNoGrad(query);

	if (value.needGrad()) {
		if (value.getNth() != x.getNth()) pQueue->regist(value, vgrad);
		else {
			CALL_MATH(add, device(), pgx, pgx, pgv, xsize);
			x.decOperandRefCount();
		}
	}
	else pQueue->registNoGrad(value);

	if (x.needGrad()) {
		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);

	if (Ow.needGrad()) pQueue->regist(Ow, Owgrad);
	else pQueue->registNoGrad(Ow);

	if (Ob.needGrad()) pQueue->regist(Ob, Obgrad);
	else pQueue->registNoGrad(Ob);

	if (Kw.needGrad()) pQueue->regist(Kw, Kwgrad);
	else pQueue->registNoGrad(Kw);

	if (Kb.needGrad()) pQueue->regist(Kb, Kbgrad);
	else pQueue->registNoGrad(Kb);

	if (Qw.needGrad()) pQueue->regist(Qw, Qwgrad);
	else pQueue->registNoGrad(Qw);

	if (Qb.needGrad()) pQueue->regist(Qb, Qbgrad);
	else pQueue->registNoGrad(Qb);

	if (Vw.needGrad()) pQueue->regist(Vw, Vwgrad);
	else pQueue->registNoGrad(Vw);

	if (Vb.needGrad()) pQueue->regist(Vb, Vbgrad);
	else pQueue->registNoGrad(Vb);
}

VTensor VTensor::dropout(bool train, VExecTracer tracer) {
	VTensor x = *this;

	if (!train) {
		VTensor y = x;
		return y;
	}

	VShape xshape = x.shape();

	HYPER_KEY drop_ratio = x.getOpArg("drop_ratio");

	VTensor mask;
	VTensor y = m_dropout(drop_ratio, mask, tracer);

	m_core->m_opArgs["mask"] = mask.cloneCore();

	return y;
}

VTensor VTensor::m_dropout(HYPER_KEY drop_ratio, VTensor& mask, VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ndat = xshape.total_size();

	VTensor y = tracer.createTensor(session(), xshape, type(), device());
	
	mask = tracer.createTensor(session(), xshape, VDataType::uint8, device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	unsigned char* pm = mask.uchar_ptr();

	CALL_MATH(dropout, device(), py, px, pm, ndat, HYPER_FETCH(drop_ratio));

	return y;
}

void VTensor::dropout_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VTensor mask = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("mask"));

		HYPER_KEY drop_ratio = getMyArg("drop_ratio");

		VTensor xgrad = m_dropout_backprop(ygrad, mask, drop_ratio, tracer);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::m_dropout_backprop(VTensor ygrad, VTensor mask, HYPER_KEY drop_ratio, VExecTracer tracer) {
	VShape xshape = ygrad.shape();

	VTensor xgrad = tracer.createTensor(session(), xshape, type(), device());

	int64 ndat = xshape.total_size();

	float* pgx = xgrad.float_ptr();
	float* pgy = ygrad.float_ptr();
	unsigned char* pm = mask.uchar_ptr();

	CALL_MATH(dropout_backward, device(), pgx, pgy, pm, ndat, HYPER_FETCH(drop_ratio));

	return xgrad;
}

VTensor VTensor::parallel_concat(VTensor x2, VExecTracer tracer) {
	VTensor x1 = *this;

	VShape xshape1 = x1.shape();
	VShape xshape2 = x2.shape();

	int64 ncol1 = xshape1[-1];
	int64 ncol2 = xshape2[-1];
	int64 nrow = xshape1.total_size() / ncol1;
	int64 nrest = 1;

	VShape yshape = xshape1.replace_end(ncol1 + ncol2);

	if (xshape1.size() == 4) {
		if (xshape2.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (xshape1.remove_nth(1) != xshape2.remove_nth(1)) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

		ncol1 = xshape1[1];
		ncol2 = xshape2[1];
		nrow = xshape1[0];
		nrest = xshape1[2] * xshape1[3];

		yshape = xshape1.replace_nth(1, ncol1 + ncol2);
	}
	else {
		if (xshape2.total_size() / ncol2 != nrow) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}

	if (x1.device() != x2.device()) VP_THROW(VERR_TENSOR_DEVICE);
	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(parallel_concat, device(), py, px1, px2, nrow, ncol1, ncol2, nrest);

	return y;
}

void VTensor::parallel_concat_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	VShape xshape1 = x1.shape();
	VShape xshape2 = x2.shape();

	int64 ncol1 = xshape1[-1];
	int64 ncol2 = xshape2[-1];
	int64 nrow = xshape1.total_size() / ncol1;
	int64 nrest = 1;

	VShape yshape = xshape1.replace_end(ncol1 + ncol2);

	if (xshape1.size() == 4) {
		ncol1 = xshape1[1];
		ncol2 = xshape2[1];
		nrow = xshape1[0];
		nrest = xshape1[2] * xshape1[3];

		yshape = xshape1.replace_nth(1, ncol1 + ncol2);
	}

	float* pgy = ygrad.float_ptr();

	if (x1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), xshape1, x1.type(), x1.device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(parallel_concat_backward_x1, device(), pgx, pgy, nrow, ncol1, ncol2, nrest);

		pQueue->regist(x1, xgrad);
	}
	else pQueue->registNoGrad(x1);

	if (x2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), xshape2, x2.type(), x2.device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(parallel_concat_backward_x2, device(), pgx, pgy, nrow, ncol1, ncol2, nrest);

		pQueue->regist(x2, xgrad);
	}
	else pQueue->registNoGrad(x2);
}

VTensor VTensor::add(VTensor second, VExecTracer tracer) {
	VTensor a = *this;
	VTensor b = second;

	if (b.hasNoData()) {
		//VTensor y = a;
		VTensor y = tracer.createTensor(a, a.shape(), TensorCloneInit::share);
		return y;
	}

	VShape ashape = a.shape();
	VShape bshape = b.shape();

	if (ashape != bshape) {
		if (ashape.total_size() % bshape.total_size() != 0) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}
	if (device() != second.device()) {
		printf("ashape: %s, a.device=%d\n", ashape.desc().c_str(), a.device());
		printf("bshape: %s, b.device=%d\n", bshape.desc().c_str(), b.device());

		VP_THROW(VERR_TENSOR_DEVICE);
	}

	int64 ndat = ashape.total_size();
	int64 ncol = bshape.total_size();
	int64 nrow = ndat / ncol;

	VTensor y = tracer.createTensor(session(), ashape, type(), device());

	float* py = y.float_ptr();
	float* pa = a.float_ptr();
	float* pb = b.float_ptr();

	if (ashape == bshape) {
		CALL_MATH(add, device(), py, pa, pb, ndat);
	}
	else {
		CALL_MATH(add_bias, device(), py, pa, pb, nrow, ncol);
	}

	return y;
}

void VTensor::add_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor a = m_core->m_operands[0];
	VTensor b = m_core->m_operands[1];

	// in the case of summation of loss terms, all added loss should start with empty loss which means 1
	if (!ygrad.isValid()) {
		if (a.needGrad()) pQueue->regist(a, ygrad);
		else pQueue->registNoGrad(a);

		if (b.needGrad()) pQueue->regist(b, ygrad);
		else pQueue->registNoGrad(b);

		return;
	}

	if (a.needGrad()) {
		VShape ashape = a.shape();
		VTensor agrad = tracer.createTensor(ygrad, ashape, TensorCloneInit::share);

		pQueue->regist(a, agrad);
	}
	else pQueue->registNoGrad(a);

	if (b.needGrad()) {
		if (a.shape() == b.shape()) {
			//VTensor bgrad = ygrad;
			VShape bshape = b.shape();
			VTensor bgrad = tracer.createTensor(ygrad, bshape, TensorCloneInit::share);

			pQueue->regist(b, bgrad);
		}
		else {
			int64 ndat = a.shape().total_size();
			int64 ncol = b.shape()[-1];
			int64 nrow = ndat / ncol;

			VTensor bgrad = tracer.createTensor(session(), b.shape(), type(), device());

			float* pgb = bgrad.float_ptr();
			float* pgy = ygrad.float_ptr();

			CALL_MATH(add_bias_backward_b, device(), pgb, pgy, nrow, ncol, false);

			pQueue->regist(b, bgrad);
		}
	}
	else pQueue->registNoGrad(b);
}

VTensor VTensor::add_2d_bias(VTensor second, VExecTracer tracer) {
	VTensor a = *this;
	VTensor b = second;

	if (b.hasNoData()) {
		VTensor y = tracer.createTensor(a, a.shape(), TensorCloneInit::share);
		return y;
	}

	VShape ashape = a.shape();
	VShape bshape = b.shape();

	int64 ndat = ashape[0];
	int64 xchn = ashape[1];
	int64 xh = ashape[2];
	int64 xw = ashape[3];

	if (xchn % bshape.total_size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (device() != second.device()) {
		VP_THROW(VERR_TENSOR_DEVICE);
	}

	VTensor y = tracer.createTensor(session(), ashape, type(), device());

	float* py = y.float_ptr();
	float* pa = a.float_ptr();
	float* pb = b.float_ptr();

	CALL_MATH(add_2d_bias, device(), py, pa, pb, ndat, xchn, xh, xw);

	return y;
}

void VTensor::add_2d_bias_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor a = m_core->m_operands[0];
	VTensor b = m_core->m_operands[1];

	VShape ashape = a.shape();

	if (a.needGrad()) {
		VTensor agrad = tracer.createTensor(ygrad, ashape, TensorCloneInit::share);

		pQueue->regist(a, agrad);
	}
	else pQueue->registNoGrad(a);

	if (b.needGrad()) {
		int64 ndat = ashape[0];
		int64 xchn = ashape[1];
		int64 xh = ashape[2];
		int64 xw = ashape[3];

		VTensor bgrad = tracer.createTensor(session(), b.shape(), type(), device());

		float* pgb = bgrad.float_ptr();
		float* pgy = ygrad.float_ptr();

		CALL_MATH(add_2d_bias_backward_b, device(), pgb, pgy, ndat, xchn, xh, xw, false);

		pQueue->regist(b, bgrad);
	}
	else pQueue->registNoGrad(b);
}

VTensor VTensor::add_residual(VTensor second, VExecTracer tracer) {
	VTensor a = *this;
	VTensor b = second;

	if (b.hasNoData()) {
		VTensor y = tracer.createTensor(a, a.shape(), TensorCloneInit::share);
		return y;
	}

	VShape ashape = a.shape();
	VShape bshape = b.shape();

	int64 nchn1 = ashape[-1];
	int64 nchn2 = bshape[-1];

	int64 ndat = ashape.total_size() / nchn1;
	int64 nrest = 1;

	if (ashape.size() == 4) {
		if (ashape.remove_nth(1) != bshape.remove_nth(1)) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (ashape[1] % bshape[1] != 0) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

		nchn1 = ashape[1];
		nchn2 = bshape[1];

		ndat = ashape[0];
		nrest = ashape[2] * ashape[3];
	}
	else {
		if (ashape.remove_end() != bshape.remove_end()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (ashape[-1] % bshape[-1] != 0) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}

	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), ashape, type(), device());

	float* py = y.float_ptr();
	float* pa = a.float_ptr();
	float* pb = b.float_ptr();

	if (ashape == bshape) {
		int64 ndat = ashape.total_size();
		CALL_MATH(add, device(), py, pa, pb, ndat);
	}
	else {
		CALL_MATH(add_residual, device(), py, pa, pb, ndat, nchn1, nchn2, nrest);
	}

	return y;
}

void VTensor::add_residual_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor a = m_core->m_operands[0];
	VTensor b = m_core->m_operands[1];

	VShape ashape = a.shape();
	VShape bshape = b.shape();

	if (a.needGrad()) {
		VTensor agrad = tracer.createTensor(ygrad, ashape, TensorCloneInit::share);
		pQueue->regist(a, agrad);
	}
	else pQueue->registNoGrad(a);

	if (b.needGrad()) {
		if (ashape == bshape) {
			VTensor bgrad = tracer.createTensor(ygrad, bshape, TensorCloneInit::share);
			pQueue->regist(b, bgrad);
		}
		else {
			int64 nchn1 = ashape[-1];
			int64 nchn2 = bshape[-1];
			int64 ndat = ashape.total_size() / nchn1;
			int64 nrest = 1;

			if (ashape.size() == 4) {
				nchn1 = ashape[1];
				nchn2 = bshape[1];
				ndat = ashape[0];
				nrest = ashape[2] * ashape[3];
			}

			VTensor bgrad = tracer.createTensor(session(), b.shape(), type(), device());

			float* pgb = bgrad.float_ptr();
			float* pgy = ygrad.float_ptr();

			CALL_MATH(add_residual_backward_b, device(), pgb, pgy, ndat, nchn1, nchn2, nrest, false);

			pQueue->regist(b, bgrad);
		}
	}
	else pQueue->registNoGrad(b);
}

VTensor VTensor::subtract(VTensor second, VExecTracer tracer) {
	VTensor a = *this;
	VTensor b = second;

	VShape ashape = a.shape();
	VShape bshape = b.shape();

	if (ashape.total_size() != bshape.total_size()) {
		if (bshape.size() != 1 || ashape[-1] != bshape[-1]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 ndat = ashape.total_size();
	int64 ncol = ashape[-1];
	int64 nrow = ndat / ncol;

	VTensor y = tracer.createTensor(session(), ashape, type(), device());

	float* py = y.float_ptr();
	float* pa = a.float_ptr();
	float* pb = b.float_ptr();

	if (ashape.total_size() == bshape.total_size()) {
		CALL_MATH(subtract, device(), py, pa, pb, ndat);
	}
	else {
		CALL_MATH(subtract_bias, device(), py, pa, pb, ndat, nrow, ncol);
	}

	return y;
}

void VTensor::subtract_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor a = m_core->m_operands[0];
	VTensor b = m_core->m_operands[1];

	if (a.needGrad()) {
		VTensor agrad = ygrad;
		pQueue->regist(a, agrad);
	}
	else pQueue->registNoGrad(a);

	if (b.needGrad()) {
		if (a.shape() == b.shape()) {
			int64 ndat = a.shape().total_size();
			int64 ncol = b.shape()[-1];
			int64 nrow = ndat / ncol;

			VTensor bgrad = tracer.createTensor(session(), b.shape(), type(), device());

			float* pgb = bgrad.float_ptr();
			float* pgy = ygrad.float_ptr();

			CALL_MATH(subtract_backward_b, device(), pgb, pgy, ndat);

			pQueue->regist(b, bgrad);
		}
		else {
			int64 ndat = a.shape().total_size();
			int64 ncol = b.shape()[-1];
			int64 nrow = ndat / ncol;

			VTensor bgrad = tracer.createTensor(session(), b.shape(), type(), device());

			float* pgb = bgrad.float_ptr();
			float* pgy = ygrad.float_ptr();

			CALL_MATH(subtract_bias_backward_b, device(), pgb, pgy, ndat, nrow, ncol);

			pQueue->regist(b, bgrad);
		}
	}
	else pQueue->registNoGrad(b);
}

VShape VTensor::m_binop_shape_check(bool bCheckSE, VShape& shape1, VShape& shape2, int64& left1, int64& left2, int64& mid, int64& right1, int64& right2) {
	while (shape1.size() > 1 && shape1[-1] == 1) shape1 = shape1.remove_end();
	while (shape2.size() > 1 && shape2[-1] == 1) shape2 = shape2.remove_end();

	if (shape1 == shape2) {
		left1 = left2 = right1 = right2 = 1;
		mid = shape1.total_size();
		return shape1;
	}
	else if (shape2.total_size() == 1) {
		left1 = shape1.total_size();
		left2 = mid = right1 = right2 = 1;
		return shape1;
	}
	else if (shape1.total_size() == 1) {
		left2 = shape1.total_size();
		left1 = mid = right1 = right2 = 1;
		return shape2;
	}
	else if (shape1.size() > shape2.size()) {
		VShape head_shape = shape1.cut_tail(shape1.size() - shape2.size());
		VShape tail_shape = shape1.tail(shape1.size() - shape2.size());

		if (head_shape == shape2) {
			left1 = left2 = right2 = 1;
			mid = shape2.total_size();
			right1 = shape1.total_size() / mid;
		}
		else if (tail_shape == shape2) {
			left2 = right1 = right2 = 1;
			mid = shape2.total_size();
			left1 = shape1.total_size() / mid;
		}
		else {
			VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		}
		return shape1;
	}
	else if (shape1.size() < shape2.size()) {
		VShape head_shape = shape1.cut_tail(shape2.size() - shape1.size());
		VShape tail_shape = shape1.tail(shape2.size() - shape1.size());

		if (head_shape == shape1) {
			left1 = left2 = right1 = 1;
			mid = shape1.total_size();
			right2 = shape2.total_size() / mid;
		}
		else if (tail_shape == shape2) {
			left1 = right1 = right2 = 1;
			mid = shape1.total_size();
			left2 = shape2.total_size() / mid;
		}
		else {
			VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		}
		return shape2;
	}
	else if (bCheckSE && shape1.size() == 4) {	// squeeze_excitation 처리를 위한 마스크 곱셈에 한하여 지원
		if (shape2[0] != shape1[0]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (shape2[1] != 1) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (shape2[2] != 1) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (shape2[3] != shape1[3]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

		left1 = -1;	// squeeze_excitation 처리가 필요함을 알려주는 다소의 편법

		return shape1;
	}
	else {
		VP_THROW(VERR_CONDITIONAL_STATEMENT);
	}
}

VTensor VTensor::mult(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(true, shape1, shape2, left1, left2, mid, right1, right2);
	
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(mult, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

void VTensor::mult_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(true, shape1, shape2, left1, left2, mid, right1, right2);

	float* pgy = ygrad.isValid() ? ygrad.float_ptr() : NULL;
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	if (x1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x1.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(mult_backward_x1, device(), pgx, pgy, px2, left1, left2, mid, right1, right2);

		pQueue->regist(x1, xgrad);
	}
	else pQueue->registNoGrad(x1);

	if (x2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x2.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(mult_backward_x2, device(), pgx, pgy, px1, left1, left2, mid, right1, right2);

		pQueue->regist(x2, xgrad);
	}
	else pQueue->registNoGrad(x2);
}

VTensor VTensor::se_mult(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	if (shape1.size() != 4 || shape2.size() != 4) VP_THROW(VERR_UNDEFINED);
	if (shape1[0] != shape2[0]) VP_THROW(VERR_UNDEFINED);
	if (shape1[1] != shape2[1]) VP_THROW(VERR_UNDEFINED);
	if (shape2[2] != 1) VP_THROW(VERR_UNDEFINED);
	if (shape2[3] != 1) VP_THROW(VERR_UNDEFINED);

	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), shape1, type(), device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(mult_se_mask, device(), py, px1, px2, shape1[0], shape1[1], shape1[2], shape1[3]);

	return y;
}

void VTensor::se_mult_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	float* pgy = ygrad.isValid() ? ygrad.float_ptr() : NULL;
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	if (x1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x1.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(mult_se_mask_backward_x1, device(), pgx, pgy, px2, shape1[0], shape1[1], shape1[2], shape1[3]);

		pQueue->regist(x1, xgrad);
	}
	else pQueue->registNoGrad(x1);

	if (x2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x2.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(mult_se_mask_backward_x2, device(), pgx, pgy, px1, shape1[0], shape1[1], shape1[2], shape1[3]);

		pQueue->regist(x2, xgrad);
	}
	else pQueue->registNoGrad(x2);
}

VTensor VTensor::div(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(div, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

void VTensor::div_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	float* pgy = ygrad.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	if (x1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x1.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(div_backward_x1, device(), pgx, pgy, px2, left1, left2, mid, right1, right2);

		pQueue->regist(x1, xgrad);
	}

	if (x2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x2.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(div_backward_x2, device(), pgx, pgy, px1, px2, left1, left2, mid, right1, right2);

		pQueue->regist(x2, xgrad);
	}
}

VTensor VTensor::maximum(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(maximum, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

void VTensor::maximum_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	float* pgy = ygrad.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	if (x1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x1.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(maximum_backward_x1, device(), pgx, pgy, px1, px2, left1, left2, mid, right1, right2);

		pQueue->regist(x1, xgrad);
	}
	else pQueue->registNoGrad(x1);

	if (x2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x2.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(maximum_backward_x2, device(), pgx, pgy, px1, px2, left1, left2, mid, right1, right2);

		pQueue->regist(x2, xgrad);
	}
	else pQueue->registNoGrad(x2);
}

VTensor VTensor::minimum(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(minimum, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

void VTensor::minimum_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	float* pgy = ygrad.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	if (x1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x1.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(minimum_backward_x1, device(), pgx, pgy, px1, px2, left1, left2, mid, right1, right2);

		pQueue->regist(x1, xgrad);
	}
	else pQueue->registNoGrad(x1);

	if (x2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x2.shape(), type(), device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(minimum_backward_x2, device(), pgx, pgy, px1, px2, left1, left2, mid, right1, right2);

		pQueue->regist(x2, xgrad);
	}
	else pQueue->registNoGrad(x2);
}

VTensor VTensor::_not(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, VDataType::float32, x.device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(_not, device(), py, px, ndat);

	return y;
}

VTensor VTensor::_and(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(_and, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

VTensor VTensor::_or(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(_or, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

VTensor VTensor::equal(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	float val = x.getOpArg("0");

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, VDataType::float32, x.device());

	if (x.type() == VDataType::float32) {
		float* py = y.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(equal_const, device(), py, px, val, ndat);
	}
	else if (x.type() == VDataType::int32) {
		float* py = y.float_ptr();
		float* px = (float*)x.int_ptr();

		CALL_MATH(equal_const, device(), py, px, val, ndat);
	}
	else {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32 or int32", VDataTypeName(x.type()));
	}

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(equal_const, device(), py, px, val, ndat);

	return y;
}

VTensor VTensor::equal(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);
	VTensor y;

	if (x1.type() == VDataType::float32) {
		if (x2.type() != VDataType::float32) {
			VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
		}
		if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

		y = tracer.createTensor(session(), yshape, VDataType::float32, device());

		float* py = y.float_ptr();
		float* px1 = x1.float_ptr();
		float* px2 = x2.float_ptr();

		CALL_MATH(equal, device(), py, px1, px2, left1, left2, mid, right1, right2);
	}
	else if (x1.type() == VDataType::int32) {
		if (x2.type() != VDataType::int32) {
			VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(x2.type()));
		}
		if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

		y = tracer.createTensor(session(), yshape, VDataType::float32, device());

		// 4바이트로 크기가 같은 타입이며 int 값이 같으면 float 형변환해도 같을테니 math 함수 추가 없이 이렇게 처리해봄
		float* py = y.float_ptr();
		float* px1 = (float*)x1.int_ptr();
		float* px2 = (float*)x2.int_ptr();

		CALL_MATH(equal, device(), py, px1, px2, left1, left2, mid, right1, right2);
	}
	else {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32 or int32", VDataTypeName(x1.type()));
	}

	return y;
}

VTensor VTensor::greater_than(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	if (x.type() == VDataType::float32) {
		float val = x.getOpArg("0");

		int64 ndat = shape.total_size();

		VTensor y = tracer.createTensor(session(), shape, VDataType::float32, x.device());

		float* py = y.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(greater_than_float_const, device(), py, px, val, ndat);

		return y;
	}
	else if (x.type() == VDataType::int32) {
		int val = x.getOpArg("0");

		int64 ndat = shape.total_size();

		VTensor y = tracer.createTensor(session(), shape, VDataType::float32, x.device());

		float* py = y.float_ptr();
		int* px = x.int_ptr();

		CALL_MATH(greater_than_int_const, device(), py, px, val, ndat);

		return y;
	}
	else {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}
}

VTensor VTensor::greater_than(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(greater_than, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

VTensor VTensor::less_than(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	float val = x.getOpArg("0");

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, VDataType::float32, x.device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(less_than_const, device(), py, px, val, ndat);

	return y;
}

VTensor VTensor::less_than(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(less_than, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

VTensor VTensor::greater_equal(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	float val = x.getOpArg("0");

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, VDataType::float32, x.device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(greater_equal_const, device(), py, px, val, ndat);

	return y;
}

VTensor VTensor::greater_equal(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(greater_equal, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

VTensor VTensor::less_equal(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	float val = x.getOpArg("0");

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, VDataType::float32, x.device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(less_equal_const, device(), py, px, val, ndat);

	return y;
}

VTensor VTensor::less_equal(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 left1, left2, mid, right1, right2;
	VShape yshape = m_binop_shape_check(false, shape1, shape2, left1, left2, mid, right1, right2);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}
	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(less_equal, device(), py, px1, px2, left1, left2, mid, right1, right2);

	return y;
}

VTensor VTensor::greater_than_cross(VTensor second, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VTensor VTensor::less_than_cross(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	if (shape1[0] != shape2[0]) VP_THROW(VERR_BAD_SHAPE_TENSOR);
	if (shape1.total_size() != shape2.total_size()) VP_THROW(VERR_BAD_SHAPE_TENSOR);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}

	if (device() != second.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 nrow = shape1[0];
	int64 ncol = shape1.total_size() / shape1[0];

	VTensor y = tracer.createTensor(session(), { nrow, ncol, ncol }, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(less_than_cross, device(), py, px1, px2, nrow, ncol);

	return y;
}

VTensor VTensor::greater_equal_cross(VTensor second, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VTensor VTensor::less_equal_cross(VTensor second, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VTensor VTensor::abs(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ndat = xshape.total_size();

	VTensor y = tracer.createTensor(session(), xshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(abs, device(), py, px, ndat);

	return y;
}

void VTensor::abs_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 ndat = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(abs_backward, device(), pgx, pgy, px, ndat);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::square(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ndat = xshape.total_size();

	VTensor y = tracer.createTensor(session(), xshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(square, device(), py, px, ndat);

	return y;
}

void VTensor::square_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 ndat = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(square_backward, device(), pgx, pgy, px, ndat);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::sqrt(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ndat = xshape.total_size();

	VTensor y = tracer.createTensor(session(), xshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(sqrt, device(), py, px, ndat);

	return y;
}

void VTensor::sqrt_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 ndat = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(sqrt_backward, device(), pgx, pgy, px, ndat);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::exp(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, x.type(), x.device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(exp, device(), py, px, ndat);

	return y;
}

void VTensor::exp_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 ndat = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(exp_backward, device(), pgx, pgy, px, ndat);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::log(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, x.type(), x.device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(log, device(), py, px, ndat);

	return y;
}

void VTensor::log_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 ndat = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(log_backward, device(), pgx, pgy, px, ndat);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::sigmoid(VExecTracer tracer) {
	VTensor logits = *this;

	VShape shape = logits.shape();

	if (logits.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(logits.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, type(), device());

	float* py = y.float_ptr();
	float* px = logits.float_ptr();

	CALL_MATH(sigmoid, device(), py, px, ndat);

	return y;
}

void VTensor::sigmoid_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 ndat = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = x.float_ptr();

		CALL_MATH(sigmoid_backward, device(), pgx, pgy, px, ndat);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::upsample(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape stride = x.getOpArg("stride");

	if (xshape.size() != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (stride.size() != 2) VP_THROW(VERR_SIZE_STRIDE);

	int64 nbat = xshape[0];
	int64 nchn = xshape[1];
	int64 nxht = xshape[2];
	int64 nxwd = xshape[3];

	int64 nyht = nxht * stride[0];
	int64 nywd = nxwd * stride[1];

	VTensor y = tracer.createTensor(session(), VShape{ nbat, nchn, nyht, nywd}, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(upsample, device(), py, px, nbat, nchn, nyht, nywd, stride[0], stride[1]);

	if (0) {
		x.dump1("upsample x");
		y.dump1("upsample y");
	}

	return y;
}

void VTensor::upsample_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();
		VShape stride = getMyArg("stride");

		int64 nbat = xshape[0];
		int64 nchn = xshape[1];
		int64 nxht = xshape[2];
		int64 nxwd = xshape[3];

		int64 nyht = nxht * stride[0];
		int64 nywd = nxwd * stride[1];

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();

		CALL_MATH(upsample_backward, device(), pgx, pgy, nbat, nchn, nxht, nxwd, stride[0], stride[1]);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::pass(VExecTracer tracer) {
	VTensor x = *this;
	VShape xshape = x.shape();

	string dump_title = x.getOpArg("dump");
	string direction = x.getOpArg("direction");
	bool exception = x.getOpArg("exception");

	if (direction != "backward") {
		if (dump_title != "") {
			x.dump1(dump_title);
		}
		if (exception) {
			VP_THROW(VERR_UNDEFINED);
		}
	}

	VTensor y = tracer.createTensor(x, xshape, TensorCloneInit::share);

	return y;
}

void VTensor::pass_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	string dump_title = getMyArg("dump");
	string direction = getMyArg("direction");
	bool exception = getMyArg("exception");

	if (direction != "forward") {
		if (dump_title != "") {
			ygrad.dump(dump_title + "_grad");
		}
		if (exception) {
			VP_THROW(VERR_UNDEFINED);
		}
	}

	if (x.needGrad()) pQueue->regist(x, ygrad);
	else pQueue->registNoGrad(x);
}

VTensor VTensor::iou_cross_xywh(VTensor second, VExecTracer tracer) {
	VTensor boxes1 = *this;
	VTensor boxes2 = second;

	VShape shape1 = boxes1.shape();
	VShape shape2 = boxes2.shape();

	if (shape1.size() != 3 || shape2.size() != 3) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape1[0] != shape2[0]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape1[-1] != 4 || shape2[-1] != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (boxes1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(boxes1.type()));
	}
	if (boxes2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(boxes2.type()));
	}
	if (boxes1.device() != boxes2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 nbat = shape1[0];
	int64 nrow = shape1[1];
	int64 ncol = shape2[1];

	VTensor y = tracer.createTensor(session(), shape1.replace_end(ncol), type(), device());

	float* py = y.float_ptr();
	float* px1 = boxes1.float_ptr();
	float* px2 = boxes2.float_ptr();

	CALL_MATH(iou_cross_xywh, device(), py, px1, px2, nbat, nrow, ncol);

	if (0) {
		boxes1.dump("iou_cross_xywh(boxes1)");
		boxes1.sum(tracer).dump("sum(iou_cross_xywh(boxes1))");

		boxes2.dump("iou_cross_xywh(boxes2)");
		boxes2.sum(tracer).dump("sum(iou_cross_xywh(boxes2))");

		y.dump("iou_cross_xywh");
		y.sum(tracer).dump("sum(iou_cross_xywh)");
	}

	return y;
}

VTensor VTensor::iou_cross_lrtb(VTensor second, VExecTracer tracer) {
	VTensor boxes1 = *this;
	VTensor boxes2 = second;

	VShape shape1 = boxes1.shape();
	VShape shape2 = boxes2.shape();

	if (shape1.size() != 3 || shape2.size() != 3) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape1[0] != shape2[0]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape1[-1] != 4 || shape2[-1] != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (boxes1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(boxes1.type()));
	}
	if (boxes2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(boxes2.type()));
	}
	if (boxes1.device() != boxes2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 nbat = shape1[0];
	int64 nrow = shape1[1];
	int64 ncol = shape2[1];

	VTensor y = tracer.createTensor(session(), shape1.replace_end(ncol), type(), device());

	float* py = y.float_ptr();
	float* px1 = boxes1.float_ptr();
	float* px2 = boxes2.float_ptr();

	CALL_MATH(iou_cross_lrtb, device(), py, px1, px2, nbat, nrow, ncol);

	return y;
}

VTensor VTensor::iou_loss(VTensor second, VGraphOpCode op_code, VExecTracer tracer) {
	VTensor boxes1 = *this;
	VTensor boxes2 = second;

	VShape shape1 = boxes1.shape();
	VShape shape2 = boxes2.shape();

	if (shape1.total_size() != shape2.total_size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape1[-1] != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape2[-1] != 4) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (boxes1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(boxes1.type()));
	}
	if (boxes2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(boxes2.type()));
	}
	if (boxes1.device() != boxes2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VShape yshape = shape1.remove_end();

	int64 nrow = yshape.total_size();

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	float* py = y.float_ptr();
	float* px1 = boxes1.float_ptr();
	float* px2 = boxes2.float_ptr();

	CALL_MATH(iou_loss, device(), py, px1, px2, nrow, (int)op_code);

	return y;
}

void VTensor::iou_loss_backprop(VTensor ygrad, VBackQueue* pQueue, VGraphOpCode op_code, VExecTracer tracer) {
	VTensor boxes1 = m_core->m_operands[0];
	VTensor boxes2 = m_core->m_operands[1];

	float* pgy = ygrad.float_ptr();
	float* px1 = boxes1.float_ptr();
	float* px2 = boxes2.float_ptr();

	int64 nrow = ygrad.shape().total_size();

	if (boxes1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), boxes1.shape(), boxes1.type(), boxes1.device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(iou_loss_backward, device(), pgx, pgy, px1, px2, nrow, (int)op_code);

		pQueue->regist(boxes1, xgrad);
	}
	else pQueue->registNoGrad(boxes1);

	if (boxes2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), boxes1.shape(), boxes1.type(), boxes1.device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(iou_loss_backward, device(), pgx, pgy, px2, px1, nrow, (int)op_code);

		pQueue->regist(boxes2, xgrad);
	}
	else pQueue->registNoGrad(boxes2);
}

VTensor VTensor::crossentropy(VTensor second, VExecTracer tracer) {
	VTensor logits = *this;
	VTensor labels = second;

	VShape shape1 = logits.shape();
	VShape shape2 = labels.shape();

	if (shape2.size() == shape1.size()) {
		if (shape1.total_size() / shape1[-1] != shape2.total_size() / shape2[-1]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (shape1[-1] != shape2[-1] && shape2[-1] != 1) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}
	else if (shape2.size() == shape1.size() - 1) {
		if (shape1.total_size() / shape1[-1] != shape2.total_size()) {
			if (1) {
				printf("shape1: %s vs. shape2:%s\n", shape1.desc().c_str(), shape2.desc().c_str());
			}
			VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		}
	}
	else VP_THROW(VERR_UNMATCHED_SHAPE_IN_CROSSENTROPY);

	if (logits.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(logits.type()));
	}
	if (device() != labels.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 ncol = shape1[-1];
	int64 nrow = shape1.total_size() / ncol;

	VTensor y = tracer.createTensor(session(), shape1.remove_end(), type(), device());

	float* py = y.float_ptr();
	float* px = logits.float_ptr();

	if (labels.type() == VDataType::int32) {
		int* pz = labels.int_ptr();

		CALL_MATH(softmax_idx_crossentropy, device(), py, px, pz, nrow, ncol);
	}
	else if (labels.type() == VDataType::int64) {
		int64* pz = labels.int64_ptr();

		CALL_MATH(softmax_i64_idx_crossentropy, device(), py, px, pz, nrow, ncol);
	}
	else if (shape1[1] == 1) {
		float* pz = labels.float_ptr();

		CALL_MATH(sigmoid_crossentropy, device(), py, px, pz, nrow);
	}
	else if (shape1[1] == shape2[1]) {
		float* pz = labels.float_ptr();

		CALL_MATH(softmax_crossentropy, device(), py, px, pz, nrow, ncol);
	}
	else {
		VP_THROW(VERR_CONDITIONAL_STATEMENT);
	}

	return y;
}

void VTensor::crossentropy_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor logits = m_core->m_operands[0];
	VTensor labels = m_core->m_operands[1];

	if (logits.needGrad()) {
		int64 nrow = logits.shape()[0];
		int64 ncol = logits.shape()[1];

		VTensor xgrad = tracer.createTensor(session(), logits.shape(), logits.type(), logits.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = logits.float_ptr();

		if (labels.type() == VDataType::int32) {
			int* pz = labels.int_ptr();

			CALL_MATH(softmax_idx_crossentropy_backward_x, device(), pgx, pgy, px, pz, nrow, ncol);
		}
		else if (labels.type() == VDataType::int64) {
			int64* pz = labels.int64_ptr();

			CALL_MATH(softmax_i64_idx_crossentropy_backward_x, device(), pgx, pgy, px, pz, nrow, ncol);
		}
		else if (logits.shape()[1] == 1) {
			float* pz = labels.float_ptr();

			CALL_MATH(sigmoid_crossentropy_backward_x, device(), pgx, pgy, px, pz, nrow);
		}
		else if (logits.shape()[1] == labels.shape()[1]) {
			float* pz = labels.float_ptr();

			CALL_MATH(softmax_crossentropy_backward_x, device(), pgx, pgy, px, pz, nrow, ncol);
		}
		else {
			VP_THROW(VERR_CONDITIONAL_STATEMENT);
		}

		pQueue->regist(logits, xgrad);
	}
	else pQueue->registNoGrad(logits);

	if (labels.needGrad()) {
		VP_THROW(VERR_UNDEFINED);
	}
	else pQueue->registNoGrad(labels);
}

VTensor VTensor::crossentropy_sigmoid(VTensor second, VExecTracer tracer) {
	VTensor logits = *this;
	VTensor labels = second;

	VShape shape1 = logits.shape();
	VShape shape2 = labels.shape();

	if (shape2.total_size() != shape1.total_size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (logits.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(logits.type()));
	}
	if (labels.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(labels.type()));
	}
	if (device() != labels.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 nrow = shape1.total_size();

	VTensor y = tracer.createTensor(session(), shape1, type(), device());

	float* py = y.float_ptr();
	float* px = logits.float_ptr();
	float* pz = labels.float_ptr();

	CALL_MATH(sigmoid_crossentropy, device(), py, px, pz, nrow);

	return y;
}

void VTensor::crossentropy_sigmoid_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor logits = m_core->m_operands[0];
	VTensor labels = m_core->m_operands[1];

	if (logits.needGrad()) {
		int64 nrow = logits.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), logits.shape(), logits.type(), logits.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = logits.float_ptr();
		float* pz = labels.float_ptr();

		CALL_MATH(sigmoid_crossentropy_backward_x, device(), pgx, pgy, px, pz, nrow);

		pQueue->regist(logits, xgrad);
	}
	else pQueue->registNoGrad(logits);

	if (labels.needGrad()) {
		VP_THROW(VERR_UNDEFINED);
	}
	else pQueue->registNoGrad(labels);
}

VTensor VTensor::crossentropy_pos_idx(VTensor second, VExecTracer tracer) {
	VTensor logits = *this;
	VTensor labels = second;

	VShape shape1 = logits.shape();
	VShape shape2 = labels.shape();

	if (shape1.total_size() / shape1[-1] != shape2.total_size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (logits.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(logits.type()));
	}
	if (labels.type() != VDataType::int32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(labels.type()));
	}

	if (device() != labels.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 ncol = shape1[-1];
	int64 nrow = shape1.total_size() / ncol;

	VTensor y = tracer.createTensor(session(), shape1.remove_end(), type(), device());

	float* py = y.float_ptr();
	float* px = logits.float_ptr();

	int* pz = labels.int_ptr();

	CALL_MATH(softmax_idx_crossentropy_pos_idx, device(), py, px, pz, nrow, ncol);

	return y;
}

void VTensor::crossentropy_pos_idx_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor logits = m_core->m_operands[0];
	VTensor labels = m_core->m_operands[1];

	if (logits.needGrad()) {
		int64 nrow = logits.shape()[0];
		int64 ncol = logits.shape()[1];

		VTensor xgrad = tracer.createTensor(session(), logits.shape(), logits.type(), logits.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		float* px = logits.float_ptr();
		int* pz = labels.int_ptr();

		CALL_MATH(softmax_idx_crossentropy_pos_idx_backward_x, device(), pgx, pgy, px, pz, nrow, ncol);

		pQueue->regist(logits, xgrad);
	}
	else pQueue->registNoGrad(logits);

	if (labels.needGrad()) {
		VP_THROW(VERR_UNDEFINED);
	}
	else pQueue->registNoGrad(labels);
}

VTensor VTensor::sigmoid_crossentropy_with_logits(VExecTracer tracer) {
	VTensor logits = *this;

	VShape shape = logits.shape();

	float z = logits.getOpArg("0");

	if (logits.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(logits.type()));
	}

	int64 ndat = shape.total_size();

	VTensor y = tracer.createTensor(session(), shape, type(), device());

	float* py = y.float_ptr();
	float* px = logits.float_ptr();

	CALL_MATH(sigmoid_crossentropy_with_logits, device(), py, px, z, ndat);

	return y;
}

void VTensor::sigmoid_crossentropy_with_logits_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor logits = m_core->m_operands[0];
	float z = getMyArg("0");

	if (logits.needGrad()) {
		VShape shape = logits.shape();

		int64 ndat = shape.total_size();

		VTensor xgrad = tracer.createTensor(session(), shape, type(), device());

		float* pgy = ygrad.float_ptr();
		float* pgx = xgrad.float_ptr();
		float* px = logits.float_ptr();

		CALL_MATH(sigmoid_crossentropy_with_logits_backward, device(), pgx, pgy, px, z, ndat);

		pQueue->regist(logits, xgrad);
	}
	else pQueue->registNoGrad(logits);
}

VTensor VTensor::sigmoid_crossentropy_with_logits_idx(VTensor second, VExecTracer tracer) {
	VTensor logits = *this;
	VTensor labels = second;

	VShape shape1 = logits.shape();
	VShape shape2 = labels.shape();

	if (shape1.total_size() / shape1[-1] != shape2.total_size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (logits.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(logits.type()));
	}
	if (labels.type() != VDataType::int32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(labels.type()));
	}

	if (device() != labels.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 ncol = shape1[-1];
	int64 nrow = shape1.total_size() / ncol;

	VTensor y = tracer.createTensor(session(), shape1, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px = logits.float_ptr();

	int* pz = labels.int_ptr();

	CALL_MATH(sigmoid_crossentropy_with_logits_idx, device(), py, px, pz, nrow, ncol);

	return y;
}

void VTensor::sigmoid_crossentropy_with_logits_idx_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor logits = m_core->m_operands[0];
	VTensor labels = m_core->m_operands[1];

	if (logits.needGrad()) {
		VShape shape = logits.shape();

		int64 ncol = shape[-1];
		int64 nrow = shape.total_size() / ncol;

		VTensor xgrad = tracer.createTensor(session(), shape, type(), device());

		float* pgy = ygrad.float_ptr();
		float* pgx = xgrad.float_ptr();
		float* px = logits.float_ptr();
		int* pz = labels.int_ptr();

		CALL_MATH(sigmoid_crossentropy_with_logits_idx_backward, device(), pgx, pgy, px, pz, nrow, ncol);

		pQueue->regist(logits, xgrad);
	}
	else pQueue->registNoGrad(logits);

	if (labels.needGrad()) {
		VP_THROW(VERR_UNDEFINED);
	}
	else pQueue->registNoGrad(labels);
}

VTensor VTensor::reshape(VExecTracer tracer) {
	VTensor x = *this;
	VShape xshape = x.shape();
	VShape tshape = x.getOpArg("shape");

	tshape = tshape.copy();	// 병렬처리 과정에서 동일한 형상 객체에 접근하게 될 수 있으므로 복사.

	if (tshape.total_size() < 0) tshape = tshape.resolve_plcaeholder(xshape.total_size());
	else if (xshape.total_size() != tshape.total_size()) {
		if (xshape.remove_head().total_size() != tshape.remove_head().total_size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		tshape[0] = xshape[0];
	}

	VTensor y = tracer.createTensor(x, tshape, TensorCloneInit::share);

	return y;
}

void VTensor::reshape_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();
		VTensor xgrad = tracer.createTensor(ygrad, xshape, TensorCloneInit::share);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

void VTensor::flatten_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();
		VTensor xgrad = tracer.createTensor(ygrad, xshape, TensorCloneInit::share);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::transpose(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();
	VShape tshape;

	VList axes = x.getOpArg("axes");

	int64 axis_size = xshape.size();
	int64 data_size = xshape.total_size();

	if (axis_size != axes.size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	int mask1 = 0;
	int mask2 = 0;

	int64 rest = data_size;
	int64 prod = data_size;

	VTensor axinfo(session(), VShape{ axis_size, 3 }, VDataType::int32, device());

	for (int n = 0; n < axis_size; n++) {
		int axis = (int)axes[n];

		if (axis < 0 || axis >= axis_size) VP_THROW(VERR_OUT_OF_RANGE);
		
		tshape = tshape.append(xshape[axis]);

		mask1 |= (1 << n);
		mask2 |= (1 << axis);

		rest /= (int64)xshape[axis];
		prod /= (int64)xshape[n];

		// 설정 내용은 메모리가 유지되지 않는 로컬 변수 내용이므로 트레이싱 대상에서 제외한다.
		VExecTracer emptyTracer;

		axinfo.setElement(VList({ n, 0 }), axis, emptyTracer);
		axinfo.setElement(VList({ n, 1 }), rest, emptyTracer);
		axinfo.setElement(VList({ n, 2 }), prod, emptyTracer);
	}

	if (mask1 != mask2) VP_THROW(VERR_TENSOR_MASK);

	// tracer.createTensor() 함수로 axinfo 텐서를 직접 생성하면 초기화 기능이 recording되어 재생시 위에서 게산해 설정한 정보가 사라ㅈ져 오류를 발생시킨다.
	tracer.addTensor(axinfo);

	VTensor y = tracer.createTensor(session(), tshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	int* pn = axinfo.int_ptr();

	CALL_MATH(transpose, device(), py, px, pn, axis_size, data_size);

	m_core->m_opArgs["axinfo"] = axinfo.cloneCore();
	
	return y;
}

void VTensor::transpose_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();

		VTensor axinfo = tracer.createTensor(session(), (VTensorCore*)(VObjCore*)getMyArg("axinfo"));

		VTensor xgrad = VTensor(session(), xshape, x.type(), x.device());

		int64 axis_size = axinfo.shape()[0];
		int64 data_size = xshape.total_size();

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();
		int* pn = axinfo.int_ptr();

		CALL_MATH(transpose_backward, device(), pgx, pgy, pn, axis_size, data_size);

		//xgrad.dump("transpose_backprop.xgrad");
		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

void VTensor::transpose_on(VTensor x, VList axes, VExecTracer tracer) {
	VTensor y = *this;

	VShape xshape = x.shape();
	VShape yshape;

	int64 axis_size = xshape.size();
	int64 data_size = xshape.total_size();

	if (axis_size != axes.size()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	int mask1 = 0;
	int mask2 = 0;

	int64 rest = data_size;
	int64 prod = data_size;

	VTensor axinfo = tracer.createTensor(session(), VShape{ axis_size, 3 }, VDataType::int32, device());

	for (int n = 0; n < axis_size; n++) {
		int axis = (int)axes[n];

		if (axis < 0 || axis >= axis_size) VP_THROW(VERR_OUT_OF_RANGE);

		yshape = yshape.append(xshape[axis]);

		mask1 |= (1 << n);
		mask2 |= (1 << axis);

		rest /= (int64)xshape[axis];
		prod /= (int64)xshape[n];

		// 설정 내용은 메모리가 유지되지 않는 로컬 변수 내용이므로 트레이싱 대상에서 제외한다.
		VExecTracer emptyTracer;

		axinfo.setElement(VList({ n, 0 }), axis, emptyTracer);
		axinfo.setElement(VList({ n, 1 }), rest, emptyTracer);
		axinfo.setElement(VList({ n, 2 }), prod, emptyTracer);
	}

	if (mask1 != mask2) VP_THROW(VERR_TENSOR_MASK);
	if (yshape != shape()) {
		VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}

	float* py = y.float_ptr();
	float* px = x.float_ptr();
	int* pn = axinfo.int_ptr();

	CALL_MATH(transpose, device(), py, px, pn, axis_size, data_size);
}

VTensor VTensor::extract(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 axis = x.getOpArg("axis");
	int64 index = x.getOpArg("index");
	int64 count = x.getOpArg("count");

	bool reduce_axis = x.getOpArg("reduce_axis");

	int64 axis_size = xshape.size();

	if (axis == -1) axis = axis_size - 1;
	if (axis < 0 || axis >= axis_size) VP_THROW(VERR_OUT_OF_RANGE);
	if (index < 0 || count <= 0 || index + count > xshape[axis]) VP_THROW(VERR_OUT_OF_RANGE);
	if (count != 1 && reduce_axis) VP_THROW(VERR_UNDEFINED);

	int64 nrow = 1;
	for (int64 n = 0; n < axis; n++) nrow *= xshape[n];

	int64 nxvec = xshape[axis];
	int64 nyvec = count;
	int64 ncol = xshape.total_size() / (nrow * nxvec);

	VShape yshape = reduce_axis ? xshape.remove_nth(axis) : xshape.replace_nth(axis, count);

	VTensor y = tracer.createTensor(session(), yshape, type(), device());

	extract_on(y, x, axis, index, count, reduce_axis, tracer);

	return y;
}

void VTensor::extract_on(VTensor y, VTensor x, int64 axis, int64 index, int64 count, bool reduce_axis, VExecTracer tracer) {
	VShape xshape = x.shape();

	int64 axis_size = xshape.size();

	if (axis == -1) axis = axis_size - 1;
	if (axis < 0 || axis >= axis_size) VP_THROW(VERR_OUT_OF_RANGE);
	if (index < 0 || count <= 0 || index + count > xshape[axis]) VP_THROW(VERR_OUT_OF_RANGE);
	if (count != 1 && reduce_axis) VP_THROW(VERR_UNDEFINED);

	int64 nrow = 1;
	for (int64 n = 0; n < axis; n++) nrow *= xshape[n];

	int64 nxvec = xshape[axis];
	int64 nyvec = count;
	int64 ncol = xshape.total_size() / (nrow * nxvec);

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(extract, x.device(), py, px, nrow, nxvec, index, nyvec, ncol);
}

void VTensor::extract_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		VShape xshape = x.shape();

		int64 axis = getMyArg("axis");
		int64 count = getMyArg("count");
		int64 index = getMyArg("index");

		int64 nrow = 1;
		for (int64 n = 0; n < axis; n++) nrow *= xshape[n];

		int64 nxvec = xshape[axis];
		int64 nyvec = count;
		int64 ncol = xshape.total_size() / (nrow * nxvec);

		VTensor xgrad = VTensor(session(), xshape, x.type(), x.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();

		CALL_MATH(extract_backward, device(), pgx, pgy, nrow, nxvec, index, nyvec, ncol);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::max(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ncol = xshape[-1];
	int64 nrow = xshape.total_size() / ncol;

	VTensor y = tracer.createTensor(session(), xshape.remove_end(), type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(max, device(), py, px, nrow, ncol);

	return y;
}

VTensor VTensor::min(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ncol = xshape[-1];
	int64 nrow = xshape.total_size() / ncol;

	VTensor y = tracer.createTensor(session(), xshape.remove_end(), type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(min, device(), py, px, nrow, ncol);

	return y;
}

VTensor VTensor::argmax(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ncol = xshape[-1];
	int64 nrow = xshape.total_size() / ncol;

	if (x.type() != VDataType::float32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

	VTensor y = tracer.createTensor(session(), xshape.remove_end(), VDataType::int32, device());

	int* py = y.int_ptr();
	float* px = x.float_ptr();

	CALL_MATH(argmax, device(), py, px, nrow, ncol);

	return y;
}

VTensor VTensor::argmin(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ncol = xshape[-1];
	int64 nrow = xshape.total_size() / ncol;

	if (x.type() != VDataType::float32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

	VTensor y = tracer.createTensor(session(), xshape.remove_end(), VDataType::int32, device());

	int* py = y.int_ptr();
	float* px = x.float_ptr();

	CALL_MATH(argmin, device(), py, px, nrow, ncol);

	return y;
}

VTensor VTensor::activate(VExecTracer tracer) {
	VTensor x = *this;

	int64 ncol = shape()[-1];
	int64 nrow = shape().total_size() / ncol;

	ActFunc actFunc = (ActFunc)(int)x.getOpArg("actfunc");
	HYPER_KEY leaky_alpha = x.getOpArg("leaky_alpha");

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(type()));
	}

	VTensor y = tracer.createTensor(session(), shape(), type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(activate, device(), py, px, nrow, ncol, (int)actFunc, HYPER_FETCH(leaky_alpha));

	return y;
}

void VTensor::activate_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	ActFunc actFunc = (ActFunc)(int)getMyArg("actfunc");
	HYPER_KEY leaky_alpha = getMyArg("leaky_alpha");

	if (x.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* px = x.float_ptr();
		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();

		int64 ncol = shape()[-1];
		int64 nrow = shape().total_size() / ncol;

		CALL_MATH(activate_backward, device(), pgx, pgy, px, nrow, ncol, (int)actFunc, HYPER_FETCH(leaky_alpha));

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::normal_noise(VExecTracer tracer) {
	VTensor x = *this;

	int64 ndat = shape().total_size();

	HYPER_KEY mean = x.getOpArg("mean");
	HYPER_KEY std = x.getOpArg("std");

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), shape(), type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(add_normal_noise, device(), py, px, ndat, HYPER_FETCH(mean), HYPER_FETCH(std));

	return y;
}

VTensor VTensor::uniform_noise(VExecTracer tracer) {
	VTensor x = *this;

	int64 ndat = shape().total_size();

	HYPER_KEY min = x.getOpArg("min");
	HYPER_KEY max = x.getOpArg("max");

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), shape(), type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(add_uniform_noise, device(), py, px, ndat, HYPER_FETCH(min), HYPER_FETCH(max));

	x.dump("uniform_noise(x)");
	y.dump("uniform_noise(y)");
	return y;
}

VTensor VTensor::normal_random(VExecTracer tracer) {
	VTensor x = *this;

	VShape rshape = x.getOpArg("shape");
	rshape = rshape.insert_head(x.shape()[0]);

	HYPER_KEY mean = x.getOpArg("mean");
	HYPER_KEY std = x.getOpArg("std");

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), rshape, type(), device());

	int64 ndat = rshape.total_size();

	float* py = y.float_ptr();

	CALL_MATH(gen_normal_random, device(), py, ndat, HYPER_FETCH(mean), HYPER_FETCH(std));

	return y;
}

VTensor VTensor::uniform_random(VExecTracer tracer) {
	VTensor x = *this;

	VShape rshape = x.getOpArg("shape");
	rshape = rshape.insert_head(x.shape()[0]);

	HYPER_KEY min = x.getOpArg("min");
	HYPER_KEY max = x.getOpArg("max");

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), rshape, type(), device());

	int64 ndat = rshape.total_size();

	float* py = y.float_ptr();

	CALL_MATH(gen_uniform_random, device(), py, ndat, HYPER_FETCH(min), HYPER_FETCH(max));

	return y;
}

void VTensor::normal_noise_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	if (x.needGrad()) pQueue->regist(x, ygrad);
	else pQueue->registNoGrad(x);
}

void VTensor::uniform_noise_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	if (x.needGrad()) pQueue->regist(x, ygrad);
	else pQueue->registNoGrad(x);
}

void VTensor::normal_random_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	if (x.needGrad()) pQueue->regist(x, ygrad);
	else pQueue->registNoGrad(x);
}

void VTensor::uniform_random_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];
	if (x.needGrad()) pQueue->regist(x, ygrad);
	else pQueue->registNoGrad(x);
}

VTensor VTensor::codeconv(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ncol = xshape[-1];
	int64 nrow = xshape.total_size() / ncol;

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), xshape.remove_end(), VDataType::int32, device());

	float* px = x.float_ptr();
	int* py = y.int_ptr();

	CALL_MATH(codeconv, device(), py, px, nrow, ncol);

	return y;
}

void VTensor::codeconv_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VTensor VTensor::cosinesim(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape xshape1 = x1.shape();
	VShape xshape2 = x2.shape();

	if (xshape1.size() < 2 || xshape2.size() < 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (xshape1[-1] != xshape2[-1]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	int64 ncol = xshape1[-1];
	int64 nrow1 = xshape1.total_size() / ncol;
	int64 nrow2 = xshape2.total_size() / ncol;

	VTensor y = tracer.createTensor(session(), { nrow1, nrow2 }, VDataType::float32, device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(cosinesim, device(), py, px1, px2, nrow1, nrow2, ncol);

	return y;
}

void VTensor::cosinesim_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VTensor VTensor::selectntop(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ntop = x.getOpArg("ntop", 0);

	int64 ncol = xshape[-1];
	int64 nrow = xshape.total_size() / ncol;

	if (xshape[-1] == 1) xshape = xshape.remove_end();
	if (xshape.size() < 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (xshape[-1] < ntop) VP_THROW(VERR_UNDEFINED);
	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), xshape.replace_end(ntop), type(), device());

	float* px = x.float_ptr();
	float* py = y.float_ptr();

	CALL_MATH(selectntop, device(), py, px, nrow, ncol, ntop);

	return y;
}

void VTensor::selectntop_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VTensor VTensor::selectntoparg(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ntop = x.getOpArg("ntop", 0);

	int64 ncol = xshape[-1];
	int64 nrow = xshape.total_size() / ncol;

	if (xshape[-1] == 1) xshape = xshape.remove_end();
	if (xshape.size() < 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (xshape[-1] < ntop) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), xshape.replace_end(ntop), VDataType::int32, device());

	float* px = x.float_ptr();
	int* py = y.int_ptr();

	CALL_MATH(selectntoparg, device(), py, px, nrow, ncol, ntop);

	return y;
}

void VTensor::selectntoparg_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

VTensor VTensor::round(VExecTracer tracer) {
	VTensor x = *this;

	int prec = x.getOpArg("prec");

	int64 ndat = shape().total_size();

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), shape(), type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(round, device(), py, px, ndat, prec);

	return y;
}

void VTensor::round_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

void VTensor::user_defined_func_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensorList operands = m_core->m_operands;
	VDict opArgs = getMyArgs();

	VFunctionCore* functor = (VFunctionCore*)(int64)opArgs["__functor__"];
	//VDict opArgs = operands[0].getMyArgs();

	VFunction function(functor);

	int nth = 0;
	int nInst = opArgs["__funcinst__"];

	for (auto& x : operands) {
		if (x.needGrad()) {
			VTensor xgrad = function.backward(nInst, ygrad, nth, operands, opArgs);
			pQueue->regist(x, xgrad);
			tracer.addCallBackwardUDF(nInst, *this, ygrad, xgrad, nth);
			nth++;
		}
		else pQueue->registNoGrad(x);
	}
}

VTensor VTensor::mean(VExecTracer tracer) {
	int64 nrow = shape().total_size();

	if (type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(m_core->m_dataType));
	}

	VTensor y = tracer.createTensor(session(), { 1 }, type(), device());

	float* py = y.float_ptr();
	float* px = this->float_ptr();

	CALL_MATH(mean, device(), py, px, nrow);

	return y;
}

void VTensor::mean_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 nrow = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.isValid() ? ygrad.float_ptr() : NULL;

		CALL_MATH(mean_backward, device(), pgx, pgy, nrow);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::sum(VExecTracer tracer) {
	int64 nrow = shape().total_size();

	VTensor y = tracer.createTensor(session(), { 1 }, type(), device());

	if (type() == VDataType::float32) {
		float* py = y.float_ptr();
		float* px = this->float_ptr();

		CALL_MATH(sum, device(), py, px, nrow);
	}
	else if (type() == VDataType::int32) {
		int* py = y.int_ptr();
		int* px = this->int_ptr();

		CALL_MATH(sum_int, device(), py, px, nrow);
	}
	else {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32 or float32", VDataTypeName(m_core->m_dataType));
	}

	return y;
}

void VTensor::sum_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 nrow = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.isValid() ? ygrad.float_ptr() : NULL;

		CALL_MATH(sum_backward, device(), pgx, pgy, nrow);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::stack_all(VSession session, VTensorDict xs, int64 tail_size, VExecTracer tracer) {
	int64 nbat = 0;
	int device = -1;
	int64 ysize = 0;

	for (auto& it : xs) {
		int64 xsize = it.second.shape().remove_head().total_size();
		if (xsize % tail_size != 0) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		if (it.second.type() != VDataType::float32) {
			VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(it.second.type()));
		}
		if (nbat == 0) {
			nbat = it.second.shape()[0];
			device = it.second.device();
		}
		else {
			if (nbat != it.second.shape()[0]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
			if (device != it.second.device()) VP_THROW(VERR_TENSOR_DEVICE);
		}

		ysize += xsize;
	}

	int64 ncol = tail_size;
	int64 nyrow = ysize / ncol;

	VTensor y = tracer.createTensor(session, VShape{ nbat, nyrow, ncol }, VDataType::float32, device);

	float* py = y.float_ptr();

	int64 nfrom = 0;
	for (auto& it : xs) {
		if (0) it.second.dump1("VTensor::stack_all(xs)");
		float* px = it.second.float_ptr();
		int64 nxrow = it.second.shape().total_size() / (nbat * ncol);
		CALL_MATH(stack, device, py, px, nbat, nyrow, nxrow, ncol, nfrom);
		nfrom += nxrow;
	}

	if (0) y.dump1("y");

	return y;
}

void VTensor::stack_all_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VShape yshape = ygrad.shape();
	
	int64 nbat = ygrad.shape()[0];
	int64 nyrow = ygrad.shape()[1];
	int64 ncol = ygrad.shape()[2];

	float* pgy = ygrad.float_ptr();

	int64 nfrom = 0;
	for (auto& it : m_core->m_operands) {
		VTensor xgrad = tracer.createTensor(session(), it.shape(), type(), device());
		float* pgx = xgrad.float_ptr();
		int64 nxrow = it.shape().total_size() / (nbat * ncol);
		CALL_MATH(stack_backward, device(), pgx, pgy, nbat, nyrow, nxrow, ncol, nfrom);
		nfrom += nxrow;

		if (it.needGrad()) pQueue->regist(it, xgrad);
		else pQueue->registNoGrad(it);
	}
}

VTensor VTensor::subvector(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	int64 nfrom = x.getOpArg("0");
	int64 ncount = x.getOpArg("1");

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 ncol = shape[-1];
	int64 nrow = shape.total_size() / ncol;

	if (nfrom < 0 || nfrom + ncount > ncol) VP_THROW(VERR_OUT_OF_RANGE);

	VTensor y = tracer.createTensor(session(), shape.replace_end(ncount), x.type(), x.device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(subvector, device(), py, px, nrow, ncol, nfrom, ncount);

	return y;
}

void VTensor::subvector_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 nfrom = getMyArg("0");
		int64 ncount = getMyArg("1");

		VShape shape = x.shape();

		int64 ncol = shape[-1];
		int64 nrow = shape.total_size() / ncol;

		VTensor xgrad = tracer.createTensor(session(), shape, type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.isValid() ? ygrad.float_ptr() : NULL;

		CALL_MATH(subvector_backward, device(), pgx, pgy, nrow, ncol, nfrom, ncount);

		pQueue->regist(x, xgrad);
	}
	else {
		pQueue->registNoGrad(x);
	}
}

VTensor VTensor::pickup(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	if (shape1.size() != 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape2.size() < 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape1[0] != shape2[0]) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (x1.type() != VDataType::int32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::int32 && x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32 or int32", VDataTypeName(x2.type()));
	}
	if (x1.device() != x2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 nbat = shape1[0];
	int64 nrow = shape1[1];
	int64 nnom = shape2[1];
	int64 ncol = shape2.total_size() / (nbat * nnom);

	VShape yshape = shape2.replace_nth(1, shape1[1]);

	VTensor y = tracer.createTensor(session(), yshape, x2.type(), x1.device());

	int* px1 = x1.int_ptr();

	if (x2.type() == VDataType::int32) {
		int* py = y.int_ptr();
		int* px2 = x2.int_ptr();

		CALL_MATH(pickup_int, device(), py, px1, px2, nbat, nrow, nnom, ncol);
	}
	else {
		float* py = y.float_ptr();
		float* px2 = x2.float_ptr();

		CALL_MATH(pickup_float, device(), py, px1, px2, nbat, nrow, nnom, ncol);
	}

	return y;
}

void VTensor::pickup_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	if (x1.needGrad()) {
		VP_THROW(VERR_UNDEFINED);
	}
	else pQueue->registNoGrad(x1);

	if (x2.needGrad()) {
		if (x2.type() == VDataType::int32) {
			VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(x2.type()));
		}

		VShape shape1 = x1.shape();
		VShape shape2 = x2.shape();

		int64 nbat = shape1[0];
		int64 nrow = shape1[1];
		int64 nnom = shape2[1];
		int64 ncol = shape2.total_size() / (nbat * nnom);

		VTensor xgrad = tracer.createTensor(session(), shape2, VDataType::float32, x2.device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.isValid() ? ygrad.float_ptr() : NULL;
		int* px1 = x1.int_ptr();

		CALL_MATH(pickup_float_backward, device(), pgx, pgy, px1, nbat, nrow, nnom, ncol);

		pQueue->regist(x2, xgrad);
	}
	else pQueue->registNoGrad(x2);
}

VTensor VTensor::pickup_static(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	if (shape1.size() != 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (x1.type() != VDataType::int32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "int32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::int32 && x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32 or int32", VDataTypeName(x2.type()));
	}
	if (x1.device() != x2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 nbat = shape1[0];
	int64 nrow = shape1[1];
	int64 nnom = shape2[0];
	int64 ncol = shape2.total_size() / nnom;

	VShape yshape = shape2.replace_nth(0, nrow).insert_head(nbat);

	VTensor y = tracer.createTensor(session(), yshape, x2.type(), x1.device());

	int* px1 = x1.int_ptr();

	if (x2.type() == VDataType::int32) {
		int* py = y.int_ptr();
		int* px2 = x2.int_ptr();

		CALL_MATH(pickup_static_int, device(), py, px1, px2, nbat, nrow, nnom, ncol);
	}
	else {
		float* py = y.float_ptr();
		float* px2 = x2.float_ptr();

		CALL_MATH(pickup_static_float, device(), py, px1, px2, nbat, nrow, nnom, ncol);
	}

	return y;
}

VTensor VTensor::to_boxes(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	if (shape1 != shape2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	if (shape1[-1] != 2) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}

	if (x1.device() != x2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	int64 nrow = shape1.total_size() / 2;

	VTensor y = tracer.createTensor(session(), shape1.replace_end(4), x1.type(), x2.device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(to_boxes, device(), py, px1, px2, nrow);

	return y;
}

VTensor VTensor::complement_1(VExecTracer tracer) {
	VTensor x = *this;

	VShape xshape = x.shape();

	int64 ndat = xshape.total_size();

	VTensor y = tracer.createTensor(session(), xshape, type(), device());

	float* py = y.float_ptr();
	float* px = x.float_ptr();

	CALL_MATH(complement_1, device(), py, px, ndat);

	return y;
}

void VTensor::complement_1_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x = m_core->m_operands[0];

	if (x.needGrad()) {
		int64 ndat = x.shape().total_size();

		VTensor xgrad = tracer.createTensor(session(), x.shape(), type(), device());

		float* pgx = xgrad.float_ptr();
		float* pgy = ygrad.float_ptr();

		CALL_MATH(complement_1_backward, device(), pgx, pgy, ndat);

		pQueue->regist(x, xgrad);
	}
	else pQueue->registNoGrad(x);
}

VTensor VTensor::concat(VTensor second, VExecTracer tracer) {
	VTensor x1 = *this;
	VTensor x2 = second;

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 ncol1 = shape1[-1];
	int64 ncol2 = shape2[-1];
	int64 nrow = shape1.total_size() / ncol1;
	int64 nrest = 1;

	VShape yshape = shape1.replace_end(ncol1 + ncol2);

	if (shape1.size() == 4) {
		if (shape1.remove_nth(1) != shape2.remove_nth(1)) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

		ncol1 = shape1[1];
		ncol2 = shape2[1];
		nrow = shape1[0];
		nrest = shape1[2] * shape1[3];

		yshape = shape1.replace_nth(1, ncol1 + ncol2);
	}
	else {
		if (shape1.remove_end() != shape2.remove_end()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
	}

	if (x1.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x1.type()));
	}
	if (x2.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x2.type()));
	}

	if (x1.device() != x2.device()) VP_THROW(VERR_TENSOR_DEVICE);

	VTensor y = tracer.createTensor(session(), yshape, x1.type(), x2.device());

	float* py = y.float_ptr();
	float* px1 = x1.float_ptr();
	float* px2 = x2.float_ptr();

	CALL_MATH(concat, device(), py, px1, px2, nrow, ncol1, ncol2, nrest);

	return y;
}

void VTensor::concat_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer) {
	VTensor x1 = m_core->m_operands[0];
	VTensor x2 = m_core->m_operands[1];

	VShape shape1 = x1.shape();
	VShape shape2 = x2.shape();

	int64 ncol1 = shape1[-1];
	int64 ncol2 = shape2[-1];
	int64 nrow = shape1.total_size() / ncol1;
	int64 nrest = 1;

	if (shape1.size() == 4) {
		ncol1 = shape1[1];
		ncol2 = shape2[1];
		nrow = shape1[0];
		nrest = shape1[2] * shape1[3];
	}

	float* pgy = ygrad.float_ptr();

	if (x1.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), shape1, VDataType::float32, x1.device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(concat_backward_x1, device(), pgx, pgy, nrow, ncol1, ncol2, nrest);

		pQueue->regist(x1, xgrad);
	}
	else pQueue->registNoGrad(x1);

	if (x2.needGrad()) {
		VTensor xgrad = tracer.createTensor(session(), shape2, VDataType::float32, x2.device());

		float* pgx = xgrad.float_ptr();

		CALL_MATH(concat_backward_x2, device(), pgx, pgy, nrow, ncol1, ncol2, nrest);

		pQueue->regist(x2, xgrad);
	}
	else pQueue->registNoGrad(x2);
}

VTensor VTensor::to_filter(VExecTracer tracer) {
	VTensor x = *this;

	VShape shape = x.shape();

	if (x.type() != VDataType::float32) {
		VP_THROW2(VERR_UNMATCH_ON_TENSOR_DATATYPE, "float32", VDataTypeName(x.type()));
	}

	int64 nrow = shape[0];
	int64 ncol = shape.total_size() / nrow;

	VTensor true_count = tracer.createTensor(session(), { nrow }, VDataType::int32, x.device());
	VTensor max_true_count = tracer.createTensor(session(), { 1 }, VDataType::int32, x.device());

	float* px = x.float_ptr();
	int* pc = true_count.int_ptr();
	int* pm = max_true_count.int_ptr();

	CALL_MATH(count_true, device(), pc, px, nrow, ncol);
	CALL_MATH(select_max, device(), pm, pc, nrow);

	int64 max_size = (int64)max_true_count.getElement({}, tracer);

	VTensor y = tracer.createTensor(session(), { nrow, max_size }, VDataType::int32, x.device());

	int* py = y.int_ptr();

	CALL_MATH(to_filter, device(), py, px, nrow, ncol, max_size);

	return y;
}

int VTensor::incOperandRefCount() {
	return ++m_core->m_nOperandRefCount;
}

int VTensor::decOperandRefCount() {
	return --m_core->m_nOperandRefCount;
}

int VTensor::getOperandRefCount() {
	return m_core ? m_core->m_nOperandRefCount : -1;
}

void VTensor::keepBackpropMergeInfo(VModule ownModule, VTensor operand) {
	operand.incOperandRefCount();

	if (0 && VBackQueue::ms_bQueueTrace) {
		printf("[%s] T#%d <=", VGraphNode::GraphOpCodeName(VGraphOpCode::merge_parallel_result).c_str(), getNth());
		printf(" T#%d(%d)", operand.getNth(), operand.getOperandRefCount());
		printf("\n");
	}

	if (session().getNoGrad()) {
		setNeedGrad(false);
		//m_core->m_bNeedGrad = false;
	}
	else {
		setNeedGrad(true);
		//m_core->m_bNeedGrad = true;
		m_core->m_opCode = VGraphOpCode::merge_parallel_result;
		m_core->m_operands.push_back(operand);
		m_core->m_pOwningModule = ownModule.getCore();
	}
}

void VTensor::keepBackpropParamInfo(VGraphOpCode opCode, VTensor grad) {
	if (session().getNoGrad()) {
		setNeedGrad(false);
		//m_core->m_bNeedGrad = false;
	}
	else {
		// moving_stat 등 기울기 필요없는 파라미터를 강제로 기울기 제공 대상으로 변경하여 이를 차단함, 부작용 여부에 유의할 것
		//setNeedGrad(true);
		//m_core->m_bNeedGrad = true;
		m_core->m_opCode = opCode;
		if (m_core->m_operands.size() == 0) {
			m_core->m_operands.push_back(grad);
		}
	}
}

void VTensor::keepBackpropOperandsInfo(bool needGrad, VGraphOpCode opCode, VTensorList operands) {
	for (auto& it : operands) {
		it.incOperandRefCount();
	}

	if (0 && VBackQueue::ms_bQueueTrace) {
		printf("[%s] T#%d <=", VGraphNode::GraphOpCodeName(opCode).c_str(), getNth());
		for (auto& it : operands) {
			printf(" T#%d(%d)", it.getNth(), it.getOperandRefCount());
		}
		printf("\n");
	}

	if (session().getNoGrad()) {
		setNeedGrad(false);
		//m_core->m_bNeedGrad = false;
	}
	else {
		setNeedGrad(needGrad);
		//m_core->m_bNeedGrad = needGrad;
	}

	m_core->m_opCode = opCode;
	m_core->m_operands = operands; // 여기서 불필한 놈 삭제, 형상만 남기고 삭제, 내용 남김의 구분 처리 필요
				// 하지만 함부로 삭제하면 역전파 참조 계수에 문제 발생함
}

void VTensor::copyParam(VTensor tensor, string mode) {
	VShape hhShape = tensor.shape();
	VShape pmShape = shape();

	if (hhShape != pmShape) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);

	float* hp = tensor.float_ptr();
	float* pp = float_ptr();

	memcpy(pp, hp, sizeof(float) * pmShape.total_size());
}

/*
void VTensor::copyWeightParam(VTensor tensor, int ngate, string mode) {
	VShape hhShape = tensor.shape();
	VShape pmShape = shape();

	if (hhShape.size() != 2) VP_THROW(VERR_UNDEFINED);
	if (pmShape.size() != 2) VP_THROW(VERR_UNDEFINED);

	int64 ninp = pmShape[0];
	int64 nvec = pmShape[1];

	if (hhShape[1] != ninp) VP_THROW(VERR_UNDEFINED);
	if (hhShape[0] != nvec) VP_THROW(VERR_UNDEFINED);

	int64 nrec = nvec / ngate;

	float* hp = tensor.float_ptr();
	float* pp = float_ptr();

	int64 n = 0;

	for (int64 ng = 0; ng < ngate; ng++) {
		for (int64 nr = 0; nr < nrec; nr++) {
			for (int64 ni = 0; ni < ninp; ni++) {
				pp[(ni * nrec + nr) * ngate + ng] = hp[n++];
			}
		}
	}
}

void VTensor::copyBiasParam(VTensor tensor, int ngate, string mode) {
	VShape hhShape = tensor.shape();
	VShape pmShape = shape();

	if (hhShape.size() != 1) VP_THROW(VERR_UNDEFINED);
	if (pmShape.size() != 1) VP_THROW(VERR_UNDEFINED);

	int64 nrow = pmShape[0];
	int64 nsub = nrow / ngate;

	if (hhShape[0] != nrow) VP_THROW(VERR_UNDEFINED);

	float* hp = tensor.float_ptr();
	float* pp = float_ptr();

	int64 n = 0;

	for (int64 ng = 0; ng < ngate; ng++) {
		for (int64 ns = 0; ns < nsub; ns++) {
			pp[ns * ngate + ng] = hp[n++];
		}
	}
}
*/

//-----------------------------------------------------------------------------------------------------
// 코어 영역 확장 코드

VTensorCore::VTensorCore() : VObjCore(VObjType::Tensor) {
	// session도 갖지 않으며 연산 과정에서 보조정보를 저장할 때에만 사용한다.
}

void VTensorCore::m_onCreate() {
	m_bNeedGrad = false;
	m_opCode = VGraphOpCode::none;
	m_nOperandRefCount = 0;
}

void VTensorCore::m_onDelete() {
}

void VTensorCore::m_initParam(TensorInitMethod init_op, float mean, float init_arg, bool adaptive) {
	switch (init_op) {
	case TensorInitMethod::fzeros:
		m_data.memset(0);
		//if (m_nDevice < 0) memset(m_pDataCore->m_pData, 0, m_pDataCore->m_size);
		//else tensor_cuda.memset(m_pData, 0, m_size);
		break;
	case TensorInitMethod::fones:
		m_data.fill_float(1.0f);
		break;
	case TensorInitMethod::random_normal:
		m_data.init_random_normal(mean, init_arg, adaptive);
		break;
	case TensorInitMethod::random_uniform:
		m_data.init_random_uniform(mean, init_arg);
		break;
		/*
	case TensorInitMethod::arange:
		tensor = KTensorMath::arange(kwArgs["shape"]);
		break;
	case TensorInitMethod::fzeros_like:
		tensor = KTensorMath::allocHostFloatZeros(((KTensor)kwArgs["proto"]).shape());
		break;
		*/
	default:
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
}

void VTensorCore::m_copyDataFrom(VTensorCore* src, VExecTracer tracer) {
	if (src->m_data.isValid()) {
		void* py = m_data.void_ptr();
		void* px = src->m_data.void_ptr();

		int64 nbytes = m_data.byte_size();

		CALL_MATH(copy_data, m_data.device(), src->m_data.device(), py, px, nbytes);
	}
	else if (src->m_shape.size() > 0) {
		printf("src->shape: %s\n", src->m_shape.desc().c_str());
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
	}
}

void VTensorCore::m_resetCbBackSlot(int sid) {
	for (vector<VCbBackSlot>::iterator it = m_cbBackSlots.begin(); it != m_cbBackSlots.end(); it++) {
		if (it->getNth() == sid) {
			m_cbBackSlots.erase(it);
			break;
		}
	}
}

//-----------------------------------------------------------------------------------------------------
// 데이터 영역 확장 코드

