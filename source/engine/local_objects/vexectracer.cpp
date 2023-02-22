#include <cuda_runtime.h>

#include "../local_objects/vexectracer_core.h"
#include "../local_objects/vexectracer.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vfunction.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vcbitem.h"
#include "../local_objects/vudfitem.h"
#include "../support/vmath.h"
#include "../utils/vutils.h"

VExecTracer::VExecTracer() {
	m_core = NULL;
}

VExecTracer::VExecTracer(VSession session, string sBuiltin, VDict kwArgs) {
	m_core = new VExecTracerCore(session, sBuiltin, kwArgs);
}

VExecTracer::VExecTracer(const VExecTracer& src) {
	m_core = src.m_core->clone();
}

VExecTracer::VExecTracer(VExecTracerCore* core) {
	m_core = core->clone();
}

VExecTracer::~VExecTracer() {
	m_core->destroy();
}

VExecTracer& VExecTracer::operator =(const VExecTracer& src) {
	if (&src != this && m_core != src.m_core) {
		m_core->destroy();
		m_core = src.m_core->clone();
	}
	return *this;
}

VExecTracerCore* VExecTracer::getClone() {
	return (VExecTracerCore*)m_core->clone_core();
}

VExecTracerCore* VExecTracer::getCore() {
	return m_core;
}

void VExecTracer::destroyCore() {
	if (m_core->getRefCnt() > 1) m_core->destroy();
	else {
		m_core->destroy();
		m_core = NULL;
	}
}

VSession VExecTracer::session() const {
	return m_core->m_session;
}

bool VExecTracer::isValid() {
	return m_core != NULL;
}

int VExecTracer::getRefCnt() {
	return m_core->getRefCnt();
}

int VExecTracer::getNth() {
	return m_core->getNth();
}

VExecTracerCore::VExecTracerCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::ExecTracer) {
	m_sBuiltin = vutils.tolower(sBuiltin);
	m_propDict = kwArgs;
	m_session = session;
	m_setup();
}

void VExecTracer::whatToDo() {
	if (m_core == NULL || m_core->m_isBlocked()) return;
	VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

bool VExecTracer::hasValidHistory(VTensorDict xs) {
	if (m_core == NULL || m_core->m_isBlocked()) return false;
	if (m_core->m_status == ExecTracerStatus::recorded && m_core->m_isSameInput(xs)) return true;
	return false;
}

void VExecTracer::reset() {
	m_core->m_cleanHistory();
}

void VExecTracer::removeBranch() {
	m_core->destroy();
}

VTensor VExecTracer::createTensor(VSession session, VShape shape, VDataType type, int nDevice) {
	VTensor tensor(session, shape, type, nDevice);
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) m_core->m_addTensor(tensor);
	return tensor;
}

VTensor VExecTracer::createTensor(VSession session, VHTensor hTensor) {
	VTensor tensor(session, hTensor);
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) m_core->m_addTensor(tensor);
	return tensor;
}
 
VTensor VExecTracer::createTensor(VSession session, VTensorCore* core) {
	VTensor tensor(session, (VHTensor)core);
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) m_core->m_addTensor(tensor);
	return tensor;
}

VTensor VExecTracer::createTensor(VTensor src, VShape shape, TensorCloneInit init) {
	VTensor tensor(src, shape, init);
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) m_core->m_addTensor(tensor);
	return tensor;
}

void VExecTracer::addTensor(VTensor tensor) {
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) m_core->m_addTensor(tensor);
}

void VExecTracer::addInvokeForwardCallback(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict) {
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) {
		m_core->m_addInvokeForwardCallback(pCbFunc, pCbClose, instInfo, statusInfo, tensorDict);
	}
}

void VExecTracer::addInvokeBackwardCallback(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict) {
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) {
		m_core->m_addInvokeBackwardCallback(pCbFunc, pCbClose, instInfo, statusInfo, tensorDict, gradDict);
	}
}

void VExecTracer::addCallForwardUDF(VFunctionCore* functor, int nInst, VTensor y, VTensorList operands, VDict opArgs) {
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) {
		m_core->m_addCallForwardUDF(functor, nInst, y, operands, opArgs);
	}
}

void VExecTracer::addCallBackwardUDF(int nInst, VTensor y, VTensor ygrad, VTensor xgrad, int nth) {
	if (m_core != NULL && m_core->m_status == ExecTracerStatus::recording) {
		m_core->m_addCallBackwardUDF(nInst, y, ygrad, xgrad, nth);
	}
}

void VExecTracer::closeRecording(VTensorDict ys) {
	if (m_core == NULL || m_core->m_isBlocked()) return;
	if (m_core->m_status == ExecTracerStatus::empty) return;
	if (m_core->m_status != ExecTracerStatus::recording) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);

	m_core->m_ys = ys;
	m_core->m_status = ExecTracerStatus::recorded;
}

void VExecTracer::setInput(VTensorDict xs) {
	if (m_core == NULL || m_core->m_isBlocked()) return;

	m_core->m_xs = xs;
	m_core->m_status = ExecTracerStatus::recording;
}

void VExecTracer::addMathCall(VMathFunc func, VFuncArgList args) {
	if (m_core == NULL || m_core->m_status != ExecTracerStatus::recording) return;
	
	if (0) {
		printf("AP1: VExecTracer::addMathCall(%s) called\n", VExecTracerCore::ms_mathFuncNames[(int)func]);
	}

	ExecAction action;
	
	action.m_func = func;
	action.m_args = args;

	m_core->m_actionHistory.push_back(action);
}

void VExecTracer::dump(VList tids) {
	if (m_core == NULL || m_core->m_isBlocked()) return;

	for(auto & it: tids) {
		int tid = it;
		if (m_core->m_actingTensors.find(tid) == m_core->m_actingTensors.end()) continue;
		VTensor tensor = m_core->m_actingTensors[tid];
	}
}

static mutex mu_print;

void VExecTracer::dumpHistoryForDebug() {
	mu_print.lock();
	printf("TRACER#%d(%s): [", getNth(), m_core->m_sBuiltin.c_str());
	for (auto& it : m_core->m_actionHistory) {
		printf("%d ", (int)it.m_func);
	}
	printf("\n");
	mu_print.unlock();
}

VTensorDict VExecTracer::executeHistory() {
	if (m_core == NULL || m_core->m_status != ExecTracerStatus::recorded) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);

	m_core->m_executeHistory();

	return m_core->m_ys;
}

void VExecTracer::openBranch(int nDivisions) {
	if (m_core == NULL || m_core->m_status != ExecTracerStatus::recording) return;

	TracerMap branchMap;

	m_core->m_branchChidren.push_back(branchMap);
}

void VExecTracer::setVoidBranch(int nth) {
	if (m_core == NULL || m_core->m_status != ExecTracerStatus::recording) return;

	TracerMap branchMap = m_core->m_branchChidren.back();

	branchMap[nth] = VExecTracer();
}

VExecTracer VExecTracer::setValidBranch(int nth, string name, VTensorDict xs) {
	if (m_core == NULL || m_core->m_status != ExecTracerStatus::recording) return VExecTracer();

	TracerMap& branchMap = m_core->m_branchChidren.back();

	VExecTracer tracer(session(), name, {});
	tracer.setInput(xs);

	branchMap[nth] = tracer;

	return tracer;
}

void VExecTracer::setFork(int nDivisions) {
	if (m_core == NULL || m_core->m_status != ExecTracerStatus::recording) return;

	addMathCall(VMathFunc::__fork__, VFuncArgList { nDivisions });
}

//--------------------------------------------------------------------------------------------------------------

const char* VExecTracerCore::ms_mathFuncNames[] = {
	"__fork__",
	"__set_curr_device__",
	"__invoke_forward_callback__",
	"__invoke_backward_callback__",
	"__invoke_forward_user_defined_func__",
	"__invoke_backward_user_defined_func__",
	"__custom_open__",
	"__custom_arg__",
	"__custom_call__",
	"copy_data",
	"accumulate_grad",
	"subtract_param_grad",
	"apply_decay",
	"eval_adam_delta",
	"free",
	"fill_int",
	"fill_float",
	"set_zero",
	"memcpy_from_host",
	"memcpy_to_host",
	"memcpy_to_device",
	"init_random_normal",
	"init_random_uniform",
	"sub",
	"get_slice",
	"copy",
	"minus",
	"add",
	"add_residual",
	"add_residual_backward_b",
	"add_bias",
	"add_bias_backward_b",
	"add_2d_bias",
	"add_2d_bias_backward_b",
	"subtract",
	"subtract_bias",
	"subtract_backward_b",
	"subtract_bias_backward_b",
	"mult",
	"mult_backward_x1",
	"mult_backward_x2",
	"mult_se_mask",
	"mult_se_mask_backward_x1",
	"mult_se_mask_backward_x2",
	"mult_scalar",
	"add_mult_scalar_to",
	"sub_mult_scalar_to",
	"acc_sqsum",
	"acc_sqsum_decay",
	"adagrad_update",
	"div",
	"div_backward_x1",
	"div_backward_x2",
	"abs",
	"abs_backward",
	"square",
	"square_backward",
	"sqrt",
	"sqrt_backward",
	"exp",
	"exp_backward",
	"log",
	"log_backward",
	"maximum",
	"maximum_backward_x1",
	"maximum_backward_x2",
	"minimum",
	"minimum_backward_x1",
	"minimum_backward_x2",
	"_not",
	"_and",
	"_or",
	"greater_than_float_const",
	"greater_than_int_const",
	"greater_than",
	"less_than_const",
	"less_than",
	"less_than_cross",
	"equal_const",
	"equal",
	"greater_equal_const",
	"greater_equal",
	"less_equal_const",
	"less_equal",
	"matmul",
	"matmul_backward_x",
	"matmul_backward_w",
	"activate",
	"activate_backward",
	"activate_backward_with_y",
	"add_normal_noise",
	"add_uniform_noise",
	"gen_normal_random",
	"gen_uniform_random",
	"round",
	"codeconv",
	"cosinesim",
	"selectntop",
	"selectntoparg",
	"conv2d",
	"conv2d_backward_x",
	"conv2d_backward_k",
	"conv2d_transposed",
	"conv2d_transposed_backward_x",
	"conv2d_transposed_backward_k",
	"conv2d_dilated",
	"conv2d_dilated_backward_x",
	"conv2d_dilated_backward_k",
	"maxpool",
	"maxpool_backward_x",
	"avgpool",
	"avgpool_backward_x",
	"globalavg",
	"globalavg_backward_x",
	"adaptiveavg",
	"adaptiveavg_backward_x",
	"stride",
	"stride_backward_x",
	"lstm_process",
	"lstm_process_backward",
	"gru_process",
	"gru_process_backward",
	"batchnorm_norm",
	"batchnorm_scale",
	"batchnorm_backward_x",
	"batchnorm_backward_scale",
	"batchnorm_backward_shift",
	"batchnorm_backward_norm",
	"layernorm",
	"layernorm_backward",
	"dropout",
	"dropout_backward",
	"embed",
	"embed_backward_w",
	"mult_on_heads",
	"mult_on_heads_backward",
	"set_mh_attention_mask",
	"softmax_direct_on_axis",
	"softmax_direct_on_axis_backward",
	"mix_values",
	"mix_values_backward_prop",
	"mix_values_backward_value",
	"parallel_concat",
	"parallel_concat_backward_x1",
	"parallel_concat_backward_x2",
	"stack",
	"stack_backward",
	"concat",
	"concat_backward_x1",
	"concat_backward_x2",
	"undo_concat",
	"count_true",
	"select_max",
	"to_filter",
	"extract",
	"extract_backward",
	"subvector",
	"subvector_backward",
	"pickup_int",
	"pickup_float",
	"pickup_float_backward",
	"pickup_static_int",
	"pickup_static_float",
	"iou_cross_xywh",
	"iou_cross_lrtb",
	"iou_loss",
	"iou_loss_backward",
	"to_boxes",
	"complement_1",
	"complement_1_backward",
	"upsample",
	"upsample_backward",
	"sigmoid",
	"sigmoid_backward",
	"sigmoid_crossentropy",
	"sigmoid_crossentropy_backward_x",
	"sigmoid_crossentropy_with_logits",
	"sigmoid_crossentropy_with_logits_backward",
	"sigmoid_crossentropy_with_logits_idx",
	"sigmoid_crossentropy_with_logits_idx_backward",
	"softmax_idx_crossentropy",
	"softmax_i64_idx_crossentropy",
	"softmax_idx_crossentropy_backward_x",
	"softmax_i64_idx_crossentropy_backward_x",
	"softmax_idx_crossentropy_pos_idx",
	"softmax_idx_crossentropy_pos_idx_backward_x",
	"softmax_crossentropy",
	"softmax_crossentropy_backward_x",
	"transpose",
	"transpose_backward",
	"transpose_bin",
	"max",
	"min",
	"argmax",
	"argmin",
	"mean",
	"mean_backward",
	"sum_int",
	"sum",
	"sum_backward",
};

int VExecTracerCore::ms_nNextId = 1;

void VExecTracerCore::m_setup() {
	m_status = m_session.getNoTracer() ? ExecTracerStatus::blocked : ExecTracerStatus::empty;
	m_nRunForDebug = 0;
	m_nStepForDebug = 0;
	
	m_nId = ms_nNextId++;

	if (0) {
		printf("VExecTracerCore[%d, %s] is created\n", m_nId, m_sBuiltin.c_str());
	}

	size_t test1 = sizeof(ms_mathFuncNames);
	size_t test2 = sizeof(ms_mathFuncNames[0]);

	if ((int)VMathFunc::__end__ != (sizeof(ms_mathFuncNames) / sizeof(ms_mathFuncNames[0]))) VP_THROW(VERR_UNDEFINED);
}

bool VExecTracerCore::m_isBlocked() {
	return m_status == ExecTracerStatus::blocked;
}

bool VExecTracerCore::m_isSameInput(VTensorDict xs) {
	if (xs.size() != m_xs.size()) return false;

	for (auto& it : xs) {
		if (m_xs.find(it.first) == m_xs.end()) return false;
		if (it.second.getNth() != m_xs[it.first].getNth()) return false;
	}

	return true;
}

void VExecTracerCore::m_cleanHistory() {
	m_xs.clear();
	m_ys.clear();
	m_actingTensors.clear();
	m_actionHistory.clear();
	
	m_status = ExecTracerStatus::reset;
}

void VExecTracerCore::m_executeHistory() {
	m_nCurrExecBranchSet = 0;

	if (0) {
		printf("VExecTracerCore[%d, %s] is reused\n", m_nId, m_sBuiltin.c_str());
	}

	m_nStepForDebug = 0;

	int histLength = (int)m_actionHistory.size();

	for (auto& it : m_actionHistory) {
		if (0 && m_nId == 6) {
			mu_print.lock();
			int nDevice;
			cudaGetDevice(&nDevice);
			if (m_nStepForDebug == 32) {
				static float buf[10];
				printf("ygrad: ");
				float* pgy = (float*)((it.m_args)[2]);
				cudaMemcpy(buf, pgy, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				for (int k = 0; k < 10; k++) printf(" %f", buf[k]);
				printf("\n");
				printf("x: ");
				float* px = (float*)((it.m_args)[3]);
				cudaMemcpy(buf, px, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				for (int k = 0; k < 10; k++) printf(" %f", buf[k]);
				printf("\n");
			}
			printf("[TRACER#%d(%s:%d) %d/%d mathCall on history(%d, op:%s) called : nDevice = %d\n", getNth(), m_sBuiltin.c_str(), m_nRunForDebug, m_nStepForDebug, histLength, m_nId, ms_mathFuncNames[(int)it.m_func], nDevice);
			mu_print.unlock();
		}

		m_mathCall(it.m_func, it.m_args);
		
		if (0 && m_nId == 6) {
			mu_print.lock();
			int nDevice;
			cudaGetDevice(&nDevice);
			if (m_nStepForDebug == 32) {
				static float buf[10];
				printf("bgrad: ");
				float* pgx = (float*)((it.m_args)[1]);
				cudaMemcpy(buf, pgx, sizeof(float) * 10, cudaMemcpyDeviceToHost);
				for (int k = 0; k < 10; k++) printf(" %f", buf[k]);
				printf("\n");
			}
			mu_print.unlock();
		}

		m_nStepForDebug++;
	}

	if (1) m_nRunForDebug++;
}

void VExecTracerCore::m_custom_call(int64 arg0, int64 arg1, int64 arg2, int64 arg3, int64 arg4) {
	//VDict result;

	VCbCustomModuleExec* cbFunc = (VCbCustomModuleExec*)arg0;
	VCbFreeReportBuffer* cbFree = (VCbFreeReportBuffer*)arg1;

	void* pInst = (void*)arg2;
	void* pAux = (void*)arg3;

	const VExBuf* pResultBuf;
	//VHTensor hResult = 0;

	VDictWrapper wrapper(m_callbackArgs);

	VObjCore* pModule = (VObjCore*)arg4;

	int res_code = cbFunc(pInst, pAux, time(NULL), (int64)pModule, wrapper.detach(), &pResultBuf);
	if (res_code != 0) VP_THROW(VERR_INVALID_CUSTOM_MODULE_CALLBACK);

	/*
	VDict ydict = VDictWrapper::unwrap(pResultBuf);
	ys = vutils.toTensorDict((VHSession)session(), ydict);

	for (auto& it : ys) {
		tracer.addTensor(it.second);
	}
	*/

	res_code = cbFree(pInst, pAux, pResultBuf);
	if (res_code != 0) VP_THROW(VERR_INVALID_FREE_REPORT_BUFFER);

	m_callbackArgs.clear();
}

void VExecTracerCore::m_mathCall(VMathFunc func, VFuncArgList a) {
	switch(func) {
	case VMathFunc::__fork__:
		m_execBranches(a[0]);
		break;
	case VMathFunc::__set_curr_device__:
		cudaSetDevice((int)a[0]);
		break;
	case VMathFunc::__invoke_forward_callback__:
		m_invokeForwardCallback((int)a[0]);
		break;
	case VMathFunc::__invoke_backward_callback__:
		m_invokeBackwardCallback((int)a[0]);
		break;
	case VMathFunc::__invoke_forward_user_defined_func__:
		m_invokeForwardUDFCall((int)a[0], (int)a[1]);
		break;
	case VMathFunc::__invoke_backward_user_defined_func__:
		m_invokeBackwardUDFCall((int)a[0], (int)a[1], (int)a[2]);
		break;
	case VMathFunc::__custom_open__:
		m_callbackArgs.clear();
		break;
	case VMathFunc::__custom_arg__:
		m_callbackArgs[(string)a[0]] = (VHTensor)(int64)a[1];
		break;
	case VMathFunc::__custom_call__:
		m_custom_call(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::copy_data:
		VMath::copy_data(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::accumulate_grad:
		VMath::accumulate_grad(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::subtract_param_grad:
		VMath::subtract_param_grad(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::apply_decay:
		VMath::apply_decay(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::eval_adam_delta:
		VMath::eval_adam_delta(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::fill_int:
		VMath::fill_int(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::fill_float:
		VMath::fill_float(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::set_zero:
		VMath::set_zero(a[0], a[1], a[2]);
		break;
	case VMathFunc::memcpy_from_host:
		VMath::memcpy_from_host(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::memcpy_to_host:
		VMath::memcpy_to_host(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::memcpy_to_device:
		VMath::memcpy_to_device(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::init_random_normal:
		VMath::init_random_normal(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::init_random_uniform:
		VMath::init_random_uniform(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::sub:
		VMath::sub(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::get_slice:
		VMath::get_slice(a[0], a[1], a[2], a[3]);
		break;
	//case VMathFunc::copy_slice_from:
	//	VMath::copy_slice_from(a[0], a[1], a[2], a[3]);
	//	break;
	case VMathFunc::copy:
		VMath::copy(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::minus:
		VMath::minus(a[0], a[1], a[2]);
		break;
	case VMathFunc::add:
		VMath::add(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::add_residual:
		VMath::add_residual(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::add_residual_backward_b:
		VMath::add_residual_backward_b(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::add_bias:
		VMath::add_bias(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::add_bias_backward_b:
		VMath::add_bias_backward_b(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::add_2d_bias:
		VMath::add_2d_bias(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::add_2d_bias_backward_b:
		VMath::add_2d_bias_backward_b(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::subtract:
		VMath::subtract(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::subtract_bias:
		VMath::subtract_bias(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::subtract_backward_b:
		VMath::subtract_backward_b(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::subtract_bias_backward_b:
		VMath::subtract_bias_backward_b(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::mult:
		VMath::mult(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::mult_backward_x1:
		VMath::mult_backward_x1(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::mult_backward_x2:
		VMath::mult_backward_x2(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::mult_se_mask:
		VMath::mult_se_mask(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::mult_se_mask_backward_x1:
		VMath::mult_se_mask_backward_x1(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::mult_se_mask_backward_x2:
		VMath::mult_se_mask_backward_x2(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::mult_scalar:
		VMath::mult_scalar(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::add_mult_scalar_to:
		VMath::add_mult_scalar_to(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::sub_mult_scalar_to:
		VMath::sub_mult_scalar_to(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::acc_sqsum:
		VMath::acc_sqsum(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::acc_sqsum_decay:
		VMath::acc_sqsum_decay(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::adagrad_update:
		VMath::adagrad_update(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::div:
		VMath::div(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::div_backward_x1:
		VMath::div_backward_x1(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::div_backward_x2:
		VMath::div_backward_x2(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::abs:
		VMath::square(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::abs_backward:
		VMath::abs_backward(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::square:
		VMath::square(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::square_backward:
		VMath::square_backward(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::sqrt:
		VMath::sqrt(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::sqrt_backward:
		VMath::sqrt_backward(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::exp:
		VMath::exp(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::exp_backward:
		VMath::exp_backward(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::log:
		VMath::log(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::log_backward:
		VMath::log_backward(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::maximum:
		VMath::maximum(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::maximum_backward_x1:
		VMath::maximum_backward_x1(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::maximum_backward_x2:
		VMath::maximum_backward_x2(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::minimum:
		VMath::minimum(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::minimum_backward_x1:
		VMath::minimum_backward_x1(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::minimum_backward_x2:
		VMath::minimum_backward_x2(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::_not:
		VMath::_not(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::_and:
		VMath::_and(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::_or:
		VMath::_or(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::greater_than_float_const:
		VMath::greater_than_float_const(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::greater_than_int_const:
		VMath::greater_than_int_const(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::greater_than:
		VMath::greater_than(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::less_than_const:
		VMath::less_than_const(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::less_than:
		VMath::less_than(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::less_than_cross:
		VMath::less_than_cross(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::equal_const:
		VMath::equal_const(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::equal:
		VMath::equal(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::greater_equal_const:
		VMath::greater_equal_const(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::greater_equal:
		VMath::greater_equal(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::less_equal_const:
		VMath::less_equal_const(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::less_equal:
		VMath::less_equal(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::matmul:
		VMath::matmul(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::matmul_backward_x:
		VMath::matmul_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::matmul_backward_w:
		VMath::matmul_backward_w(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::activate:
		VMath::activate(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::activate_backward:
		VMath::activate_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::activate_backward_with_y:
		VMath::activate_backward_with_y(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::add_normal_noise:
		VMath::add_normal_noise(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::add_uniform_noise:
		VMath::add_uniform_noise(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::gen_normal_random:
		VMath::gen_normal_random(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::gen_uniform_random:
		VMath::gen_uniform_random(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::round:
		VMath::round(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::codeconv:
		VMath::codeconv(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::cosinesim:
		VMath::cosinesim(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::selectntop:
		VMath::selectntop(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::selectntoparg:
		VMath::selectntoparg(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::conv2d:
		VMath::conv2d(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
		break;
	case VMathFunc::conv2d_backward_x:
		VMath::conv2d_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
		break;
	case VMathFunc::conv2d_backward_k:
		VMath::conv2d_backward_k(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
		break;
	case VMathFunc::conv2d_transposed:
		VMath::conv2d_transposed(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case VMathFunc::conv2d_transposed_backward_x:
		VMath::conv2d_transposed_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case VMathFunc::conv2d_transposed_backward_k:
		VMath::conv2d_transposed_backward_k(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case VMathFunc::conv2d_dilated:
		VMath::conv2d_dilated(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case VMathFunc::conv2d_dilated_backward_x:
		VMath::conv2d_dilated_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case VMathFunc::conv2d_dilated_backward_k:
		VMath::conv2d_dilated_backward_k(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case VMathFunc::maxpool:
		VMath::maxpool(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::maxpool_backward_x:
		VMath::maxpool_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::avgpool:
		VMath::avgpool(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::avgpool_backward_x:
		VMath::avgpool_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::globalavg:
		VMath::globalavg(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::globalavg_backward_x:
		VMath::globalavg_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::adaptiveavg:
		VMath::adaptiveavg(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::adaptiveavg_backward_x:
		VMath::adaptiveavg_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::stride:
		VMath::stride(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]);
		break;
	case VMathFunc::stride_backward_x:
		VMath::stride_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]);
		break;
	case VMathFunc::lstm_process:
		VMath::lstm_process(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::lstm_process_backward:
		VMath::lstm_process_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::gru_process:
		VMath::gru_process(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::gru_process_backward:
		VMath::gru_process_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10]);
		break;
	case VMathFunc::batchnorm_norm:
		VMath::batchnorm_norm(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12]);
		break;
	case VMathFunc::batchnorm_scale:
		VMath::batchnorm_scale(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::batchnorm_backward_x:
		VMath::batchnorm_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::batchnorm_backward_scale:
		VMath::batchnorm_backward_scale(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::batchnorm_backward_shift:
		VMath::batchnorm_backward_shift(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::batchnorm_backward_norm:
		VMath::batchnorm_backward_norm(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::layernorm:
		VMath::layernorm(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::layernorm_backward:
		VMath::layernorm_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::dropout:
		VMath::dropout(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::dropout_backward:
		VMath::dropout_backward(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::embed:
		VMath::embed(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::embed_backward_w:
		VMath::embed_backward_w(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::mult_on_heads:
		VMath::mult_on_heads(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
		break;
	case VMathFunc::mult_on_heads_backward:
		VMath::mult_on_heads_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::set_mh_attention_mask:
		VMath::set_mh_attention_mask(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::softmax_direct_on_axis:
		VMath::softmax_direct_on_axis(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::softmax_direct_on_axis_backward:
		VMath::softmax_direct_on_axis_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::mix_values:
		VMath::mix_values(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::mix_values_backward_prop:
		VMath::mix_values_backward_prop(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::mix_values_backward_value:
		VMath::mix_values_backward_value(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::parallel_concat:
		VMath::parallel_concat(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::parallel_concat_backward_x1:
		VMath::parallel_concat_backward_x1(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::parallel_concat_backward_x2:
		VMath::parallel_concat_backward_x2(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::stack:
		VMath::stack(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::stack_backward:
		VMath::stack_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::concat:
		VMath::concat(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::concat_backward_x1:
		VMath::concat_backward_x1(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::concat_backward_x2:
		VMath::concat_backward_x2(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::undo_concat:
		VMath::undo_concat(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::count_true:
		VMath::count_true(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::select_max:
		VMath::select_max(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::to_filter:
		VMath::to_filter(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::extract:
		VMath::extract(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::extract_backward:
		VMath::extract_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::subvector:
		VMath::subvector(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::subvector_backward:
		VMath::subvector_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::pickup_int:
		VMath::pickup_int(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::pickup_float:
		VMath::pickup_float(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::pickup_float_backward:
		VMath::pickup_float_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::pickup_static_int:
		VMath::pickup_static_int(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::pickup_static_float:
		VMath::pickup_static_float(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
		break;
	case VMathFunc::iou_cross_xywh:
		VMath::iou_cross_xywh(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::iou_cross_lrtb:
		VMath::iou_cross_lrtb(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::iou_loss:
		VMath::iou_loss(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::iou_loss_backward:
		VMath::iou_loss_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::to_boxes:
		VMath::to_boxes(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::complement_1:
		VMath::complement_1(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::complement_1_backward:
		VMath::complement_1_backward(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::upsample:
		VMath::upsample(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::upsample_backward:
		VMath::upsample_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
		break;
	case VMathFunc::sigmoid:
		VMath::sigmoid(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::sigmoid_backward:
		VMath::sigmoid_backward(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::sigmoid_crossentropy:
		VMath::sigmoid_crossentropy(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::sigmoid_crossentropy_backward_x:
		VMath::sigmoid_crossentropy_backward_x(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::sigmoid_crossentropy_with_logits:
		VMath::sigmoid_crossentropy_with_logits(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::sigmoid_crossentropy_with_logits_backward:
		VMath::sigmoid_crossentropy_with_logits_backward(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::sigmoid_crossentropy_with_logits_idx:
		VMath::sigmoid_crossentropy_with_logits_idx(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::sigmoid_crossentropy_with_logits_idx_backward:
		VMath::sigmoid_crossentropy_with_logits_idx_backward(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::softmax_idx_crossentropy:
		VMath::softmax_idx_crossentropy(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::softmax_i64_idx_crossentropy:
		VMath::softmax_i64_idx_crossentropy(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::softmax_idx_crossentropy_backward_x:
		VMath::softmax_idx_crossentropy_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::softmax_i64_idx_crossentropy_backward_x:
		VMath::softmax_i64_idx_crossentropy_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::softmax_idx_crossentropy_pos_idx:
		VMath::softmax_idx_crossentropy_pos_idx(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::softmax_idx_crossentropy_pos_idx_backward_x:
		VMath::softmax_idx_crossentropy_pos_idx_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::softmax_crossentropy:
		VMath::softmax_crossentropy(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::softmax_crossentropy_backward_x:
		VMath::softmax_crossentropy_backward_x(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
		break;
	case VMathFunc::transpose:
		VMath::transpose(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::transpose_backward:
		VMath::transpose_backward(a[0], a[1], a[2], a[3], a[4], a[5]);
		break;
	case VMathFunc::transpose_bin:
		VMath::transpose_bin(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::max:
		VMath::max(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::min:
		VMath::min(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::argmax:
		VMath::argmax(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::argmin:
		VMath::argmin(a[0], a[1], a[2], a[3], a[4]);
		break;
	case VMathFunc::mean:
		VMath::mean(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::mean_backward:
		VMath::mean_backward(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::sum_int:
		VMath::sum_int(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::sum:
		VMath::sum(a[0], a[1], a[2], a[3]);
		break;
	case VMathFunc::sum_backward:
		VMath::sum_backward(a[0], a[1], a[2], a[3]);
		break;
	default:
		VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		break;
	}
}

void VExecTracerCore::m_addTensor(VTensor tensor) {
	if (this == NULL || m_isBlocked()) return;
	m_actingTensors[tensor.getNth()] = tensor;
}

void VExecTracerCore::m_addInvokeForwardCallback(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict) {
	VCbItem cbItem(m_session, pCbFunc, pCbClose, instInfo, statusInfo, tensorDict, VDict());

	int id = cbItem.getNth();

	m_callbackSettings[id] = cbItem;

	// 대부분의 텐서가 tracer에 이미 등록될 것으로 보이지만 혹시라도 놓칠 가능성에 대비해 아래의 전수조사를 통해 빠진 항목 등록
	for (auto& it : tensorDict) {
		VTensorDict tensors = vutils.toTensorDict(m_session, it.second);

		for (auto& it : tensors) {
			m_addTensor(it.second);
		}
	}

	ExecAction action;

	action.m_func = VMathFunc::__invoke_forward_callback__;
	action.m_args = { id };

	m_actionHistory.push_back(action);
}

void VExecTracerCore::m_addInvokeBackwardCallback(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict) {
	VCbItem cbItem(m_session, pCbFunc, pCbClose, instInfo, statusInfo, tensorDict, gradDict);

	int id = cbItem.getNth();

	m_callbackSettings[id] = cbItem;

	ExecAction action;

	action.m_func = VMathFunc::__invoke_backward_callback__;
	action.m_args = { id };

	m_actionHistory.push_back(action);
}

void VExecTracerCore::m_invokeForwardCallback(int id) {
	VCbItem cbItem = m_callbackSettings[id];

	extern VDict V_invokeModuleForwardCallback(VHSession hSession, VCbForwardModule * pCbFunc, VCbClose * pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict);

	VCbForwardModule* pCbFunc = (VCbForwardModule*)cbItem.getCbFunc();
	VCbClose* pCbClose = (VCbClose*)cbItem.getCbClose();
	VDict tensorDict = cbItem.getTensorDict();

	// 콜백 반값 result는 아직 특별한 용도가 없지만 차후 확장에 대비해 전달받을 수 있게 한다.
	VDict result = V_invokeModuleForwardCallback(m_session, pCbFunc, pCbClose, cbItem.getInstInfo(), cbItem.getStatusInfo(), tensorDict);

	// 아래 줄은 메모리 이중 반납 문제로 문제를 일으킬 소지가 있음, 유사 사례 발견됨, 문제 발생 확인시 삭제
	vutils.freeDictInternal(tensorDict);
}

void VExecTracerCore::m_invokeBackwardCallback(int id) {
	VCbItem cbItem = m_callbackSettings[id];

	extern VDict V_invokeModuleBackwardCallback(VHSession hSession, VCbBackwardModule * pCbFunc, VCbClose * pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict);

	VCbBackwardModule* pCbFunc = (VCbBackwardModule*)cbItem.getCbFunc();
	VCbClose* pCbClose = (VCbClose*)cbItem.getCbClose();
	VDict tensorDict = cbItem.getTensorDict();
	VDict gradDict = cbItem.getGradDict();

	// 콜백 반값 result는 아직 특별한 용도가 없지만 차후 확장에 대비해 전달받을 수 있게 한다.
	VDict result = V_invokeModuleBackwardCallback(m_session, pCbFunc, pCbClose, cbItem.getInstInfo(), cbItem.getStatusInfo(), tensorDict, gradDict);

	// 아래 두 줄은 메모리 이중 반납 문제로 문제를 일으킬 소지가 있음, 유사 사례 발견됨, 문제 발생 확인시 삭제
	vutils.freeDictInternal(tensorDict);
	vutils.freeDictInternal(gradDict);
}

void VExecTracerCore::m_addCallForwardUDF(VFunctionCore* functor, int nInst, VTensor y, VTensorList operands, VDict opArgs) {
	VUDFItem udfItem(m_session, functor, y, operands, opArgs);

	int id = y.getNth();

	//operands[0].setOpArg("__udf_item__", (VObjCore*)udfItem.getClone());
	operands[0].setOpArg("__udf_item__", (int64)(VObjCore*)udfItem.getCore());

	//m_udfSettings[id] = udfItem.getClone();	// ref#를 1 이상으로 유지시켜 operands에 getClone() 대신 getCore()를 주면서도 인스턴스를 유지시킴
	m_udfSettings[id] = udfItem;	// ref#를 1 이상으로 유지시켜 operands에 getClone() 대신 getCore()를 주면서도 인스턴스를 유지시킴

	// 대부분의 텐서가 tracer에 이미 등록될 것으로 보이지만 혹시라도 놓칠 가능성에 대비해 아래의 전수조사를 통해 빠진 항목 등록
	for (auto& it : operands) {
		m_addTensor(it);
	}

	m_addTensor(y);

	ExecAction action;

	action.m_func = VMathFunc::__invoke_forward_user_defined_func__;
	action.m_args = { id, nInst };

	m_actionHistory.push_back(action);

	udfItem.dump("add forward");
}

void VExecTracerCore::m_addCallBackwardUDF(int nInst, VTensor y, VTensor ygrad, VTensor xgrad, int nth) {
	VUDFItemCore* pUdfItem = (VUDFItemCore*)(int64)y.getMyArg("__udf_item__");
	VUDFItem udfItem(pUdfItem);

	int id = y.getNth();

	udfItem.setGrad(nth, ygrad, xgrad);

	m_udfSettings[id] = udfItem;

	m_addTensor(ygrad);
	m_addTensor(xgrad);

	ExecAction action;

	action.m_func = VMathFunc::__invoke_backward_user_defined_func__;
	action.m_args = { id, nth, nInst };

	m_actionHistory.push_back(action);

	udfItem.dump("add backward");
}

void VExecTracerCore::m_invokeForwardUDFCall(int id, int nInst) {
	VUDFItem udfItem = m_udfSettings[id];

	VFunctionCore* functor = udfItem.getFunctor();
	VTensorList operands = udfItem.getXs();
	VDict opArgs = udfItem.getOpArgs();
	VTensor y = udfItem.getY();

	udfItem.dump("invoke forward before");

	VFunction function(functor);

	VTensor result = function.forward(nInst, operands, opArgs);

	if (result.getNth() != y.getNth()) {
		if (result.shape() != y.shape()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		void* py = y.void_ptr();
		void* pr = result.void_ptr();
		VMath::copy_data(y.device(), result.device(), py, pr, y.byte_size());
	}

	udfItem.dump("invoke forward after");
}

void VExecTracerCore::m_invokeBackwardUDFCall(int id, int nth, int nInst) {
	VUDFItem udfItem = m_udfSettings[id];

	VFunctionCore* functor = udfItem.getFunctor();
	VTensorList operands = udfItem.getXs();
	VDict opArgs = udfItem.getOpArgs();
	VTensor y = udfItem.getY();
	VTensor ygrad = udfItem.getYGrad();
	VTensorList xgrads = udfItem.getXGrads();

	VFunction function(functor);

	VTensor result = function.backward(nInst, ygrad, nth, operands, opArgs);
	VTensor xgrad = xgrads[nth];

	if (xgrad.getNth() != result.getNth()) {
		if (xgrad.shape() != result.shape()) VP_THROW1(VERR_BAD_SHAPE_TENSOR, __func__);
		void* prg = result.void_ptr();
		void* pxg = xgrad.void_ptr();
		VMath::copy_data(xgrad.device(), result.device(), pxg, prg, xgrad.byte_size());
	}

	udfItem.dump("invoke backward");
}

VTensor VExecTracerCore::m_lookupTensor(VFuncArg arg) {
	void* ptr = arg;

	for (auto& it : m_actingTensors) {
		if (!it.second.isValid()) continue;
		if (it.second.hasNoData()) continue;
		if (it.second.void_ptr() == ptr) {
			return it.second;
		}
	}

	return VTensor();
}

void VExecTracerCore::m_execBranches(int nDivisions) {
	if (m_branchChidren.size() <= m_nCurrExecBranchSet) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
	
	TracerMap children = m_branchChidren[m_nCurrExecBranchSet++];

	std::thread** ppThreads = new std::thread * [nDivisions];

	for (int n = 0; n < nDivisions; n++) {
		VExecTracer child = children[n];

		if (child.isValid()) {
			ppThreads[n] = new std::thread(ms_execBranchMain, child.getClone());
		}
		else {
			ppThreads[n] = NULL;
		}
	}

	for (int n = 0; n < nDivisions; n++) {
		if (ppThreads[n] == NULL) continue;

		ppThreads[n]->join();
		delete ppThreads[n];

		VExecTracer child = children[n];
		child.destroyCore();
	}

	delete[] ppThreads;
}

void VExecTracerCore::ms_execBranchMain(void* aux) {
	VExecTracer tracer((VExecTracerCore*)(VObjCore*)aux);
	tracer.executeHistory();
}
