#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VCbBackSlotCore;

#define CALL_MATH(fname, ...) { \
	tracer.addMathCall(VMathFunc::fname, { __VA_ARGS__ }); \
	VMath::fname(__VA_ARGS__); }

enum class VMathFunc {
	__fork__,
	__set_curr_device__,
	__invoke_forward_callback__,
	__invoke_backward_callback__,
	__invoke_forward_user_defined_func__,
	__invoke_backward_user_defined_func__,
	__custom_open__,
	__custom_arg__,
	__custom_call__,
	copy_data,
	accumulate_grad,
	subtract_param_grad,
	apply_decay,
	eval_adam_delta,
	free,
	fill_int,
	fill_float,
	set_zero,
	memcpy_from_host,
	memcpy_to_host,
	memcpy_to_device,
	init_random_normal,
	init_random_uniform,
	sub,
	get_slice,
	copy,
	minus,
	add,
	add_residual,
	add_residual_backward_b,
	add_bias,
	add_bias_backward_b,
	add_2d_bias,
	add_2d_bias_backward_b,
	subtract,
	subtract_bias,
	subtract_backward_b,
	subtract_bias_backward_b,
	mult,
	mult_backward_x1,
	mult_backward_x2,
	mult_se_mask,
	mult_se_mask_backward_x1,
	mult_se_mask_backward_x2,
	mult_scalar,
	add_mult_scalar_to,
	sub_mult_scalar_to,
	acc_sqsum,
	acc_sqsum_decay,
	adagrad_update,
	div,
	div_backward_x1,
	div_backward_x2,
	abs,
	abs_backward,
	square,
	square_backward,
	sqrt,
	sqrt_backward,
	exp,
	exp_backward,
	log,
	log_backward,
	maximum,
	maximum_backward_x1,
	maximum_backward_x2,
	minimum,
	minimum_backward_x1,
	minimum_backward_x2,
	_not,
	_and,
	_or,
	equal_const,
	equal,
	greater_than_float_const,
	greater_than_int_const,
	greater_than,
	less_than_const,
	less_than,
	less_than_cross,
	greater_equal_const,
	greater_equal,
	less_equal_const,
	less_equal,
	matmul,
	matmul_backward_x,
	matmul_backward_w,
	activate,
	activate_backward,
	activate_backward_with_y,
	add_normal_noise,
	add_uniform_noise,
	gen_normal_random,
	gen_uniform_random,
	round,
	codeconv,
	cosinesim,
	selectntop,
	selectntoparg,
	conv2d,
	conv2d_backward_x,
	conv2d_backward_k,
	conv2d_transposed,
	conv2d_transposed_backward_x,
	conv2d_transposed_backward_k,
	conv2d_dilated,
	conv2d_dilated_backward_x,
	conv2d_dilated_backward_k,
	maxpool,
	maxpool_backward_x,
	avgpool,
	avgpool_backward_x,
	globalavg,
	globalavg_backward_x,
	adaptiveavg,
	adaptiveavg_backward_x,
	stride,
	stride_backward_x,
	lstm_process,
	lstm_process_backward,
	gru_process,
	gru_process_backward,
	batchnorm_norm,
	batchnorm_scale,
	batchnorm_backward_x,
	batchnorm_backward_scale,
	batchnorm_backward_shift,
	batchnorm_backward_norm,
	layernorm,
	layernorm_backward,
	dropout,
	dropout_backward,
	embed,
	embed_backward_w,
	mult_on_heads,
	mult_on_heads_backward,
	set_mh_attention_mask,
	softmax_direct_on_axis,
	softmax_direct_on_axis_backward,
	mix_values,
	mix_values_backward_prop,
	mix_values_backward_value,
	parallel_concat,
	parallel_concat_backward_x1,
	parallel_concat_backward_x2,
	stack,
	stack_backward,
	concat,
	concat_backward_x1,
	concat_backward_x2,
	undo_concat,
	count_true,
	select_max,
	to_filter,
	extract,
	extract_backward,
	subvector,
	subvector_backward,
	pickup_int,
	pickup_float,
	pickup_float_backward,
	pickup_static_int,
	pickup_static_float,
	iou_cross_xywh,
	iou_cross_lrtb,
	iou_loss,
	iou_loss_backward,
	to_boxes,
	complement_1,
	complement_1_backward,
	upsample,
	upsample_backward,
	sigmoid,
	sigmoid_backward,
	sigmoid_crossentropy,
	sigmoid_crossentropy_backward_x,
	sigmoid_crossentropy_with_logits,
	sigmoid_crossentropy_with_logits_backward,
	sigmoid_crossentropy_with_logits_idx,
	sigmoid_crossentropy_with_logits_idx_backward,
	softmax_idx_crossentropy,
	softmax_i64_idx_crossentropy,
	softmax_idx_crossentropy_backward_x,
	softmax_i64_idx_crossentropy_backward_x,
	softmax_idx_crossentropy_pos_idx,
	softmax_idx_crossentropy_pos_idx_backward_x,
	softmax_crossentropy,
	softmax_crossentropy_backward_x,
	transpose,
	transpose_backward,
	transpose_bin,
	max,
	min,
	argmax,
	argmin,
	mean,
	mean_backward,
	sum_int,
	sum,
	sum_backward,
	__end__
};

class VCbItem;
class VUDFItem;

enum class ExecTracerStatus { blocked, empty, reset, recording, recorded, playing };
enum class VFuncArgType { fptr, iptr, i64ptr, ptr, ucptr, f32, i64, i32, flag, str };

class VFuncArg {
public:
	VFuncArg(float* arg) { m_value.m_pfloat = arg; type = VFuncArgType::fptr; }
	VFuncArg(int* arg) { m_value.m_pint32 = arg; type = VFuncArgType::iptr; }
	VFuncArg(int64* arg) { m_value.m_pint64 = arg; type = VFuncArgType::i64ptr; }
	VFuncArg(void* arg) { m_value.m_pvoid = arg; type = VFuncArgType::ptr; }
	VFuncArg(unsigned char* arg) { m_value.m_puchar = arg; type = VFuncArgType::ucptr; }
	VFuncArg(float arg) { m_value.m_float = arg; type = VFuncArgType::f32; }
	VFuncArg(int64 arg) { m_value.m_int64 = arg; type = VFuncArgType::i64; }
	VFuncArg(int arg) { m_value.m_int32 = arg; type = VFuncArgType::i32; }
	VFuncArg(bool arg) { m_value.m_bool = arg; type = VFuncArgType::flag; }
	VFuncArg(string arg) { m_strValue = arg; type = VFuncArgType::str; }

	operator float* () const { if (type != VFuncArgType::fptr) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_pfloat; }
	operator int* () const { if (type != VFuncArgType::iptr) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_pint32; }
	operator int64* () const { if (type != VFuncArgType::i64ptr) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_pint64; }
	operator unsigned char* () const { if (type != VFuncArgType::ucptr) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_puchar; }
	operator float () const { if (type != VFuncArgType::f32) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_float; }
	operator int64() const { if (type != VFuncArgType::i64) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_int64; }
	operator int() const { if (type != VFuncArgType::i32) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_int32; }
	operator bool() const { if (type != VFuncArgType::flag) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_value.m_bool; }
	operator string() const { if (type != VFuncArgType::str) VP_THROW(VERR_FUNCTION_ARGUMENT_TYPE);  return m_strValue; }
	operator void* () const {
		if (type == VFuncArgType::fptr) return (void*)m_value.m_pfloat;
		else if (type == VFuncArgType::iptr) return (void*)m_value.m_pint32;
		else if (type == VFuncArgType::ptr) return (void*)m_value.m_pvoid;
		else {
			VP_THROW(VERR_CONDITIONAL_STATEMENT);
		}
	}

	bool is_data_pointer() { return (type == VFuncArgType::fptr) || (type == VFuncArgType::iptr); }
	void* get_data_pointer() {
		if (type == VFuncArgType::fptr) return (void*)m_value.m_pfloat;
		else if (type == VFuncArgType::iptr) return (void*)m_value.m_pint32;
		else {
			VP_THROW(VERR_NOT_IMPLEMENTED_YET);
		}
	}

protected:
	//VValueType m_type;
	union value_union {
		int* m_pint32;
		int64* m_pint64;
		float* m_pfloat;
		void* m_pvoid;
		unsigned char* m_puchar;
		int m_int32;
		int64 m_int64;
		float m_float;
		bool m_bool;
	} m_value;
	string m_strValue;
	VFuncArgType type;
};

typedef vector<VFuncArg> VFuncArgList;

struct ExecAction {
	VMathFunc m_func;
	VFuncArgList m_args;
};

typedef vector<ExecAction> ExecActionList;

class VExecTracer;
class VFunctionCore;

typedef map<int, VExecTracer> TracerMap;
typedef vector<TracerMap> TracerMapList;

class VExecTracerCore : public VObjCore {
protected:
	friend class VExecTracer;
protected:
	VExecTracerCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
	VExecTracerCore* clone() { return (VExecTracerCore*)clone_core(); }
	VSession session() { return m_session; }
	void m_setup();
protected:
	VSession m_session;
	string m_sBuiltin;
	VDict m_propDict;

protected:
	ExecTracerStatus m_status;

	int m_nId;

	static int ms_nNextId;

	VTensorDict m_xs;
	VTensorDict m_ys;

	//VTensorList m_actingTensors; // 재활용 텐서의 자리보전을 확실하게 하기 위해  ref cnt 증가시켜 두는 것이 목적임
	VTensorMap m_actingTensors; // VTensorList만으로도 본연의 목적에는 충분하지만 디버깅 과정에서의 쉬운 내용 확인을 위해 임시로 list 대신 map을 사용함

	ExecActionList m_actionHistory;

	TracerMapList m_branchChidren;

	int m_nCurrExecBranchSet;

	VDict m_callbackArgs;
	
	int m_nRunForDebug;
	int m_nStepForDebug;

	map<int, VCbItem> m_callbackSettings;
	map<int, VUDFItem> m_udfSettings;

protected:
	bool m_isBlocked();
	bool m_isSameInput(VTensorDict xs);
	void m_cleanHistory();
	void m_addTensor(VTensor tensor);
	void m_executeHistory();
	void m_mathCall(VMathFunc func, VFuncArgList args);
	void m_execBranches(int nDivision);
	void m_custom_call(int64 arg0, int64 arg1, int64 arg2, int64 arg3, int64 arg4);
	
	void m_addInvokeForwardCallback(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict);
	void m_addInvokeBackwardCallback(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict);
	
	void m_invokeForwardCallback(int id);
	void m_invokeBackwardCallback(int id);

	void m_addCallForwardUDF(VFunctionCore* functor, int nInst, VTensor y, VTensorList operands, VDict opArgs);
	void m_addCallBackwardUDF(int nInst, VTensor y, VTensor ygrad, VTensor xgrad, int nth);

	void m_invokeForwardUDFCall(int id, int nInst);
	void m_invokeBackwardUDFCall(int id, int nth, int nInst);

	VTensor m_lookupTensor(VFuncArg arg);

	static void ms_execBranchMain(void* aux);
	static const char* ms_mathFuncNames[];

};
