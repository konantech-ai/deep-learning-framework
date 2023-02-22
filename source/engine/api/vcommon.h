#pragma once

#include <string>

#define VP_THROW(x)						{ throw VException(x, __FILE__, __LINE__); }
#define VP_THROW1(x,arg1)				{ throw VException(x, arg1, __FILE__, __LINE__); }
#define VP_THROW2(x,arg1,arg2)			{ throw VException(x, arg1, arg2, __FILE__, __LINE__); }
#define VP_THROW3(x,arg1,arg2, arg3)	{ throw VException(x, arg1, arg2, arg3, __FILE__, __LINE__); }

#define TP_THROW(x) VP_THROW(x)

#define VP_MEMO(x)

#include "../include/vapi.h"
#include "../include/vvalue.h"

#include "../utils/vexception.h"
#include "../utils/vutils.h"

#define MAX(x,y) ((x>y)?(x):(y))
#define MIN(x,y) ((x<y)?(x):(y))

#define SESSION_OPEN()   \
		try { \
			VSession session(hSession); \
			try { \
				try {

#define SESSION_CLOSE()  \
					return VERR_OK; \
				} \
				catch (ValException ex) { VP_THROW(ex.m_nErrCode); } \
			} \
			catch (VException ex) { session.SetLastError(ex); return ex.GetErrorCode(); } \
			catch (...) { return VERR_UNKNOWN; } \
		} \
		catch (...) { return VERR_MACRO_SESSION; }


#define HANDLE_OPEN(cls, hObject, capsule) \
		cls capsule(session, hObject)

#define HANDLE_OPEN_OR_NULL(cls, hObject, capsule) \
		if (hObject == NULL) return VERR_OK; \
		cls capsule(session, hObject)

#define POINTER_CHECK(ptr) \
		if (ptr == NULL) return VERR_INVALID_POINTER;

enum class VModuleType { layer, network, /* model, */ macro, custom, user_defined };
enum class TensorInitMethod { arange, fzeros, fzeros_like, fones, random_uniform, random_normal };
enum class OptAlgorithm { sgd, adam, momentum, nesterov, adagrad, rmsprop };
enum class PaddingMode { zeros, reflect, replicate, circular };

enum class VGraphOpCode {
	none,
	input,
	term,
	pm,
	pmset,
	shape,
	list,
	_int,
	_float,
	_bool,
	_user_defined_function,
	str,
	side,
	flatten,
	activate,
	normal_noise,
	uniform_noise,
	normal_random,
	uniform_random,
	round,
	codeconv,
	cosinesim, 
	selectntop,
	selectntoparg,
	maxpool,
	avgpool,
	globalavg,
	adaptiveavg,
	stride,
	upsample,
	pass,
	max,
	min,
	argmax,
	argmin,
	maximum,
	minimum,
	matmul,
	conv2d,
	conv2d_transposed,
	conv2d_dilated,
	add,
	add_2d_bias,
	add_residual,
	subtract,
	abs,
	square,
	reshape,
	transpose,
	embed,
	rnn,
	lstm,
	gru,
	batchnorm,
	layernorm,
	mh_attention,
	dropout,
	extract,
	concat,
	crossentropy,
	crossentropy_sigmoid,
	crossentropy_pos_idx,
	mean,
	sum,
	merge_parallel_result,
	parallel_all,
	add_all,
	stack_all,
	parallel_concat,
	mult,
	se_mult,
	div,
	exp,
	log,
	_and,
	_or,
	_not,
	_eq,
	_ne,
	_gt,
	_lt,
	_ge,
	_le,
	gt_cross,
	lt_cross,
	ge_cross,
	le_cross,
	subvector,
	pickup,
	pickup_static,
	to_filter,
	to_int,
	to_float,
	to_tensor,
	get_column,
	hstack,
	select_best_with_idx,
	iou_cross_xywh,
	iou_cross_lrtb,
	ciou_loss,
	diou_loss,
	giou_loss,
	iou_loss,
	to_boxes,
	complement_1,
	sigmoid,
	sigmoid_crossentropy_with_logits,
	sigmoid_crossentropy_with_logits_idx,
	__end__
};

typedef vector<VTensor> VTensorList;
typedef map<int, VTensor> VTensorMap;
typedef map<string, VTensor> VTensorDict;

typedef vector<VModule> VModuleList;
typedef map<string, VModule> VModuleDict;

struct VFuncCbHandlerInfo {
	bool isValid;
	VCbForwardFunction* m_pFuncCbForward;
	VCbBackwardFunction* m_pFuncCbBackward;
	VCbClose* m_pFuncCbClose;
	void* m_pFuncCbAux;
};

//#define YOLO_DEBUG_TEMPORAL