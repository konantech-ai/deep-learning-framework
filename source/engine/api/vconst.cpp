#pragma once

#include "../api/vconst.h"

// 주의: side 정보를 요하는 레이어는 반드시 VModule::m_evaluate() 15행의 검사에 등록되어야 함
//      현재 concat, cosinesim 레이어가 이에 해당

string VConsts::getLayerExpression(string sLayerName) {
    if (sLayerName == "flatten")		return "flatten(#x)";
	if (sLayerName == "linear")			return "add(matmul(#x, pm:w), pm:b)";
	if (sLayerName == "dense")			return "activate(add(matmul(#x, pm:w), pm:b), int:actfunc, float:leaky_alpha)";
	if (sLayerName == "addbias")		return "add(#x, pm:b)";
	if (sLayerName == "activate")		return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "normal_noise")	return "normal_noise(#x, float:mean, float:std)";
	if (sLayerName == "uniform_noise")	return "uniform_noise(#x, float:min, float:max)";
	if (sLayerName == "normal_random")	return "normal_random(#x, shape:shape, float:mean, float:std)";
	if (sLayerName == "uniform_random")	return "uniform_random(#x, shape:shape, float:min, float:max)";
	if (sLayerName == "round")			return "round(#x, int:prec)";
	if (sLayerName == "codeconv")		return "codeconv(#x)";
	if (sLayerName == "cosinesim")		return "cosinesim(#x, side:with)";
	if (sLayerName == "selectntop")		return "selectntop(#x, int:ntop)";
	if (sLayerName == "selectntoparg")	return "selectntoparg(#x, int:ntop)";
	if (sLayerName == "layernorm")		return "layernorm(#x, int:axis, float:scale)";
	if (sLayerName == "batchnorm")		return "batchnorm(#x, pm:mavg, pm:mvar, pm:rescale, pm:shift, float:momentum, float:epsilon)";
	if (sLayerName == "max")			return "stride(maxpool(#x, shape:kernel, shape:padding, int:padding_mode), shape:stride)";
	if (sLayerName == "avg")			return "stride(avgpool(#x, shape:kernel, shape:padding, int:padding_mode), shape:stride)";
	if (sLayerName == "upsample")		return "upsample(#x, shape:stride)";
	if (sLayerName == "dropout")		return "dropout(#x, float:drop_ratio)";
	if (sLayerName == "globalavg")		return "globalavg(#x)";
	if (sLayerName == "adaptiveavg")	return "adaptiveavg(#x, shape:size)";
	if (sLayerName == "reshape")		return "reshape(#x, shape:shape)";
	if (sLayerName == "transpose")		return "transpose(#x, list:axes)";
	//if (sLayerName == "rnn")			return "rnn(#x, pm:wi, pm:wr, pm:bi, pm:br, int:rec_size, bool:in_seq, bool:out_seq, bool:use_bias, bool:batch_first, int:actfunc, float:leaky_alpha)";
	//if (sLayerName == "lstm")			return "lstm(#x, pm:wi, pm:wr, pm:bi, pm:br, int:rec_size, bool:in_seq, bool:out_seq, bool:use_bias, bool:use_state, bool:batch_first)";
	//if (sLayerName == "gru")			return "gru(#x, pm:wi, pm:wr, pm:bi, pm:br, int:rec_size, bool:in_seq, bool:out_seq, bool:use_bias, bool:batch_first)";
	if (sLayerName == "rnn")			return "rnn(#x, pmset:rnn, int:rec_size, bool:in_seq, bool:out_seq, int:num_layers, bool:bidirectional, bool:use_bias, bool:batch_first, int:actfunc, float:leaky_alpha, float:drop_ratio)";
	if (sLayerName == "lstm")			return "lstm(#x, pmset:rnn, int:rec_size, bool:in_seq, bool:out_seq, int:num_layers, bool:bidirectional, bool:use_bias, bool:use_state, bool:batch_first, float:drop_ratio)";
	if (sLayerName == "gru")			return "gru(#x, pmset:rnn, int:rec_size, bool:in_seq, bool:out_seq, int:num_layers, bool:bidirectional, bool:use_bias, bool:batch_first, float:drop_ratio)";
	if (sLayerName == "embed")			return "embed(#x, pm:w, bool:position, int:ndim)";
	if (sLayerName == "mh_attention")	return "mh_attention(#x, side:key, side:query, side:value, pm:Kw, pm:Kb, pm:Qw, pm:Qb, pm:Vw, pm:Vb, pm:Ow, pm:Ob, int:head_cnt, bool:mask, float:coef)";
	if (sLayerName == "extract")		return "extract(#x, int:axis, int:index, int:count, bool:reduce_axis)";
	if (sLayerName == "concat")			return "concat(#x, side:branch)";
	if (sLayerName == "pass")			return "pass(#x, str:direction, str:dump, bool:exception)";
	if (sLayerName == "residual_net")	return "add_residual(#x, #residual)";
	if (sLayerName == "parallel_net")	return "parallel_all";
	if (sLayerName == "add_net")		return "add_all";
	if (sLayerName == "stack_net")		return "stack_all";
	if (sLayerName == "se_net")			return "se_mult(#residual, #x)";
	if (sLayerName == "leaky")			return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "relu")			return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "gelu")			return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "selu")			return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "tanh")			return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "sigmoid")		return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "mish")			return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "swish")			return "activate(#x, int:actfunc, float:leaky_alpha)";
	if (sLayerName == "softmax")		return "activate(#x, int:actfunc, float:leaky_alpha)";

	// conv2d 관련 레이어 내부에서 지원하려던 batchnorm 기능을 지원하지 않기로 함
	if (sLayerName == "conv2d")
		return "add_2d_bias(activate(stride(conv2d(#x, pm:w, shape:padding, int:padding_mode), shape:stride), int:actfunc, float:leaky_alpha), pm:b)";
	if (sLayerName == "conv2d_transposed")		
		return "activate(add_2d_bias(conv2d_transposed(#x, pm:w, shape:stride, shape:padding, int:padding_mode), pm:b), int:actfunc, float:leaky_alpha)";
	if (sLayerName == "conv2d_dilated")	// 패딩 정보 필요한지 검토 필요
		return "activate(add_2d_bias(stride(conv2d_dilated(#x, pm:w, shape:gap), shape:stride), pm:b), int:actfunc, float:leaky_alpha)";

    VP_THROW(VERR_CONDITIONAL_STATEMENT);
}

string VConsts::getLossExpression(string sLossName) {
	if (sLossName == "crossentropy")				return "crossentropy(#logit, #label)";
	if (sLossName == "binary_crossentropy")   		return "crossentropy_sigmoid(#logit, #label)";
	if (sLossName == "crossentropy_sigmoid")		return "crossentropy_sigmoid(#logit, #label)";
	if (sLossName == "crossentropy_pos_idx")     return "crossentropy_pos_idx(#logit, #label)";
	if (sLossName == "mse")							return "square(subtract(#estimate, #answer))";

    VP_THROW(VERR_CONDITIONAL_STATEMENT);
}

string VConsts::getMetricExpression(string sInfName) {
	if (sInfName == "sigmoid")						return "sigmoid(#logit)";
	if (sInfName == "softmax")						return "softmax(#logit)";

    VP_THROW(VERR_CONDITIONAL_STATEMENT);
}

VGraphOpCode VConsts::convToOpCode(string opcode) {
	if (opcode == "pm")				return VGraphOpCode::pm;
	if (opcode == "pmset")			return VGraphOpCode::pmset;
	if (opcode == "shape")			return VGraphOpCode::shape;
	if (opcode == "list")			return VGraphOpCode::list;
	if (opcode == "int")			return VGraphOpCode::_int;
	if (opcode == "bool")			return VGraphOpCode::_bool;
	if (opcode == "float")			return VGraphOpCode::_float;
	if (opcode == "str")			return VGraphOpCode::str;
	if (opcode == "side")			return VGraphOpCode::side;

	if (opcode == "flatten")		return VGraphOpCode::flatten;
	if (opcode == "reshape")		return VGraphOpCode::reshape;
	if (opcode == "transpose")		return VGraphOpCode::transpose;
	if (opcode == "add")			return VGraphOpCode::add;
	if (opcode == "add_2d_bias")	return VGraphOpCode::add_2d_bias;
	if (opcode == "add_residual")	return VGraphOpCode::add_residual;
	if (opcode == "subtract")		return VGraphOpCode::subtract;
	if (opcode == "abs")			return VGraphOpCode::abs;
	if (opcode == "square")			return VGraphOpCode::square;
	if (opcode == "matmul")			return VGraphOpCode::matmul;
	if (opcode == "activate")		return VGraphOpCode::activate;
	if (opcode == "normal_noise")	return VGraphOpCode::normal_noise;
	if (opcode == "uniform_noise")	return VGraphOpCode::uniform_noise;
	if (opcode == "normal_random")	return VGraphOpCode::normal_random;
	if (opcode == "uniform_random")	return VGraphOpCode::uniform_random;
	if (opcode == "round")			return VGraphOpCode::round;
	if (opcode == "codeconv")		return VGraphOpCode::codeconv;
	if (opcode == "cosinesim")		return VGraphOpCode::cosinesim;
	if (opcode == "selectntop")		return VGraphOpCode::selectntop;
	if (opcode == "selectntoparg")	return VGraphOpCode::selectntoparg;
	if (opcode == "conv2d")			return VGraphOpCode::conv2d;
	if (opcode == "conv2d_transposed")	return VGraphOpCode::conv2d_transposed;
	if (opcode == "conv2d_dilated")		return VGraphOpCode::conv2d_dilated;
	if (opcode == "rnn")			return VGraphOpCode::rnn;
	if (opcode == "lstm")			return VGraphOpCode::lstm;
	if (opcode == "gru")			return VGraphOpCode::gru;
	if (opcode == "maxpool")		return VGraphOpCode::maxpool;
	if (opcode == "avgpool")		return VGraphOpCode::avgpool;
	if (opcode == "max")			return VGraphOpCode::max;
	if (opcode == "argmax")			return VGraphOpCode::argmax;
	if (opcode == "maximum")		return VGraphOpCode::maximum;
	if (opcode == "stride")			return VGraphOpCode::stride;
	if (opcode == "upsample")		return VGraphOpCode::upsample;
	if (opcode == "globalavg")		return VGraphOpCode::globalavg;
	if (opcode == "adaptiveavg")	return VGraphOpCode::adaptiveavg;
	if (opcode == "mean")			return VGraphOpCode::mean;
	if (opcode == "sum")			return VGraphOpCode::sum;
	if (opcode == "embed")			return VGraphOpCode::embed;
	if (opcode == "batchnorm")		return VGraphOpCode::batchnorm;
	if (opcode == "layernorm")		return VGraphOpCode::layernorm;
	if (opcode == "mh_attention")	return VGraphOpCode::mh_attention;
	if (opcode == "dropout")		return VGraphOpCode::dropout;
	if (opcode == "extract")		return VGraphOpCode::extract;
	if (opcode == "concat")			return VGraphOpCode::concat;
	if (opcode == "parallel_all")	return VGraphOpCode::parallel_all;
	if (opcode == "add_all")		return VGraphOpCode::add_all;
	if (opcode == "stack_all")		return VGraphOpCode::stack_all;
	if (opcode == "crossentropy")   return VGraphOpCode::crossentropy;
	if (opcode == "crossentropy_sigmoid")	return VGraphOpCode::crossentropy_sigmoid;
	if (opcode == "crossentropy_pos_idx")	return VGraphOpCode::crossentropy_pos_idx;

	if (opcode == "mult")			return VGraphOpCode::mult;
	if (opcode == "se_mult")		return VGraphOpCode::se_mult;
	if (opcode == "div")			return VGraphOpCode::div;
	if (opcode == "exp")			return VGraphOpCode::exp;
	if (opcode == "log")			return VGraphOpCode::log;
	if (opcode == "subvector")		return VGraphOpCode::subvector;
	if (opcode == "and")			return VGraphOpCode::_and;
	if (opcode == "or")				return VGraphOpCode::_or;
	if (opcode == "not")			return VGraphOpCode::_not;
	if (opcode == "eq")				return VGraphOpCode::_eq;
	if (opcode == "ne")				return VGraphOpCode::_ne;
	if (opcode == "gt")				return VGraphOpCode::_gt;
	if (opcode == "lt")				return VGraphOpCode::_lt;
	if (opcode == "ge")				return VGraphOpCode::_ge;
	if (opcode == "le")				return VGraphOpCode::_le;
	if (opcode == "gt_cross")		return VGraphOpCode::gt_cross;
	if (opcode == "lt_cross")		return VGraphOpCode::lt_cross;
	if (opcode == "ge_cross")		return VGraphOpCode::ge_cross;
	if (opcode == "le_cross")		return VGraphOpCode::le_cross;
	if (opcode == "pickup")			return VGraphOpCode::pickup;
	if (opcode == "pickup_static")	return VGraphOpCode::pickup_static;
	if (opcode == "to_filter")		return VGraphOpCode::to_filter;
	if (opcode == "to_int")			return VGraphOpCode::to_int;
	if (opcode == "to_float")		return VGraphOpCode::to_float;
	if (opcode == "to_tensor")		return VGraphOpCode::to_tensor;
	if (opcode == "get_column")		return VGraphOpCode::get_column;
	if (opcode == "iou_cross_xywh")	return VGraphOpCode::iou_cross_xywh;
	if (opcode == "iou_cross_lrtb")	return VGraphOpCode::iou_cross_lrtb;
	if (opcode == "ciou_loss")		return VGraphOpCode::ciou_loss;
	if (opcode == "hstack")			return VGraphOpCode::hstack;
	if (opcode == "sigmoid")		return VGraphOpCode::sigmoid;
	if (opcode == "pass")			return VGraphOpCode::pass;
	if (opcode == "to_boxes")		return VGraphOpCode::to_boxes;
	if (opcode == "complement_1")	return VGraphOpCode::complement_1;
	if (opcode == "select_best_with_idx")					return VGraphOpCode::select_best_with_idx;
	if (opcode == "sigmoid_crossentropy_with_logits")		return VGraphOpCode::sigmoid_crossentropy_with_logits;
	if (opcode == "sigmoid_crossentropy_with_logits_idx")	return VGraphOpCode::sigmoid_crossentropy_with_logits_idx;

	if (opcode == "")				return VGraphOpCode::none;

	printf("BP1: opcode = %s\n", opcode.c_str());
	VP_THROW(VERR_CONDITIONAL_STATEMENT);
};

OptAlgorithm VConsts::getOptAlgorithm(string sBuiltin) {
	sBuiltin = vutils.tolower(sBuiltin);

	if (sBuiltin == "sgd")				return OptAlgorithm::sgd;
	if (sBuiltin == "adam")				return OptAlgorithm::adam;
	if (sBuiltin == "momentum")			return OptAlgorithm::momentum;
	if (sBuiltin == "nesterov")			return OptAlgorithm::nesterov;
	if (sBuiltin == "adagrad")			return OptAlgorithm::adagrad;
	if (sBuiltin == "rmsprop")			return OptAlgorithm::rmsprop;

	VP_THROW(VERR_CONDITIONAL_STATEMENT);
}

ActFunc VConsts::getActFunc(string sActFunc) {
	sActFunc = vutils.tolower(sActFunc);

	if (sActFunc == "none")				return ActFunc::none;
	if (sActFunc == "relu")				return ActFunc::relu;
	if (sActFunc == "gelu")				return ActFunc::gelu;
	if (sActFunc == "selu")				return ActFunc::selu;
	if (sActFunc == "tanh")				return ActFunc::tanh;
	if (sActFunc == "sigmoid")			return ActFunc::sigmoid;
	if (sActFunc == "mish")				return ActFunc::mish;
	if (sActFunc == "swish")			return ActFunc::swish;
	if (sActFunc == "leaky")			return ActFunc::leaky;
	if (sActFunc == "softmax")			return ActFunc::softmax;

    VP_THROW(VERR_CONDITIONAL_STATEMENT);
}

bool VConsts::needXForBackwardgetActFunc(ActFunc func) {
	switch (func) {
	case ActFunc::relu:
	case ActFunc::selu:
	case ActFunc::tanh:
	case ActFunc::sigmoid:
	case ActFunc::leaky:
		return false;
	case ActFunc::gelu:
	case ActFunc::mish:
	case ActFunc::swish:
	case ActFunc::softmax:
		return true;
	default:
		VP_THROW(VERR_CONDITIONAL_STATEMENT);
	}
}
