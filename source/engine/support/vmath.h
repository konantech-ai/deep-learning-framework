#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../api/vdefine.h"
#include "../local_objects/vhypermanager.h"

class VMath {
public:
	static void DumpUsage();
	static void cudaCheck(cudaError_t cuda_ret, string name, string file, int line);

public:
	static void seed_random(int64 random_seed);

	static void* mem_alloc(int device, int64 size);
	static void mem_free(int device, void* ptr);

	static void memcpy_host_to_host(void* dptr, void* sptr, int64 size);
	static void memcpy_host_to_device(void* dptr, void* sptr, int64 size);
	static void memcpy_device_to_host(void* dptr, void* sptr, int64 size);
	static void memcpy_device_to_device(void* dptr, void* sptr, int64 size);

public: // 최적화 지원 연산
	static void copy_data(int dest_device, int src_device, void* py, void* px, int64 nbytes);
	static void accumulate_grad(int device, float* py, float* px, int64 nrow);
	static void subtract_param_grad(int device, float* pp, float* pg, int64 nrow, HYPER hLearningRate);
	static void apply_decay(int device, float* pd, float* pp, float* pg, int64 nrow, HYPER hL1Decay, HYPER hL2Decay);
	static void eval_adam_delta(int device, float* pa, float* pg, float* ps, float* pt, float* pn, int64 nrow, HYPER hRo1, HYPER hRo2, HYPER hEpsilon);
	static void mult_scalar(int device, float* py, int64 ndat, HYPER hCoef);	// y *= c
	static void add_mult_scalar_to(int device, float* py, float* pa, float* px, int64 ndat, HYPER hCoef); // y = a + x * c
	static void sub_mult_scalar_to(int device, float* py, float* pa, float* px, int64 ndat, HYPER hCoef); // y = a - x * c
	static void acc_sqsum(int device, float* pr, float* pg, int64 ndat);
	static void acc_sqsum_decay(int device, float* pr, float* pg, int64 ndat, HYPER hDecay);
	static void adagrad_update(int device, float* pn, float* pg, float* pr, int64 ndat, HYPER hLearningRate, HYPER hSigma);

public: //기본 연산
	static void fill_int(int device, int* ptr, int64 size, int value);
	static void fill_float(int device, float* ptr, int64 size, float value);
	static void set_zero(int device, void* ptr, int64 size);

	static void memcpy_from_host(int device, void* py, void* px, int64 size);
	static void memcpy_to_host(int device, void* py, void* px, int64 size);
	static void memcpy_to_device(int device, void* py, void* px, int64 size);

	static void init_random_normal(int device, float* ptr, int64 ndat, float mean, float init_arg, bool adaptive);
	static void init_random_uniform(int device, float* ptr, int64 ndat, float mean, float init_arg);

	static void sub(int device, float* ptr, int64 ndat, float val);

	static float get_sum(int device, float* ptr, int64 ndat);

public: // 병렬 처리 지원 연산
	static void get_slice(int device, void* py, void* px, int64 nbytes);
	//static void copy_slice_from(int device, void* py, void* px, int64 nbytes);

public: // 산술 연산
	static void copy(int device, float* py, float* pa, int64 ndat);
	static void minus(int device, float* py, int64 ndat);
	static void add(int device, float* py, float* pa, float* pb, int64 ndat);
	static void add_residual(int device, float* py, float* pa, float* pb, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest);
	static void add_residual_backward_b(int device, float* pgb, float* pgy, int64 ndat, int64 nchn1, int64 nchn2, int64 nrest, bool acc);
	static void add_bias(int device, float* py, float* pa, float* pb, int64 nrow, int64 ncol);
	static void add_bias_backward_b(int device, float* pgb, float* pgy, int64 nrow, int64 ncol, bool acc);
	static void add_2d_bias(int device, float* py, float* pa, float* pb, int64 ndat, int64 xchn, int64 xh, int64 xw);
	static void add_2d_bias_backward_b(int device, float* pgb, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, bool acc);
	static void subtract(int device, float* py, float* pa, float* pb, int64 ndat);
	static void subtract_bias(int device, float* py, float* pa, float* pb, int64 ndat, int64 nrow, int64 ncol);
	static void subtract_backward_b(int device, float* pgb, float* pgy, int64 ndat);
	static void subtract_bias_backward_b(int device, float* pgb, float* pgy, int64 ndat, int64 nrow, int64 ncol);
	static void mult(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void mult_backward_x1(int device, float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void mult_backward_x2(int device, float* pgx, float* pgy, float* px1, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void mult_se_mask(int device, float* py, float* px1, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth);
	static void mult_se_mask_backward_x1(int device, float* pgx, float* pgy, float* px2, int64 ndat, int64 nchn, int64 nheight, int64 nwidth);
	static void mult_se_mask_backward_x2(int device, float* pgx, float* pgy, float* px1, int64 ndat, int64 nchn, int64 nheight, int64 nwidth);
	static void div(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void div_backward_x1(int device, float* pgx, float* pgy, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void div_backward_x2(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);

public: // 특수 함수 연산
	static void abs(int device, float* py, float* px, int64 ndat);
	static void abs_backward(int device, float* pgx, float* pgy, float* px, int64 ndat);

	static void square(int device, float* py, float* px, int64 ndat);
	static void square_backward(int device, float* pgx, float* pgy, float* px, int64 ndat);

	static void sqrt(int device, float* py, float* px, int64 ndat);
	static void sqrt_backward(int device, float* pgx, float* pgy, float* px, int64 ndat);

	static void exp(int device, float* py, float* px, int64 ndat);
	static void exp_backward(int device, float* pgx, float* pgy, float* px, int64 ndat);

	static void log(int device, float* py, float* px, int64 ndat);
	static void log_backward(int device, float* pgx, float* pgy, float* px, int64 ndat);

	static void complement_1(int device, float* py, float* px, int64 ndat);
	static void complement_1_backward(int device, float* pgx, float* pgy, int64 ndat);

public: // 선택 연산
	static void maximum(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void maximum_backward_x1(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void maximum_backward_x2(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);

	static void minimum(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void minimum_backward_x1(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void minimum_backward_x2(int device, float* pgx, float* pgy, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);

public: // 논리 연산
	static void _not(int device, float* py, float* px, int64 ndat);
	static void _and(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void _or(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);

public: // 비교 연산
	static void equal(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void equal_const(int device, float* py, float* px, float val, int64 ndat);
	static void greater_than_float_const(int device, float* py, float* px, float val, int64 ndat);
	static void greater_than_int_const(int device, float* py, int* px, int val, int64 ndat);
	static void greater_than(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void less_than(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void less_than_const(int device, float* py, float* px, float val, int64 ndat);
	static void less_than_cross(int device, float* py, float* px1, float* px2, int64 nrow, int64 ncol);
	static void greater_equal(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void greater_equal_const(int device, float* py, float* px, float val, int64 ndat);
	static void less_equal(int device, float* py, float* px1, float* px2, int64 left1, int64 left2, int64 mid, int64 right1, int64 right2);
	static void less_equal_const(int device, float* py, float* px, float val, int64 ndat);

public: // 다층 퍼셉트론 신경망 연산
	/*
	static void matmul(int device, float* py, float* px, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc);
	static void matmul_backward_x(int device, float* pgx, float* pgy, float* pw, int64 ndat, int64 xvec, int64 yvec, bool acc);
	static void matmul_backward_w(int device, float* pgw, float* pgy, float* px, int64 ndat, int64 xvec, int64 yvec, bool acc);
	*/

	static void matmul(int device, float* py, float* pw, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc);
	static void matmul_backward_x(int device, float* pgx, float* pgy, float* pw, int64 yvec, int64 ndat, int64 xvec, bool acc);
	static void matmul_backward_w(int device, float* pgw, float* pgy, float* px, int64 yvec, int64 ndat, int64 xvec, bool acc);

	static void activate(int device, float* py, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha);
	static void activate_backward(int device, float* pgx, float* pgy, float* px, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha);
	static void activate_backward_with_y(int device, float* pgx, float* pgy, float* py, int64 nrow, int64 ncol, int actFunc, HYPER hLeakyAlpha);

public: // 컨볼루션 신경망 연산
	static void conv2d(
		int device, float* py, float* px, float* pk, int64 ndat, int64 ychn, int64 yh, int64 yw,
		int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw, int pmode);
	static void conv2d_backward_x(
		int device, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw,
		int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw);
	static void conv2d_backward_k(
		int device, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw,
		int64 ychn, int64 yh, int64 yw, int64 kh, int64 kw, int64 group, int64 ph, int64 pw);

	static void conv2d_transposed(int device, float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw);
	static void conv2d_transposed_backward_x(int device, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw);
	static void conv2d_transposed_backward_k(int device, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 sh, int64 sw);

	static void conv2d_dilated(int device, float* py, float* px, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw);
	static void conv2d_dilated_backward_x(int device, float* pgx, float* pgy, float* pk, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw);
	static void conv2d_dilated_backward_k(int device, float* pgk, float* pgy, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 ychn, int64 kh, int64 kw, int64 gh, int64 gw);

	static void maxpool(int device, float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw);
	static void maxpool_backward_x(int device, float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw);

	static void avgpool(int device, float* py, float* px, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw);
	static void avgpool_backward_x(int device, float* pgx, float* pgy, int* pm, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 kh, int64 kw);

	static void globalavg(int device, float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw);
	static void globalavg_backward_x(int device, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw);

	static void adaptiveavg(int device, float* py, float* px, int64 ndat, int64 yh, int64 xchn, int64 yw, int64 hratio, int64 wratio);
	static void adaptiveavg_backward_x(int device, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 hratio, int64 wratio);

	static void stride(int device, float* py, float* px, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw);
	static void stride_backward_x(int device, float* pgx, float* pgy, int64 ndat, int64 xchn, int64 xh, int64 xw, int64 yh, int64 yw, int64 sh, int64 sw);

public: // 리커런트 신경망 연산
	static void lstm_process(int device, float* pr, float* ps, float* ps1, float* pa, int64 ndat, int64 nrec, int64 ninp);
	static void lstm_process_backward(int device, float* pgr, float* pgs, float* pga, float* ps, float* pa, int64 ndat, int64 nrec, int64 ninp);
	
	static void gru_process(int device, float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec);
	static void gru_process_backward(int device, float* pgr, float* pgai, float* pgar, float* pr, float* pai, float* par, int64 nt, int64 ntimes, int64 ndat, int64 nrec);

	/*
	static void rnn_fetch_timedata(int device, float* pd, float* pds, int64 ndat, int64 ntimes, int64 nt, int64 nrec);

	static void gru_process_rz(int device, float* pe, float* prs, float* pas, float* pa2, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec);
	static void gru_process_h(int device, float* prs, float* pas, float* pa1, int64 ndat, int64 ntimes, int64 nt, int64 nrec);
	static void gru_ext_input1_for_backward(int device, float* pe, float* px, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq);
	static void gru_load_output_backward(int device, float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq);
	static void gru_process_h_backward(int device, float* pgr, float* pga2, float* pga1, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nrec);
	static void gru_process_rz_backward(int device, float* pgr, float* pge, float* pga2, float* prs, float* pas, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec);

	static void rnn_ext_input(int device, float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq);
	static void rnn_save_output(int device, float* po, float* pr, int64 ndat, int64 ntimes, int64 nt, int64 nrec);
	static void rnn_ext_input_for_backward(int device, float* pe, float* px, float* prs, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq);
	static void rnn_ext_split_backward(int device, float* pgx, float* pgr, float* pge, int64 ndat, int64 ntimes, int64 nt, int64 nvec, int64 nrec, bool bInSeq);
	static void rnn_load_output_backward(int device, float* pgr, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq);
	static void lstm_process(int device, float* prs, float* pss, float* pas, float* pa, int64 ndat, int64 ntimes, int64 nt, int64 nrec);
	static void lstm_load_output_backward(int device, float* pgr, float* pgs, float* pgy, int64 ndat, int64 ntimes, int64 nt, int64 nrec, bool bOutSeq, bool bUseState);
	*/

public: // 정규화 연산
	static void batchnorm_norm(int device, float* py, float* px, float* pma, float* pmv, float* pba, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hMomentum, HYPER hEpsilon, bool train);
	static void batchnorm_scale(int device, float* py, float* px, float* pscale, float* pshift, int64 ndat, int64 ncol, int64 nrest);

	static void batchnorm_backward_x(int device, float* pgx, float* pgy, float* pscale, int64 ndat, int64 ncol, int64 nrest);
	static void batchnorm_backward_scale(int device, float* pgr, float* pgy, float* px, int64 ndat, int64 ncol, int64 nrest);
	static void batchnorm_backward_shift(int device, float* pgs, float* pgy, int64 ndat, int64 ncol, int64 nrest);
	static void batchnorm_backward_norm(int device, float* pgx, float* pbv, int64 ndat, int64 ncol, int64 nrest, HYPER hEpsilon);

	static void layernorm(int device, float* py, float* px, float* ps, int64 nrow, int64 ncol, HYPER hScale);
	static void layernorm_backward(int device, float* pgx, float* pgy, float* ps, int64 nrow, int64 ncol, HYPER hScale);

	static void dropout(int device, float* py, float* px, unsigned char* pm, int64 ndat, HYPER hDropRatio);
	static void dropout_backward(int device, float* pgx, float* pgy, unsigned char* pm, int64 ndat, HYPER hDropRatio);

	static void add_normal_noise(int device, float* py, float* px, int64 ndat, HYPER hMean, HYPER hStd);
	static void add_uniform_noise(int device, float* py, float* px, int64 ndat, HYPER hMin, HYPER hMax);
	static void gen_normal_random(int device, float* py, int64 ndat, HYPER hMean, HYPER hStd);
	static void gen_uniform_random(int device, float* py, int64 ndat, HYPER hMin, HYPER hMax);
	static void round(int device, float* py, float* px, int64 ndat, int prec);
	static void codeconv(int device, int* py, float* px, int64 nrow, int64 ncol);
	static void cosinesim(int device, float* py, float* px1, float* px2, int64 nrow1, int64 nrow2, int64 ncol);
	static void selectntop(int device, float* py, float* px, int64 nrow, int64 ncol, int64 ntop);
	static void selectntoparg(int device, int* py, float* px, int64 nrow, int64 ncol, int64 ntop);

public: // 트랜스포머 지원 련산
	static void embed(int device, float* py, int* px, float* pw, int64 ndat, int64 nword, int64 nvec, bool position);
	static void embed_backward_w(int device, float* pgw, float* pgy, int* px, int64 ndat, int64 nword, int64 nvec, bool position);

	static void mult_on_heads(int device, float* pp, float* pK, float* pQ, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece, HYPER hCoef);
	static void mult_on_heads_backward(int device, float* pgQKV, float* pgp, float* pQKV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece);
	static void set_mh_attention_mask(int device, float* pp, int64 ndat, int64 ntimes, int64 nhead, bool forward); 
	static void softmax_direct_on_axis(int device, float* pp, int64 nrow, int64 nvec, int64 ncol);
	static void softmax_direct_on_axis_backward(int device, float* pgp, float* pp, int64 nbat, int64 ntimes, int64 nhead, HYPER hCoef);
	static void mix_values(int device, float* py, float* pp, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece);
	static void mix_values_backward_prop(int device, float* pgp, float* pgy, float* pV, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece);
	static void mix_values_backward_value(int device, float* pV, float* pgy, float* pp, int64 nbat, int64 ntimes, int64 nvec, int64 nhead, int64 npiece);

public: // 병렬 처리 지원 연산
	static void parallel_concat(int device, float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest);
	static void parallel_concat_backward_x1(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest);
	static void parallel_concat_backward_x2(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest);

	static void stack(int device, float* py, float* px, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom);
	static void stack_backward(int device, float* pgx, float* pgy, int64 nbat, int64 nyrow, int64 nxrow, int64 ncol, int64 nfrom);

	static void concat(int device, float* py, float* px1, float* px2, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest);
	static void concat_backward_x1(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest);
	static void concat_backward_x2(int device, float* pgx, float* pgy, int64 nrow, int64 ncol1, int64 ncol2, int64 nrest);
	static void undo_concat(int device, float* py1, float* py2, float* px, int64 nrow, int64 ncol1, int64 ncol2);

public: // 기타 지원 연산
	static void count_true(int device, int* pc, float* px, int64 nrow, int64 ncol);

public: // 선택 연산
	static void select_max(int device, int* pm, int* pc, int64 nrow);
	static void to_filter(int device, int* py, float* px, int64 nrow, int64 nxcol, int64 nycol);

	static void extract(int device, float* py, float* px, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol);
	static void extract_backward(int device, float* pgx, float* pgy, int64 nrow, int64 nxvec, int64 index, int64 nyvec, int64 ncol);

	static void subvector(int device, float* py, float* px, int64 nrow, int64 ncol, int64 nfrom, int64 ncount);
	static void subvector_backward(int device, float* pgx, float* pgy, int64 nrow, int64 ncol, int64 nfrom, int64 ncount);

	static void pickup_int(int device, int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol);
	static void pickup_float(int device, float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol);

	static void pickup_float_backward(int device, float* pgx, float* pgy, int* px1, int64 nbat, int64 nrow, int64 nnom, int64 ncol);

	static void pickup_static_int(int device, int* py, int* px1, int* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol);

	static void pickup_static_float(int device, float* py, int* px1, float* px2, int64 nbat, int64 nrow, int64 nnom, int64 ncol);

public: // 객체인식 지원 연산
	static void iou_cross_xywh(int device, float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol);
	static void iou_cross_lrtb(int device, float* py, float* px1, float* px2, int64 nbat, int64 nrow, int64 ncol);
	static void iou_loss(int device, float* py, float* px1, float* px2, int64 nrow, int /*VGraphOpCode*/ op_code);
	static void iou_loss_backward(int device, float* pgx, float* pgy, float* px1, float* px2, int64 nrow, int /*VGraphOpCode*/ op_code);

	static void to_boxes(int device, float* py, float* px1, float* px2, int64 nrow);

	static void upsample(int device, float* py, float* px, int64 nbat, int64 nchn, int64 nyht, int64 nywd, int64 hratio, int64 wratio);
	static void upsample_backward(int device, float* pgx, float* pgy, int64 nbat, int64 nchn, int64 nxht, int64 nxwd, int64 hratio, int64 wratio);

public: // 활성화 함수 관련 연산
	static void sigmoid(int device, float* py, float* px, int64 nrow);
	static void sigmoid_backward(int device, float* pgx, float* pgy, float* px, int64 ndat);
	static void sigmoid_crossentropy(int device, float* py, float* px, float* pc, int64 nrow);
	static void sigmoid_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, float* pz, int64 nrow);
	static void sigmoid_crossentropy_with_logits(int device, float* py, float* px, float z, int64 ndat);
	static void sigmoid_crossentropy_with_logits_backward(int device, float* pgx, float* pgy, float* px, float z, int64 ndat);
	static void sigmoid_crossentropy_with_logits_idx(int device, float* py, float* px, int* pz, int64 nrow, int64 ncol);
	static void sigmoid_crossentropy_with_logits_idx_backward(int device, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol);

	static void softmax_idx_crossentropy(int device, float* py, float* px, int* pz, int64 nrow, int64 ncol);
	static void softmax_i64_idx_crossentropy(int device, float* py, float* px, int64* pz, int64 nrow, int64 ncol);
	static void softmax_idx_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol);
	static void softmax_i64_idx_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, int64* pz, int64 nrow, int64 ncol);
	static void softmax_idx_crossentropy_pos_idx(int device, float* py, float* px, int* pz, int64 nrow, int64 ncol);
	static void softmax_idx_crossentropy_pos_idx_backward_x(int device, float* pgx, float* pgy, float* px, int* pz, int64 nrow, int64 ncol);
	static void softmax_crossentropy(int device, float* py, float* px, float* pz, int64 nrow, int64 ncol);
	static void softmax_crossentropy_backward_x(int device, float* pgx, float* pgy, float* px, float* pz, int64 nrow, int64 ncol);

public: // 형상 관리 관련 연산
	static void transpose(int device, float* py, float* px, int* pn, int64 axis_size, int64 data_size);
	static void transpose_backward(int device, float* pgx, float* pgy, int* pn, int64 axis_size, int64 data_size);
	static void transpose_bin(int device, float* py, float* px, int64 nrow, int64 ncol);

public: // 최대 최소 추출 연산
	static void max(int device, float* py, float* px, int64 nrow, int64 ncol);
	static void min(int device, float* py, float* px, int64 nrow, int64 ncol);

	static void argmax(int device, int* py, float* px, int64 nrow, int64 ncol);
	static void argmin(int device, int* py, float* px, int64 nrow, int64 ncol);

public: // 집계 연산
	static void mean(int device, float* py, float* px, int64 nrow);
	static void mean_backward(int device, float* pgx, float* pgy, int64 nrow);

	static void sum_int(int device, int* py, int* px, int64 nrow);
	static void sum(int device, float* py, float* px, int64 nrow);
	static void sum_backward(int device, float* pgx, float* pgy, int64 nrow);

public: // 유틸리티 지원 연산
	static void fft_wave_to_complex(float* pbuf, float* pwave, int64 bsize, int64 spec_interval, int64 step_cnt, int64 fft_width, int64 samples_in_data);
	static void fft_step_split(float* pdst, float* psrc, int64 ssize, int64 fft_width, int64 step);
	static void fft_complex_to_abs_mean(float* pffts, float* psrc, int64 fsize, int64 fft_width, int64 freq_in_spectrum);
};
