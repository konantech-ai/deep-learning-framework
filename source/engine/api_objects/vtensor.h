#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vgraph.h"
#include "../local_objects/vexectracer.h"
#include "../local_objects/vcbbackinfo.h"
#include "../local_objects/vhypermanager.h"

enum class TensorCloneInit { share, alloc, /* copy, */ empty, zeros };

class VTensorData;
class VBackQueue;
class VLossCore;
class VTensorCore;

class VTensor {
public:
    VTensor();
    VTensor(VSession session, string sBuiltin, VDict kwArgs = {});
    VTensor(const VTensor& src);
    VTensor(VSession session, VHTensor handle);
    VTensor(VTensorCore* core);
    virtual ~VTensor();
    VTensor& operator =(const VTensor& src);
    VHTensor cloneCore();
    VHTensor cloneHandle();
    VTensorCore* getClone();
    VTensorCore* getCore();
    bool isValid();
    void closeHandle();
    VSession session();
    int getNth();
    int getRefCnt();
    void incRefCount();
protected:
    VTensorCore* m_core;

public:
    VTensor(VSession session);
    VTensor(VSession session, VShape shape, VDataType type, int nDevice);
    VTensor(VTensor src, VShape shape, TensorCloneInit init);

    VShape shape();
    VDataType type();
    int device();

    bool hasData();
    bool hasNoData();
    bool needGrad();

    void setNeedGrad(bool needGrad);

    VTensorData data();

    int64 byte_size();

    void* void_ptr();
    int* int_ptr();
    int64* int64_ptr();
    float* float_ptr();
    unsigned char* uchar_ptr();

    void setFeature(VShape shape, VDataType type, int nDevice);
    void getFeature(VShape* pshape, VDataType* ptype, int* pnDevice);

    void allocData(int nDevice);
    void setZero(VExecTracer tracer);
    void initParam(TensorInitMethod init_op, float mean, float init_arg, bool adaptive);
    void initParam();

    void uploadData(void* pData, int64 nByteSize);
    void downloadData(void* pData, int64 nByteSize);

    void setLossFunc(VLossCore* pLossCore);

    void backward();
    void backwardWithGradient(VTensor grad);

    void dump(string title, bool full = false);
    void dump1(string title, bool full = false);
    void dump_arr_feat(int nth, string title);

    void setCbBackSlot(VCbBackSlot slot);
    void resetCbBackSlot(int sid);

    VDict getOpArgs();
    VDict getMyArgs();

    /*
    void copyRnnWeight(string nameDot, VTensorDict tensors, string mode);
    void copyRnnBias(string nameDot, VTensorDict tensors, string mode);
    void copyLinearWeight(string nameDot, VTensorDict tensors, string mode);
    */

    //void copyWeightParam(VTensor tensor, int ngate, string mode);
    //void copyBiasParam(VTensor tensor, int ngate, string mode);

    void copyParam(VTensor tensor, string mode);

public:
    VTensor getNthSlice(int nDivisions, int nth, int nDevice, VExecTracer tracer);
    VTensor getSlicePiece(int64 nRowFrom, int64 nRowCount, int nDevice, VExecTracer tracer);

    VTensor toDevice(int nDevice, VExecTracer tracer);
    
    int64 copySliceFrom(VTensor slice, int64 startRow, VExecTracer tracer);

    void setElement(VList pos, VValue value, VExecTracer tracer);

    VValue getElement(VList pos, VExecTracer tracer);

    void setOpArgs(VDict args);
    void setMyArgs(VDict args=VDict());

    void setOpArg(string name, VValue value);

    VDict detachOpArgs();

    VValue getOpArg(string name, VValue def = VValue());
    VValue getMyArg(string name);

    VValue getOpArg(string name, int nth, VValue def = VValue());
    VValue getMyArg(string name, int nth);

    void fill_float(float value, VExecTracer tracer);
    void fill_int(int value, VExecTracer tracer);

public:
    VTensor matmul(VTensor w, VExecTracer tracer);
    VTensor conv2d(VTensor k, VExecTracer tracer);
    VTensor conv2d_transposed(VTensor k, VExecTracer tracer);
    VTensor conv2d_dilated(VTensor k, VExecTracer tracer);
    VTensor rnn(bool train, VTensor pmset, VExecTracer tracer);
    VTensor lstm(bool train, VTensor pmset, VExecTracer tracer);
    VTensor gru(bool train, VTensor pmset, VExecTracer tracer);
    VTensor add(VTensor opnd2, VExecTracer tracer);
    VTensor add_2d_bias(VTensor opnd2, VExecTracer tracer);
    VTensor subtract(VTensor opnd2, VExecTracer tracer);
    VTensor mult(VTensor opnd2, VExecTracer tracer);
    VTensor se_mult(VTensor opnd2, VExecTracer tracer);
    VTensor div(VTensor opnd2, VExecTracer tracer);
    VTensor add_residual(VTensor opnd2, VExecTracer tracer);
    VTensor maximum(VTensor opnd2, VExecTracer tracer);
    VTensor minimum(VTensor opnd2, VExecTracer tracer);
    VTensor _not(VExecTracer tracer);
    VTensor _and(VTensor opnd2, VExecTracer tracer);
    VTensor _or(VTensor opnd2, VExecTracer tracer);
    VTensor equal(VExecTracer tracer);
    VTensor greater_than(VExecTracer tracer);
    VTensor less_than(VExecTracer tracer);
    VTensor greater_equal(VExecTracer tracer);
    VTensor less_equal(VExecTracer tracer);
    VTensor equal(VTensor opnd2, VExecTracer tracer);
    VTensor greater_than(VTensor opnd2, VExecTracer tracer);
    VTensor less_than(VTensor opnd2, VExecTracer tracer);
    VTensor greater_equal(VTensor opnd2, VExecTracer tracer);
    VTensor less_equal(VTensor opnd2, VExecTracer tracer);
    VTensor greater_than_cross(VTensor opnd2, VExecTracer tracer);
    VTensor less_than_cross(VTensor opnd2, VExecTracer tracer);
    VTensor greater_equal_cross(VTensor opnd2, VExecTracer tracer);
    VTensor less_equal_cross(VTensor opnd2, VExecTracer tracer);
    VTensor pickup(VTensor opnd2, VExecTracer tracer);
    VTensor pickup_static(VTensor opnd2, VExecTracer tracer);
    VTensor iou_cross_xywh(VTensor opnd2, VExecTracer tracer);
    VTensor iou_cross_lrtb(VTensor opnd2, VExecTracer tracer);
    VTensor iou_loss(VTensor opnd2, VGraphOpCode op_code, VExecTracer tracer);
    VTensor crossentropy(VTensor opnd2, VExecTracer tracer);
    VTensor crossentropy_sigmoid(VTensor opnd2, VExecTracer tracer);
    VTensor crossentropy_pos_idx(VTensor opnd2, VExecTracer tracer);
    VTensor sigmoid(VExecTracer tracer);
    VTensor sigmoid_crossentropy_with_logits(VExecTracer tracer);
    VTensor sigmoid_crossentropy_with_logits_idx(VTensor opnd2, VExecTracer tracer);
    VTensor flatten(VExecTracer tracer);
    VTensor activate(VExecTracer tracer);
    VTensor normal_noise(VExecTracer tracer);
    VTensor uniform_noise(VExecTracer tracer);
    VTensor normal_random(VExecTracer tracer);
    VTensor uniform_random(VExecTracer tracer);
    VTensor round(VExecTracer tracer);
    VTensor codeconv(VExecTracer tracer);
    VTensor cosinesim(VTensor opnd2, VExecTracer tracer);
    VTensor selectntop(VExecTracer tracer);
    VTensor selectntoparg(VExecTracer tracer);
    VTensor mean(VExecTracer tracer);
    VTensor sum(VExecTracer tracer);
    VTensor abs(VExecTracer tracer);
    VTensor square(VExecTracer tracer);
    VTensor sqrt(VExecTracer tracer);
    VTensor exp(VExecTracer tracer);
    VTensor log(VExecTracer tracer);
    VTensor upsample(VExecTracer tracer);
    VTensor pass(VExecTracer tracer);
    VTensor reshape(VExecTracer tracer);
    VTensor transpose(VExecTracer tracer);
    VTensor extract(VExecTracer tracer);
    VTensor stride(VExecTracer tracer);
    VTensor maxpool(VExecTracer tracer);
    VTensor avgpool(VExecTracer tracer);
    VTensor max(VExecTracer tracer);
    VTensor min(VExecTracer tracer);
    VTensor argmax(VExecTracer tracer);
    VTensor argmin(VExecTracer tracer);
    VTensor globalavg(VExecTracer tracer);
    VTensor adaptiveavg(VExecTracer tracer);
    VTensor embed(VTensor w, VExecTracer tracer);
    VTensor batchnorm(VTensor mavg, VTensor mvar, VTensor rescale, VTensor shift, bool train, VExecTracer tracer);
    VTensor layernorm(VExecTracer tracer);
    VTensor mh_attention(VTensor key, VTensor query, VTensor value, VTensor Kw, VTensor Kb, VTensor Qw, VTensor Qb, VTensor Vw, VTensor Vb, VTensor Ow, VTensor Ob, VExecTracer tracer);
    VTensor dropout(bool train, VExecTracer tracer);
    VTensor concat(VTensor opnd2, VExecTracer tracer);
    VTensor subvector(VExecTracer tracer);
    VTensor to_filter(VExecTracer tracer);
    VTensor to_boxes(VTensor opnd2, VExecTracer tracer);
    VTensor complement_1(VExecTracer tracer);
    VTensor parallel_concat(VTensor opnd2, VExecTracer tracer);

    static VTensor stack_all(VSession session, VTensorDict xs, int64 tail_size, VExecTracer tracer);

public:
    bool is_pm();
    string getOpName();

    void invokeBackpropCallback(VTensor grad, VExecTracer tracer);

    void pm_backprop(VTensor grad, VExecTracer tracer);
    void accGrad(VTensor grad, VExecTracer tracer);

    void backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void backprop_noGgrad(VBackQueue* pQueue);

    void user_defined_func_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);

    void mean_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void add_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void add_2d_bias_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void add_residual_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void matmul_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void conv2d_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void conv2d_transposed_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void conv2d_dilated_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void conv2d_depthwise_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void rnn_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void lstm_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void gru_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void sum_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void subtract_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void mult_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void se_mult_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void div_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void maximum_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void minimum_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void abs_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void square_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void sqrt_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void exp_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void log_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void sigmoid_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void upsample_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void activate_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void normal_noise_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void uniform_noise_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void normal_random_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void uniform_random_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void round_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void codeconv_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void cosinesim_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void selectntop_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void selectntoparg_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void reshape_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void flatten_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void transpose_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void extract_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void dropout_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void stride_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void maxpool_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void avgpool_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void globalavg_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void adaptiveavg_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void embed_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void batchnorm_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void layernorm_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void multihead_attention_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void subvector_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void pickup_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void concat_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void pass_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void crossentropy_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void crossentropy_sigmoid_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void crossentropy_pos_idx_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void sigmoid_crossentropy_with_logits_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void sigmoid_crossentropy_with_logits_idx_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void iou_loss_backprop(VTensor ygrad, VBackQueue* pQueue, VGraphOpCode op_code, VExecTracer tracer);
    void complement_1_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);
    void parallel_concat_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);

    void stack_all_backprop(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);

protected:
    VTensor m_rnn_base(bool train, string mode, VTensor pmset, VExecTracer tracer);
    void m_rnn_backprop_base(VTensor ygrad, VBackQueue* pQueue, VExecTracer tracer);

    VTensor m_rnn(VTensor& recurs_internal, string cell_type, bool forward, VTensor x, VTensor wi, VTensor wr, VTensor bi, VTensor br, string prefix, VExecTracer tracer);
    VTensor m_rnn_backprop(string cell_type, bool forward, VTensor rgrad, VTensor rtgrad, VTensor wi, VTensor wr, VTensor bi, VTensor br, string prefix, VBackQueue* pQueue, VExecTracer tracer);

    VTensor m_mergeRnnBidirectPairForInput(VTensor for_y, VTensor rev_y, VExecTracer tracer);
    
    VTensor m_pickupParam(VDict params, string name);

public:
    void keepBackpropMergeInfo(VModule ownModule, VTensor operand);
    void keepBackpropParamInfo(VGraphOpCode opCode, VTensor grad);
    void keepBackpropOperandsInfo(bool needGrad, VGraphOpCode opCode, VTensorList operands);

    int incOperandRefCount();
    int decOperandRefCount();
    int getOperandRefCount();

    VTensor getNthOperand(int nth);

    static void extract_on(VTensor y, VTensor x, int64 axis, int64 index, int64 count, bool reduce_axis, VExecTracer tracer);

    // public member인 아래 method들의 이름을 일관성 있게 수정할 것
    void m_undo_concat(VTensor dst1, VTensor dst2, VExecTracer tracer);

    VTensor m_dropout(HYPER_KEY drop_ratio, VTensor& mask, VExecTracer tracer);
    VTensor m_dropout_backprop(VTensor ygrad,VTensor mask, HYPER_KEY drop_ratio, VExecTracer tracer);

public:
    void transpose_on(VTensor src, VList axes, VExecTracer tracer);   // yolo4.weights 등 darknet에서 저장한 kernel 등을 올바로 읽어들이기 위한 보정 함수

protected:
    void m_dump(VShape shape, void* pHostBuf, int64 nth, int indent, bool full);
    //void m_dumpElement(void* pHostBuf, int64 n);
    void m_dumpNthElement(void* pHostBuf, int64 n, bool bSpaceOnLeft=true);
    
    string m_getElementDesc(void* pHostBuf, int64 n);

    VShape m_binop_shape_check(bool bCheckSE, VShape& shape1, VShape& shape2, int64& left1, int64& left2, int64& mid, int64& right1, int64& right2);

protected:
    static bool ms_bBlockDump;

    static mutex ms_grad_mutex;
    static mutex ms_dump_mutex;

};
