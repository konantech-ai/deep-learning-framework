#include <cuda_runtime.h>

#include "../api_objects/voptimizer_core.h"
#include "../local_objects/vexectracer_core.h"
#include "../api_objects/voptimizer.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vtensor.h"
#include "../local_objects/vdevicemanager.h"
#include "../support/vmath.h"
#include "../api/vconst.h"
#include "../utils/vutils.h"

int VOptimizerCore::ms_nCheckCode = 96737307;

VStrList VOptimizer::ms_builtin = { "sgd", "adam", "momentum", "nesterov", "adagrad", "rmsprop" };

//=========== API Object Common Part Start =======================

VOptimizer::VOptimizer() {
    m_core = NULL;
}

VOptimizer::VOptimizer(const VOptimizer& src) {
    m_core = src.m_core->clone();
}

VOptimizer::VOptimizer(VOptimizerCore* core) {
    m_core = core->clone();
}

VOptimizer::VOptimizer(VSession session, string sBuiltin, VDict kwArgs) {
    m_core = new VOptimizerCore(session, sBuiltin, kwArgs);
}

VOptimizer::VOptimizer(VSession session, VHOptimizer handle) {
    m_core = NULL;
    VOptimizerCore* core = (VOptimizerCore*)handle;
    if (core == NULL) VP_THROW1(VERR_INVALID_CORE, "Optimizer");
    if (core->m_nCheckCode != VOptimizerCore::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, "Optimizer");
    if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, "Optimizer");
    m_core = (VOptimizerCore*)core->clone_core();
}

VOptimizer::~VOptimizer() { m_core->destroy(); }

VOptimizer& VOptimizer::operator =(const VOptimizer& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VHOptimizer VOptimizer::cloneCore() {
    return (VHOptimizer)m_core->clone();
}

VHOptimizer VOptimizer::cloneHandle() {
    return (VHOptimizer)m_core->clone_handle();
}

VOptimizerCore* VOptimizer::getClone() {
    return (VOptimizerCore*)m_core->clone_core();
}

VOptimizerCore* VOptimizer::getCore() {
    return m_core;
}

bool VOptimizer::isValid() {
    return m_core != NULL;
}
void VOptimizer::closeHandle() {
    if (this) m_core->destroy_handle();
}

VSession VOptimizer::session() {
    return m_core->m_session;
}

int VOptimizer::getRefCnt() {
    return m_core->getRefCnt();
}

int VOptimizer::getNth() {
    return m_core->getNth();
}

void VOptimizer::incRefCount() {
    m_core->incRefCnt();
}

VOptimizerCore::VOptimizerCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Optimizer) {
    m_nCheckCode = ms_nCheckCode;
    m_session = session;
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;

    m_onCreate();
}

VOptimizerCore::~VOptimizerCore() {
    m_onDelete();
    m_nCheckCode = 0;
}

//=========== API Object Common Part End =======================

VOptimizer::VOptimizer(VSession session, VParameters parameters, string sBuiltin, VDict kwArgs) {
    m_core = new VOptimizerCore(session, sBuiltin, kwArgs);
    m_core->m_setup(parameters);
}

VList VOptimizer::GetBuiltinNames() {
    VList list;
    for (auto& it : ms_builtin) list.push_back(it);
    return list;
}

void VOptimizer::createParam(VSession session, string name, VShape shape, bool needGrad, string init_method, float init_arg, VDict pm) {
    if (shape.total_size() == 0) VP_THROW(VERR_INPUT_SHAPE);

    pm["type"] = "standalone";
    
    pm[name] = VOptimizerCore::ms_init_module_pmset(session, shape, -1, needGrad, init_method, init_arg);
}

void VOptimizer::createEmptyParam(VSession session, string name, VDict pm) {
    pm[name] = VOptimizerCore::ms_create_empty_pmset(session);
}

void VOptimizer::createAffineParam(VSession session, VShape wshape, bool use_bias, string init_method, float init_arg, VDict pm, string prefix) {
    if (wshape.total_size() == 0) VP_THROW(VERR_INPUT_SHAPE);

    pm["type"] = "affine";
    pm["use_bias"] = use_bias;

    pm[prefix + "w"] = VOptimizerCore::ms_init_module_pmset(session, wshape, -1, true, init_method, init_arg);

    if (use_bias)
        pm[prefix + "b"] = VOptimizerCore::ms_init_module_pmset(session, VShape{ wshape[0] }, -1, true, init_method, init_arg);
    else
        pm[prefix + "b"] = VOptimizerCore::ms_create_empty_pmset(session);
}

void VOptimizer::createRnnParam(VSession session, int nGates, int64 nRecurSize, int64 nInputSize, int64 nLayers, bool bi_direct, bool use_bias, VDict pm) {
    pm["type"] = "rnn";
    pm["use_bias"] = use_bias;

    VShape wishape = VShape{ nGates * nRecurSize, nInputSize };
    VShape wrshape = VShape{ nGates * nRecurSize, nRecurSize };

    //float range = ::sqrt(1.0f / (float)wishape[0]);
    float range = ::sqrt(1.0f / (float)nRecurSize);

    for (int64 n = 0; n < nLayers; n++) {
        ms_createUniformParam(session, wishape, wrshape, range, use_bias, pm, "L" + std::to_string(n) + "F_");
        nInputSize = nRecurSize;

        if (bi_direct) {
            ms_createUniformParam(session, wishape, wrshape, range, use_bias, pm, "L" + std::to_string(n) + "R_");
            nInputSize += nRecurSize;
        }

        wishape = VShape{ nGates * nRecurSize, nInputSize };
        wrshape = VShape{ nGates * nRecurSize, nRecurSize };

        //float range = ::sqrt(1.0f / (float)wishape[0]);
    }
}

void VOptimizer::ms_createUniformParam(VSession session, VShape wishape, VShape wrshape, float range, bool use_bias, VDict pm, string prefix) {
    pm[prefix + "iw"] = VOptimizerCore::ms_init_module_pmset(session, wishape, -1, true, "uniform", range);
    pm[prefix + "rw"] = VOptimizerCore::ms_init_module_pmset(session, wrshape, -1, true, "uniform", range);

    if (use_bias) {
        pm[prefix + "ib"] = VOptimizerCore::ms_init_module_pmset(session, VShape{ wishape[0] }, -1, true, "uniform", range);
        pm[prefix + "rb"] = VOptimizerCore::ms_init_module_pmset(session, VShape{ wrshape[0] }, -1, true, "uniform", range);
    }
    else {
        pm[prefix + "ib"] = VOptimizerCore::ms_create_empty_pmset(session);
        pm[prefix + "rb"] = VOptimizerCore::ms_create_empty_pmset(session);
    }
}

/*
void VOptimizer::createLstmParam(VSession session, VShape wshape, bool use_bias, VDict pm, string prefix) {
    if (wshape.total_size() == 0) VP_THROW(VERR_INTERNAL_LOGIC_ERROR);

    float range = ::sqrt(1.0f / (float)wshape[0]);

    pm["type"] = "lstm";
    pm[prefix + "w"] = VOptimizerCore::ms_init_module_pmset(session, wshape, -1, true, "uniform", range); 

    if (use_bias)
        pm[prefix + "b"] = VOptimizerCore::ms_init_module_pmset(session, VShape{ wshape[0] }, -1, true, "uniform", range);
    else
        pm[prefix + "b"] = VOptimizerCore::ms_create_empty_pmset(session);
}
*/

void VOptimizer::createBiasParam(VSession session, VShape bshape, string init_method, float init_arg, VDict pm) {
    if (bshape.total_size() == 0) VP_THROW(VERR_INPUT_SHAPE);

    pm["type"] = "bias";
    pm["b"] = VOptimizerCore::ms_init_module_pmset(session, bshape, -1, true, init_method, init_arg);
}

void VOptimizer::createBatchNormalParam(VSession session, VShape xshape, bool rescale, bool shift, VDict pm) {
    VDict pm_moving_stat;

    pm["type"] = "batchnorm";
    if (xshape.size() > 0) {
        VShape wshape{ xshape[-1] };

        pm["mavg"] = VOptimizerCore::ms_init_module_pmset(session, wshape, -1, false, "zeros", 0);
        pm["mvar"] = VOptimizerCore::ms_init_module_pmset(session, wshape, -1, false, "zeros", 0);

        if (rescale) {
            pm["rescale"] = VOptimizerCore::ms_init_module_pmset(session, wshape, -1, true, "ones", 0);
        }
        else
            pm["rescale"] = VOptimizerCore::ms_create_empty_pmset(session);

        if (rescale && shift)
            pm["shift"] = VOptimizerCore::ms_init_module_pmset(session, wshape, -1, true, "zeros", 0);
        else
            pm["shift"] = VOptimizerCore::ms_create_empty_pmset(session);
    }
    else {
        pm["mavg"] = VOptimizerCore::ms_create_empty_pmset(session);
        pm["mvar"] = VOptimizerCore::ms_create_empty_pmset(session);
        pm["rescale"] = VOptimizerCore::ms_create_empty_pmset(session);
        pm["shift"] = VOptimizerCore::ms_create_empty_pmset(session);
    }
}

void VOptimizer::set_option(VDict kwArgs) {
    m_core->m_setOption(kwArgs);
}

void VOptimizer::step() {
    VExecTracer tracer = m_core->m_execTracer;

    if (!tracer.isValid() && !session().getNoTracer()) {
        tracer = VExecTracer(session(), "optimizer", {});
        m_core->m_execTracer = tracer;
    }
    else if (tracer.isValid() && session().getNoTracer()) {
        tracer = VExecTracer();
        m_core->m_execTracer = tracer;
    }

    if (tracer.hasValidHistory({})) {
        if (session().isTracerDirty()) {
            tracer.reset();
        }
        else {
            tracer.executeHistory();
            return;
        }
    }

    tracer.setInput({});

    m_core->m_step(m_core->m_parameters.getParams(), tracer);

    tracer.closeRecording({});
}

VTensor VOptimizer::preproc(VTensor param, VDict pmset, VExecTracer tracer) {
    if (m_core == NULL) return param;

    if (m_core->m_optAlgorithm == OptAlgorithm::nesterov) {
        return m_core->m_preprocNesterovParam(param, pmset, tracer);
    }
    else {
        return param;
    }
}

//-----------------------------------------------------------------------------------------------------
// 코어 영역 확장 코드

void VOptimizerCore::m_onCreate() {
    HYPER_REGIST(m_learning_rate);

    HYPER_REGIST(m_l1Decay);
    HYPER_REGIST(m_l2Decay);

    HYPER_REGIST(m_ro1);
    HYPER_REGIST(m_ro2);
    HYPER_REGIST(m_epsilon);
    HYPER_REGIST(m_sigma);
    HYPER_REGIST(m_decay);
    HYPER_REGIST(m_momentum);

    float learning_rate = vutils.seek_dict(m_propDict, "learning_rate", 0);
    if (learning_rate == 0) learning_rate = vutils.seek_dict(m_propDict, "lr", 0.01f);

    HYPER_SET(m_learning_rate, learning_rate);

    HYPER_SET(m_l1Decay, vutils.seek_dict(m_propDict, "l1_decay", 0));
    HYPER_SET(m_l2Decay, vutils.seek_dict(m_propDict, "l2_decay", 0));

    m_bUseDecay = (HYPER_GET(m_l1Decay) > 0) || (HYPER_GET(m_l2Decay) > 0);

    m_optAlgorithm = VConsts::getOptAlgorithm(m_sBuiltin);
    
    if (m_optAlgorithm == OptAlgorithm::sgd);
    else if (m_optAlgorithm == OptAlgorithm::adam) {
        HYPER_SET(m_ro1, vutils.seek_dict(m_propDict, "ro1", 0.9f));
        HYPER_SET(m_ro2, vutils.seek_dict(m_propDict, "ro2", 0.999f));
        HYPER_SET(m_epsilon, vutils.seek_dict(m_propDict, "epsilon", 1.0e-8f));
    }
    else if (m_optAlgorithm == OptAlgorithm::momentum || m_optAlgorithm == OptAlgorithm::nesterov) {
        HYPER_SET(m_momentum, vutils.seek_dict(m_propDict, "momentum", 0.9f));
        m_initMethod = (string)vutils.seek_dict(m_propDict, "moment_init", "zeros");
    }
    else if (m_optAlgorithm == OptAlgorithm::adagrad) {
        HYPER_SET(m_sigma, vutils.seek_dict(m_propDict, "sigma", 1e-7f));
        m_initMethod = (string)vutils.seek_dict(m_propDict, "moment_init", "zeros");
    }
    else if (m_optAlgorithm == OptAlgorithm::rmsprop) {
        HYPER_SET(m_sigma, vutils.seek_dict(m_propDict, "sigma", 1e-6f));
        HYPER_SET(m_decay, vutils.seek_dict(m_propDict, "decay", 0.9f));
        m_initMethod = (string)vutils.seek_dict(m_propDict, "moment_init", "zeros");
    }
    else {
        VP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void VOptimizerCore::m_onDelete() {}

void VOptimizerCore::m_setup(VParameters parameters) {
    m_parameters = parameters;

    int nDevice = (parameters.getDevice() == "cpu") ? -1 : 0;

    m_prepareParam(m_parameters.getParams(), nDevice);
}

void VOptimizerCore::m_setOption(VDict kwArgs) {
    float learning_rate = HYPER_GET(m_learning_rate);

    learning_rate = vutils.seek_dict(kwArgs, "learning_rate", learning_rate);
    learning_rate = vutils.seek_dict(kwArgs, "lr", learning_rate);

    HYPER_SET(m_learning_rate, learning_rate);

    HYPER_SET(m_l1Decay, vutils.seek_dict(kwArgs, "l1_decay", HYPER_GET(m_l1Decay)));
    HYPER_SET(m_l2Decay, vutils.seek_dict(kwArgs, "l2_decay", HYPER_GET(m_l2Decay)));

    m_bUseDecay = (HYPER_GET(m_l1Decay) > 0) || (HYPER_GET(m_l2Decay) > 0);

    if (m_optAlgorithm == OptAlgorithm::sgd);
    else if (m_optAlgorithm == OptAlgorithm::adam) {
        HYPER_SET(m_ro1, vutils.seek_dict(kwArgs, "ro1", HYPER_GET(m_ro1)));
        HYPER_SET(m_ro2, vutils.seek_dict(kwArgs, "ro2", HYPER_GET(m_ro2)));
        HYPER_SET(m_epsilon, vutils.seek_dict(kwArgs, "epsilon", HYPER_GET(m_epsilon)));
    }
    else if (m_optAlgorithm == OptAlgorithm::momentum || m_optAlgorithm == OptAlgorithm::nesterov) {
        HYPER_SET(m_momentum, vutils.seek_dict(m_propDict, "momentum", HYPER_GET(m_momentum)));
        m_initMethod = (string)vutils.seek_dict(m_propDict, "moment_init", m_initMethod);
    }
    else if (m_optAlgorithm == OptAlgorithm::adagrad) {
        HYPER_SET(m_sigma, vutils.seek_dict(m_propDict, "sigma", HYPER_GET(m_sigma)));
        m_initMethod = (string)vutils.seek_dict(m_propDict, "moment_init", m_initMethod);
    }
    else if (m_optAlgorithm == OptAlgorithm::rmsprop) {
        HYPER_SET(m_sigma, vutils.seek_dict(m_propDict, "sigma", HYPER_GET(m_sigma)));
        HYPER_SET(m_decay, vutils.seek_dict(m_propDict, "decay", HYPER_GET(m_decay)));
        m_initMethod = (string)vutils.seek_dict(m_propDict, "moment_init", m_initMethod);
    }
    else {
        VP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    m_execTracer.reset();
}

void VOptimizerCore::m_prepareParam(VList params, int nDevice) {
    for (auto& it : params) {
        if (it.is_list()) m_prepareParam((VList)it, nDevice);
        else if (it.is_dict()) m_prepareParam((VDict)it, nDevice);
        else VP_THROW(VERR_CONDITIONAL_STATEMENT);
    }
}

void VOptimizerCore::m_prepareParam(VDict params, int nDevice) {
    for (auto& it : params) {
        if (it.second.is_dict()) m_preparePmSet((VDict)it.second, nDevice);
    }
}

void VOptimizerCore::m_preparePmSet(VDict pmset, int nDevice) {
    VTensor param(m_session, (VHTensor)pmset["pm"]);

    if (m_optAlgorithm == OptAlgorithm::sgd);
    else if (m_optAlgorithm == OptAlgorithm::adam) {
        VShape pshape = param.shape();

        VTensor s = ms_init_module_pm_tensor(m_session, pshape, nDevice, false, "zeros", 0);
        VTensor t = ms_init_module_pm_tensor(m_session, pshape, nDevice, false, "zeros", 0);
        VTensor n = ms_init_module_pm_tensor(m_session, pshape, nDevice, false, "zeros", 0);

        pmset["s"] = s.cloneCore();
        pmset["t"] = t.cloneCore();
        pmset["n"] = n.cloneCore();
    }
    else if (m_optAlgorithm == OptAlgorithm::momentum || m_optAlgorithm == OptAlgorithm::nesterov) {
        VShape pshape = param.shape();

        VTensor v = ms_init_module_pm_tensor(m_session, pshape, nDevice, false, m_initMethod, 0);

        pmset["v"] = v.cloneCore();
    }
    else if (m_optAlgorithm == OptAlgorithm::adagrad || m_optAlgorithm == OptAlgorithm::rmsprop) {
        VShape pshape = param.shape();

        VTensor r = ms_init_module_pm_tensor(m_session, pshape, nDevice, false, m_initMethod, 0);

        pmset["r"] = r.cloneCore();
    }
    else {
        VP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

VDict VOptimizerCore::ms_init_module_pmset(VSession session, VShape shape, int nDevice, bool bNeedGrad, string init_type, float init_arg) {
    VTensor param = ms_init_module_pm_tensor(session, shape, nDevice, bNeedGrad, init_type, init_arg);
    VTensor grad(param, shape, TensorCloneInit::empty);

    VDict pmset{ {"pm", param.cloneCore() }, {"grad", grad.cloneCore() } };

    return pmset;
}

VDict VOptimizerCore::ms_create_empty_pmset(VSession session) {
    VTensor param(session);
    VTensor grad(session);

    VDict pmset{ {"pm", param.cloneCore() }, {"grad", grad.cloneCore() } };

    return pmset;
}

VTensor VOptimizerCore::ms_init_module_pm_tensor(VSession session, VShape shape, int nDevice, bool bNeedGrad, string init_method, float init_arg) {
    VTensor param(session, "param");

    if (shape.size() > 0) {
        TensorInitMethod init_op;
        float mean = 0;
        bool adaptive = false;

        if (init_method == "zeros") init_op = TensorInitMethod::fzeros;
        else if (init_method == "ones")  init_op = TensorInitMethod::fones;
        else if (init_method == "uniform" || init_method == "xavier" || init_method == "normalized_xavier") init_op = TensorInitMethod::random_uniform;
        else if (init_method == "random" || init_method == "gaussian" || init_method == "he" || init_method == "gauss" || init_method == "normal") init_op = TensorInitMethod::random_normal;
        else if (init_method == "adaptive_gaussian" || init_method == "adaptive_gauss" || init_method == "adaptive_normal") {
            init_op = TensorInitMethod::random_normal;
            adaptive = true;
        }
        else {
            VP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }

        param.setFeature(shape, VDataType::float32, nDevice);
        param.initParam(init_op, mean, init_arg, adaptive);

        param.setNeedGrad(bNeedGrad);
    }

    return param;
}

void VOptimizerCore::m_step(VList params, VExecTracer tracer) {
    for (auto& it : params) {
        if (it.is_list()) m_step((VList)it, tracer);
        else if (it.is_dict()) {
            VDict param = it;
            for (auto& it2 : param) {
                if (it2.second.is_dict()) {
                    m_optimize(it2.second, tracer);
                }
                int nnn = 0;
            }
        }
        else {
            VP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }
    }
}

void VOptimizerCore::m_optimize(VDict pmset, VExecTracer tracer) {
    VTensor param = tracer.createTensor(m_session, (VHTensor)pmset["pm"]);
    VTensor grad = tracer.createTensor(m_session, (VHTensor)pmset["grad"]);
    //grad.dump("grad1");

    if (param.shape() != grad.shape()) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);

    int64 ndat = param.shape().total_size();

    if (param.device() >= 0) VP_THROW(VERR_UNDEFINED);

    if (ndat == 0) return;

    float* pp = param.float_ptr();

    if (grad.shape().size() <= 0) return;

    switch (m_optAlgorithm) {
    case OptAlgorithm::sgd:
    {
        if (m_bUseDecay) grad = m_applyDecay(pmset, grad, tracer);

        VTensor hostGrad = tracer.createTensor(m_session, grad.shape(), VDataType::float32, -1);

        float* pg = grad.float_ptr();
        float* ph = hostGrad.float_ptr();

        CALL_MATH(memcpy_to_host, grad.device(), ph, pg, grad.byte_size());
        CALL_MATH(subtract_param_grad, -1, pp, ph, ndat, HYPER_FETCH_CPU(m_learning_rate));
    }
        break;
    case OptAlgorithm::adam:
    {
        grad = m_evalAdamDelta(pmset, grad, tracer);
        if (m_bUseDecay) grad = m_applyDecay(pmset, grad, tracer);

        VTensor hostGrad = tracer.createTensor(m_session, grad.shape(), VDataType::float32, -1);

        float* pg = grad.float_ptr();
        float* ph = hostGrad.float_ptr();

        CALL_MATH(memcpy_to_host, grad.device(), ph, pg, grad.byte_size());
        CALL_MATH(subtract_param_grad, -1, pp, ph, ndat, HYPER_FETCH_CPU(m_learning_rate));
    }
        break;
    case OptAlgorithm::momentum:
    case OptAlgorithm::nesterov:
    {
        if (m_bUseDecay) grad = m_applyDecay(pmset, grad, tracer);

        VTensor velocity = m_evalMomentumDelta(pmset, grad, tracer);

        VTensor hostVelocity = tracer.createTensor(m_session, grad.shape(), VDataType::float32, -1);

        float* pv = velocity.float_ptr();
        float* ph = hostVelocity.float_ptr();

        CALL_MATH(memcpy_to_host, velocity.device(), ph, pv, velocity.byte_size());
        CALL_MATH(add, -1, pp, pp, ph, ndat);
    }
        break;
    case OptAlgorithm::adagrad:
    {
        if (m_bUseDecay) grad = m_applyDecay(pmset, grad, tracer);

        VTensor delta = m_evalAdaGradDelta(pmset, grad, tracer);

        VTensor hostDelta = tracer.createTensor(m_session, delta.shape(), VDataType::float32, -1);

        float* pd = delta.float_ptr();
        float* ph = hostDelta.float_ptr();

        CALL_MATH(memcpy_to_host, grad.device(), ph, pd, grad.byte_size());
        CALL_MATH(add, -1, pp, pp, ph, ndat);
    }
    break;
    case OptAlgorithm::rmsprop:
    {
        if (m_bUseDecay) grad = m_applyDecay(pmset, grad, tracer);

        VTensor delta = m_evalRMSPropDelta(pmset, grad, tracer);

        VTensor hostDelta = tracer.createTensor(m_session, delta.shape(), VDataType::float32, -1);

        float* pd = delta.float_ptr();
        float* ph = hostDelta.float_ptr();

        CALL_MATH(memcpy_to_host, grad.device(), ph, pd, grad.byte_size());
        CALL_MATH(add, -1, pp, pp, ph, ndat);
    }
    break;
    default:
        VP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

VTensor VOptimizerCore::m_applyDecay(VDict pmset, VTensor grad, VExecTracer tracer) {
    int64 ndat = grad.shape().total_size();

    VTensor param = tracer.createTensor(m_session, (VHTensor)pmset["pm"]);
    VTensor decay_grad = tracer.createTensor(m_session, grad.shape(), grad.type(), grad.device());

    float* pg = grad.float_ptr();
    float* pd = decay_grad.float_ptr();

    if (grad.device() < 0) {
        float* pp = param.float_ptr();
        //VMath::apply_decay_host(pd, pp, pg, ndat, m_l1Decay, m_l2Decay);
        CALL_MATH(apply_decay, -1, pd, pp, pg, ndat, HYPER_FETCH_CPU(m_l1Decay), HYPER_FETCH_CPU(m_l2Decay));
    }
    else {
        int device = grad.device();
        VTensor cudaParam = param.toDevice(device, tracer);
        float* pp = cudaParam.float_ptr();
        //VMath::apply_decay_cuda(pd, pp, pg, ndat, m_l1Decay, m_l2Decay);
        CALL_MATH(apply_decay, grad.device(), pd, pp, pg, ndat, HYPER_FETCH_DEV(device, m_l1Decay), HYPER_FETCH_DEV(device, m_l2Decay));
    }

    return decay_grad;
}

VTensor VOptimizerCore::m_evalAdamDelta(VDict pmset, VTensor grad, VExecTracer tracer) {
    VTensor s = tracer.createTensor(m_session, (VHTensor)pmset["s"]);
    VTensor t = tracer.createTensor(m_session, (VHTensor)pmset["t"]);
    VTensor n = tracer.createTensor(m_session, (VHTensor)pmset["n"]);

    int64 ndat = grad.shape().total_size();

    if (ndat == 0) return VTensor();

    VTensor adam_delta = tracer.createTensor(m_session, grad.shape(), grad.type(), grad.device());

    float* pa = adam_delta.float_ptr();
    float* pg = grad.float_ptr();
    float* ps = s.float_ptr();
    float* pt = t.float_ptr();
    float* pn = n.float_ptr();
    
    int device = grad.device();

    CALL_MATH(eval_adam_delta, device, pa, pg, ps, pt, pn, ndat, HYPER_FETCH_DEV(device, m_ro1), HYPER_FETCH_DEV(device, m_ro2), HYPER_FETCH_DEV(device, m_epsilon));

    return adam_delta;
}

VTensor VOptimizerCore::m_evalMomentumDelta(VDict pmset, VTensor grad, VExecTracer tracer) {
    VTensor velocity = tracer.createTensor(m_session, (VHTensor)pmset["v"]);

    int64 ndat = grad.shape().total_size();
    int device = grad.device();

    if (ndat == 0) return VTensor();

    float* pv = velocity.float_ptr();
    float* pg = grad.float_ptr();

    CALL_MATH(mult_scalar, device, pv, ndat, HYPER_FETCH_DEV(device, m_momentum));
    CALL_MATH(sub_mult_scalar_to, device, pv, pv, pg, ndat, HYPER_FETCH_DEV(device, m_learning_rate));

    return velocity;
}

VTensor VOptimizerCore::m_preprocNesterovParam(VTensor param, VDict pmset, VExecTracer tracer) {
    VTensor velocity = tracer.createTensor(m_session, (VHTensor)pmset["v"]);

    int device = param.device();
    int64 ndat = param.shape().total_size();
    
    if (ndat == 0) return VTensor();

    VTensor new_param = tracer.createTensor(m_session, param.shape(), param.type(), device);

    float* pv = velocity.float_ptr();
    float* pn = new_param.float_ptr();
    float* pp = param.float_ptr();

    CALL_MATH(add_mult_scalar_to, device, pn, pp, pv, ndat, HYPER_FETCH_DEV(device, m_momentum));

    return new_param;
}

VTensor VOptimizerCore::m_evalAdaGradDelta(VDict pmset, VTensor grad, VExecTracer tracer) {
    VTensor grad_acc = tracer.createTensor(m_session, (VHTensor)pmset["r"]);

    int64 ndat = grad.shape().total_size();
    int device = grad.device();

    if (ndat == 0) return VTensor();

    VTensor new_grad = tracer.createTensor(m_session, grad.shape(), grad.type(), device);

    float* pr = grad_acc.float_ptr();
    float* pg = grad.float_ptr();
    float* pn = new_grad.float_ptr();

    CALL_MATH(acc_sqsum, device, pr, pg, ndat);
    CALL_MATH(adagrad_update, device, pn, pg, pr, ndat, HYPER_FETCH_DEV(device, m_learning_rate), HYPER_FETCH_DEV(device, m_sigma));

    return new_grad;
}

VTensor VOptimizerCore::m_evalRMSPropDelta(VDict pmset, VTensor grad, VExecTracer tracer) {
    VTensor grad_acc = tracer.createTensor(m_session, (VHTensor)pmset["r"]);

    int64 ndat = grad.shape().total_size();
    int device = grad.device();

    if (ndat == 0) return VTensor();

    VTensor new_grad = tracer.createTensor(m_session, grad.shape(), grad.type(), device);

    float* pr = grad_acc.float_ptr();
    float* pg = grad.float_ptr();
    float* pn = new_grad.float_ptr();

    CALL_MATH(acc_sqsum_decay, device, pr, pg, ndat, HYPER_FETCH_DEV(device, m_decay));
    CALL_MATH(adagrad_update, device, pn, pg, pr, ndat, HYPER_FETCH_DEV(device, m_learning_rate), HYPER_FETCH_DEV(device, m_sigma));

    return new_grad;
}
