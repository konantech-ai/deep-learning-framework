#define ECO_EXPORTS
#include "../include/eco_api.h"

#include "../objects/tp_nn.h"
#include "../objects/tp_nn_core.h"
#include "../objects/tp_module.h"
#include "../objects/tp_module_core.h"
#include "../objects/tp_loss.h"
#include "../objects/tp_loss_core.h"
#include "../objects/tp_optimizer.h"
#include "../objects/tp_optimizer_core.h"
#include "../objects/tp_parameters.h"
#include "../objects/tp_parameters_core.h"
#include "../objects/tp_tensor.h"
#include "../objects/tp_tensor_core.h"
#include "../objects/tp_scalar.h"
#include "../objects/tp_scalar_core.h"
#include "../objects/tp_function.h"
#include "../objects/tp_function_core.h"
#include "../objects/tp_metric.h"
#include "../objects/tp_metric_core.h"
#include "../objects/tp_audio_file_reader.h"
#include "../objects/tp_audio_file_reader_core.h"

#include "../utils/tp_exception.h"
#include "../utils/tp_eco_exception_info.h"
#include "../utils/tp_json_parser.h"
#include "../utils/tp_stream.h"
#include "../utils/tp_cuda.h"

namespace api = konan::eco;

//======================================================================================

typedef api::NN ANN;
typedef api::Module AModule;
typedef api::Loss ALoss;
typedef api::Optimizer AOptimizer;
typedef api::Parameters AParameters;
typedef api::Scalar AScalar;
typedef api::Tensor ATensor;
typedef api::Function AFunction;
typedef api::Metric AMetric;
typedef api::Account AAccount;
typedef api::Util AUtil;
typedef api::AudioSpectrumReader AAudioSpectrumReader;

typedef api::StringList AStringList;

typedef ENN EAccount;
typedef ENNCore EAccountCore;

ENN toEco(ANN x) { return ENN((ENNCore*)x.get_core()); }
ANN toApi(ENN x) { return ANN((ENNCore*)x.createApiClone()); }

EModule toEco(AModule x) { return EModule((EModuleCore*)x.get_core()); }
AModule toApi(EModule x) { return AModule((EModuleCore*)x.createApiClone()); }

ELoss toEco(ALoss x) { return ELoss((ELossCore*)x.get_core()); }
ALoss toApi(ELoss x) { return ALoss((ELossCore*)x.createApiClone()); }

EOptimizer toEco(AOptimizer x) { return EOptimizer((EOptimizerCore*)x.get_core()); }
AOptimizer toApi(EOptimizer x) { return AOptimizer((EOptimizerCore*)x.createApiClone()); }

EParameters toEco(AParameters x) { return EParameters((EParametersCore*)x.get_core()); }
AParameters toApi(EParameters x) { return AParameters((EParametersCore*)x.createApiClone()); }

EScalar toEco(AScalar x) { return EScalar((EScalarCore*)x.get_core()); }
AScalar toApi(EScalar x) { return AScalar((EScalarCore*)x.createApiClone()); }

ETensor toEco(ATensor x) {
	return ETensor((ETensorCore*)x.get_core());
}
ATensor toApi(ETensor x) {
	return ATensor((ETensorCore*)x.createApiClone());
}

EFunction toEco(AFunction x) { return EFunction((EFunctionCore*)x.get_core()); }
AFunction toApi(EFunction x) { return AFunction((EFunctionCore*)x.createApiClone()); }

EMetric toEco(AMetric x) { return EMetric((EMetricCore*)x.get_core()); }
AMetric toApi(EMetric x) { return AMetric((EMetricCore*)x.createApiClone()); }

EAccount toEco(AAccount x) { return EAccount((EAccountCore*)x.get_core()); }

EAudioFileReader toEco(AAudioSpectrumReader x) { return EAudioFileReader((EAudioFileReaderCore*)x.get_core()); }
AAudioSpectrumReader toApi(EAudioFileReader x) { return AAudioSpectrumReader((EAudioFileReaderCore*)x.createApiClone()); }

EModuleList toEco(api::ModuleList modules) {
	EModuleList list;

	for (auto& it : modules) {
		list.push_back(EModule((EModuleCore*)it.get_core()));
	}

	return list;
}

ETensorDict toEco(api::TensorDict losses) {
	ETensorDict dict;

	for (auto& it : losses) {
		dict[it.first] = ETensor((ETensorCore*)it.second.get_core());
	}

	return dict;
}

ELossDict toEco(api::LossDict losses) {
	ELossDict dict;

	for (auto& it : losses) {
		dict[it.first] = ELoss((ELossCore*)it.second.get_core());
	}

	return dict;
}

api::TensorDict toApi(ETensorDict tensors) {
	api::TensorDict dict;

	for (auto& it : tensors) {
		api::Tensor apiTensor(it.second.createApiClone()); // ((ELossCore*)it.second.get_core());
		dict[it.first] = apiTensor;
	}

	return dict;
}

EMetricDict toEco(api::MetricDict metrics) {
	EMetricDict dict;

	for (auto& it : metrics) {
		dict[it.first] = EMetric((EMetricCore*)it.second.get_core());
	}

	return dict;
}

//======================================================================================

ECO_API const char* konan::eco::getEcoVersion() noexcept {
	return sEcoVersion.c_str();
}

//======================================================================================

api::Object::Object(void* ptr) {
	m_core = ptr;
	if (ptr) ((EcoObjCore*)m_core)->clone_core();
}

ECO_API api::Object::~Object() {
	if (m_core) ((EcoObjCore*)m_core)->destroy();
}

ECO_API int api::Object::get_local_id() {
	return m_core ? ((EcoObjCore*)m_core)->getNth() : -1;
}

ECO_API void api::Object::free(void* core) {
	((EcoObjCore*)core)->destroy();
}

void* api::Object::get_core() {
	return m_core;
}

ECO_API void* api::Object::clone_core() {
	((EcoObjCore*)m_core)->clone_core();
	return m_core;
}

//======================================================================================

api::EcoException::EcoException(void* errInfo, string func, string cinfo) : api::Object(NULL) {
	m_core = errInfo;
}

api::EcoException::EcoException(TpException& ex, string func, string cinfo) : api::Object(NULL) {
	TpEcoExceptionInfo* pInfo = new TpEcoExceptionInfo(ex, func, cinfo);
	m_core = (void*) pInfo;
}

ECO_API api::EcoException::EcoException(const api::EcoException& a) : api::Object(NULL) {
	m_core = a.m_core;
}

ECO_API api::EcoException& api::EcoException::operator=(const EcoException& a) {
	if (&a != this) {
		m_core = a.m_core;
	}

	return *this;
}

ECO_API int api::EcoException::get_error_code() const {
	TpEcoExceptionInfo* pInfo = (TpEcoExceptionInfo*)m_core;
	return pInfo->get_error_code();
}

ECO_API string api::EcoException::get_error_message(int indent) const {
	TpEcoExceptionInfo* pInfo = (TpEcoExceptionInfo*)m_core;
	return pInfo->get_error_message(indent);
}

ECO_API void api::EcoException::setDevelopperMode(bool bForDevelopper) {
	TpEcoExceptionInfo::setDevelopperMode(bForDevelopper);
}

//======================================================================================

ECO_API ANN::NN() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API ANN::NN(void* core) : api::Object(core) {
}

ECO_API ANN::NN(const char* pUrl) : api::Object(NULL) {
	try {
		string url = pUrl;
		ENN nn(url);
		m_core = nn.createApiClone();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ANN::NN(std::string server_url, std::string client_url) : api::Object(NULL) {
	try {
		ENN nn(server_url, client_url);
		m_core = nn.createApiClone();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ANN::NN(const NN& a) : api::Object(NULL) {
	try {
		m_core = ((ENNCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ANN& ANN::operator=(const NN& a) {
	try {
		if (&a != this) {
			((ENNCore*)m_core)->destroy();
			m_core = ((ENNCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}

	return *this;
}

ECO_API ANN::~NN() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((ENNCore*)m_core)->destroy();
}

ECO_API string ANN::get_engine_version() {
	try {
		return toEco(*this).get_engine_version();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API bool ANN::cuda_is_available() const {
	try {
		return toEco(*this).Cuda_isAvailable();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API int ANN::cuda_get_device_count() const {
	try {
		return toEco(*this).Cuda_getDeviceCount();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::srand(int64 random_seed) {
	try {
		toEco(*this).srand(random_seed);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::set_no_grad() {
	try {
		toEco(*this).set_no_grad();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::unset_no_grad() {
	try {
		toEco(*this).unset_no_grad();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::set_no_tracer() {
	try {
		toEco(*this).set_no_tracer();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::unset_no_tracer() {
	try {
		toEco(*this).unset_no_tracer();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::save_module(AModule module, string filename) {
	try {
		toEco(*this).saveModule(toEco(module), filename);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::load_module(string filename) {
	try {
		return toApi(toEco(*this).loadModule(filename));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

/*
ECO_API int ANN::add_forward_callback_handler(TCbForwardCallback* pCbFunc, VDict instInfo, VDict filters) {
	try {
		return toEco(*this).addForwardCallbackHandler(pCbFunc, instInfo, filters);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API int ANN::add_backward_callback_handler(TCbBackwardCallback* pCbFunc, VDict instInfo, VDict filters) {
	try {
		return toEco(*this).addBackwardCallbackHandler(pCbFunc, instInfo, filters);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::remove_callback_handler(int nId) {
	try {
		toEco(*this).removeCallbackHandler(nId);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}
*/

ECO_API string ANN::get_layer_formula(string layerName) {
	try {
		return toEco(*this).GetLayerFormula(layerName);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

/*
ECO_API void ANN::regist_user_def_func(VHFunction hFunction, Function* pUserFunc) {
	try {
		toEco(*this).registUserDefFunc(hFunction, pUserFunc);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}
*/

//ETensor func_cb_forward(VHFunction hFunction, int nInst, ETensorList operands, VDict opArgs);
//ETensor func_cb_backward(VHFunction hFunction, int nInst, ETensor ygrad, int nth, ETensorList operands, VDict opArgs);

ECO_API VDict ANN::get_builtin_names() {
	try {
		return toEco(*this).get_builtin_names();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AStringList ANN::get_builtin_names(string domain) {
	try {
		return toEco(*this).get_builtin_names(domain);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API bool ANN::is_builtin_name(string domain, string sBuiltin) {
	try {
		return toEco(*this).isInBuiltinName(domain, sBuiltin);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API VDict ANN::get_leak_info(bool sessionOnly) {
	try {
		return toEco(*this).getLeakInfo(sessionOnly);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::dump_leak_info(bool sessionOnly) {
	try {
		toEco(*this).dumpLeakInfo(sessionOnly);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

//======================================================================================

ECO_API AModule ANN::Model(string modelName, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Model(modelName, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Macro(string macroName, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Macro(macroName, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::create_user_defined_layer(string name, string formula, VDict paramInfo, VDict kwArgs) {
	try {
		return toApi(toEco(*this).createUserDefinedLayer(name, formula, paramInfo, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API void ANN::regist_macro(string macroName, AModule contents, VDict kwArgs) {
	try {
		toEco(*this).RegistMacro(macroName, toEco(contents), kwArgs);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

//======================================================================================

ECO_API AModule ANN::If(bool cond, AModule trueModule) {
	try {
		if (cond) return trueModule;
		else return toApi(toEco(*this).Pass({}));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::If(bool cond, AModule trueModule, AModule falseModule) {
	try {
		return cond ? trueModule : falseModule;
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

//======================================================================================

ECO_API AModule ANN::Linear(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Linear(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Dense(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Dense(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv1D(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv1D(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv2D(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv2D_Transposed(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D_Transposed(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv2D_Dilated(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D_Dilated(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

/*
ECO_API AModule ANN::Conv2D_Separable(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D_Separable(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv2D_Depthwise_Separable(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D_Depthwise_Separable(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv2D_Pointwise(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D_Pointwise(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv2D_Grouped(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D_Grouped(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Conv2D_Degormable(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Conv2D_Degormable(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}
*/

ECO_API AModule ANN::Deconv(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Deconv(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Max(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Max(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Avg(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Avg(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Rnn(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Rnn(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Lstm(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Lstm(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Gru(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Gru(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Embed(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Embed(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Dropout(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Dropout(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Extract(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Extract(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::MultiHeadAttention(VDict kwArgs) {
	try {
		return toApi(toEco(*this).MultiHeadAttention(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::AddBias(VDict kwArgs) {
	try {
		return toApi(toEco(*this).AddBias(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Flatten(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Flatten(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Reshape(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Reshape(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::GlobalAvg(VDict kwArgs) {
	try {
		return toApi(toEco(*this).GlobalAvg(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::AdaptiveAvg(VDict kwArgs) {
	try {
		return toApi(toEco(*this).AdaptiveAvg(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Transpose(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Transpose(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Layernorm(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Layernorm(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Batchnorm(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Batchnorm(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Upsample(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Upsample(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Concat(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Concat(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Pass(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Pass(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Noise(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Noise(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Random(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Random(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Round(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Round(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::CodeConv(VDict kwArgs) {
	try {
		return toApi(toEco(*this).CodeConv(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::CosineSim(VDict kwArgs) {
	try {
		return toApi(toEco(*this).CosineSim(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::SelectNTop(VDict kwArgs) {
	try {
		return toApi(toEco(*this).SelectNTop(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::SelectNTopArg(VDict kwArgs) {
	try {
		return toApi(toEco(*this).SelectNTopArg(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Activate(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Activate(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::ReLU(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Relu(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Leaky(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Leaky(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Softmax(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Softmax(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Sigmoid(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Sigmoid(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Tanh(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Tanh(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Gelu(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Gelu(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Mish(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Mish(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Swish(VDict kwArgs) {
	try {
		return toApi(toEco(*this).Swish(kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Sequential(ModuleList children, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Sequential(toEco(children), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Parallel(ModuleList children, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Parallel(toEco(children), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Add(ModuleList children, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Add(toEco(children), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Residual(ModuleList children, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Residual(toEco(children), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Pruning(ModuleList children, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Pruning(toEco(children), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Stack(ModuleList children, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Stack(toEco(children), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::SqueezeExcitation(ModuleList children, VDict kwArgs) {
	try {
		return toApi(toEco(*this).SqueezeExcitation(toEco(children), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Formula(string formula, VDict kwArgs) {
	try {
		return toApi(toEco(*this).Formula(formula, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AModule ANN::Formula(VDict kwArgs) {
	try {
		string formula = kwArgs["formula"];
		return toApi(toEco(*this).Formula(formula, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ALoss ANN::MSELoss(VDict kwArgs, string sEstName, string sAnsName) {
	try {
		return toApi(toEco(*this).MSELoss(kwArgs, sEstName, sAnsName));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ALoss ANN::CrossEntropyLoss(VDict kwArgs, string sLogitName, string sLabelName) {
	try {
		return toApi(toEco(*this).CrossEntropyLoss(kwArgs, sLogitName, sLabelName));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ALoss ANN::BinaryCrossEntropyLoss(VDict kwArgs, string sLogitName, string sLabelName) {
	try {
		return toApi(toEco(*this).BinaryCrossEntropyLoss(kwArgs, sLogitName, sLabelName));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ALoss ANN::CrossEntropySigmoidLoss(VDict kwArgs, string sLogitName, string sLabelName) {
	try {
		return toApi(toEco(*this).CrossEntropySigmoidLoss(kwArgs, sLogitName, sLabelName));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ALoss ANN::CrossEntropyPositiveIdxLoss(VDict kwArgs, string sLogitName, string sLabelName) {
	try {
		return toApi(toEco(*this).CrossEntropyPositiveIdxLoss(kwArgs, sLogitName, sLabelName));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ALoss ANN::MultipleLoss(LossDict losses) {
	try {
		return toApi(toEco(*this).MultipleLoss(toEco(losses)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API ALoss ANN::CustomLoss(VDict lossTerms, TensorDict statistics, VDict kwArgs) {
	try {
		return toApi(toEco(*this).CustomLoss(lossTerms, toEco(statistics), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "NN");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "NN");
	}
}

ECO_API AOptimizer ANN::createOptimizer(string name, AParameters parameters, VDict kwArgs) {
	try {
		EParameters params((EParametersCore*)parameters.clone_core());
		return toApi(toEco(*this).createOptimizer(name, toEco(parameters), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API AOptimizer ANN::SGDOptimizer(AParameters parameters, VDict kwArgs) {
	try {
		EParameters params((EParametersCore*)parameters.clone_core());
		return toApi(toEco(*this).SGDOptimizer(toEco(parameters), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API AOptimizer ANN::AdamOptimizer(AParameters parameters, VDict kwArgs) {
	try {
		EParameters params((EParametersCore*)parameters.clone_core());
		return toApi(toEco(*this).AdamOptimizer(toEco(parameters), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API AOptimizer ANN::NesterovOptimizer(AParameters parameters, VDict kwArgs) {
	try {
		EParameters params((EParametersCore*)parameters.clone_core());
		return toApi(toEco(*this).NesterovOptimizer(toEco(parameters), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API AOptimizer ANN::MomentumOptimizer(AParameters parameters, VDict kwArgs) {
	try {
		EParameters params((EParametersCore*)parameters.clone_core());
		return toApi(toEco(*this).MomentumOptimizer(toEco(parameters), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API AOptimizer ANN::AdaGradOptimizer(AParameters parameters, VDict kwArgs) {
	try {
		EParameters params((EParametersCore*)parameters.clone_core());
		return toApi(toEco(*this).AdaGradOptimizer(toEco(parameters), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API AOptimizer ANN::RMSPropOptimizer(AParameters parameters, VDict kwArgs) {
	try {
		EParameters params((EParametersCore*)parameters.clone_core());
		return toApi(toEco(*this).RMSPropOptimizer(toEco(parameters), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API AMetric ANN::FormulaMetric(string sName, string sFormula, VDict kwArgs) {
	try {
		return toApi(toEco(*this).FormulaMetric(sName, sFormula, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}
}

ECO_API AMetric ANN::MultipleMetric(MetricDict metrics, VDict kwArgs) {
	try {
		return toApi(toEco(*this).MultipleMetric(toEco(metrics), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}
}

ECO_API AMetric ANN::CustomMetric(VDict expTerms, TensorDict statistics, VDict kwArgs) {
	try {
		return toApi(toEco(*this).CustomMetric(expTerms, toEco(statistics), kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}
}

ECO_API ATensor ANN::createTensor(VShape shape, VDataType type) {
	try {
		return toApi(ETensor(toEco(*this), shape, type));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::createTensor(VShape shape, VDataType type, void* pData) {
	try {
		return toApi(ETensor(toEco(*this), shape, type, pData));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::createTensor(VShape shape, VDataType type, int device) {
	try {
		return toApi(ETensor(toEco(*this), shape, type, device));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::createTensor(VShape shape, VDataType type, string initMethod) {
	try {
		return toApi(ETensor(toEco(*this), shape, type, initMethod));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API ATensor::Tensor(ANN nn, VShape shape, VDataType dataType, VDataType loadType, FILE* fin) : api::Object(NULL) {
	try {
		ETensor tensor(toEco(nn), shape, dataType, loadType, fin);
		m_core = tensor.createApiClone();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API ATensor ANN::createTensor(VShape shape, VDataType type, VList values) {
	try {
		return toApi(ETensor(toEco(*this), shape, type, values));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API ATensor ANN::createTensor(Tensor src, VShape shape) {
	try {
		ETensor tensor(toEco(src), shape);
		m_core = tensor.createApiClone();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

/*
ECO_API ATensor ANN::createTensor(VHTensor hTensor, bool needToClose, bool needToUpload) {
	try {
		ETensor tensor(toEco(nn), hTensor, needToClose, needToUpload);
		m_core = tensor.createApiClone();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API AAudioSpectrumReader ANN::createAudioSpectrumReader(VDict args) {
	try {
		return toApi(EAudioFileReader(toEco(*this), args));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "AudioSpectrumReader");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "AudioSpectrumReader");
	}
}

ECO_API ATensor ANN::arange(int64 from) {
	try {
		return toApi(ETensor::arange(toEco(*this), from));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::arange(int64 from, int64 to) {
	try {
		return toApi(ETensor::arange(toEco(*this), from, to));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API AAccount ANN::open_account() {
	try {
		AAccount account(clone_core());
		return account;
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

//======================================================================================

ECO_API AModule::Module() : Object(NULL) {
}

ECO_API AModule::Module(void* core) : Object(core) {
}

ECO_API AModule::Module(const Module& a) : api::Object(NULL) {
	try {
		m_core = ((EModuleCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API int AModule::get_engine_id() {
	return toEco(*this).nn().getEngineObjId(toEco(*this).cloneCore());
}

ECO_API AModule& AModule::operator=(const Module& a) {
	try {
		if (&a != this) {
			((EModuleCore*)m_core)->destroy();
			m_core = ((EModuleCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}

	return *this;
}

ECO_API AModule::~Module() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EModuleCore*)m_core)->destroy();
}

ECO_API ANN AModule::get_nn() {
	try {
		return toApi(toEco(*this).nn());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API string AModule::get_name() const {
	try {
		return toEco(*this).getName();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API AModule::ModuleType AModule::get_type() const {
	try {
		return (ModuleType)toEco(*this).getType();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API VShape AModule::get_xshape() const {
	try {
		return toEco(*this).getInShape();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API VShape AModule::get_yshape() const {
	try {
		return toEco(*this).getOutShape();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API AModule AModule::expand(VShape shape, VDict kwArgs) {
	try {
		return toApi(toEco(*this).expand(shape, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

/*
ECO_API AModule AModule::expand_macro(VShape shape, VDict kwArgs) {
	try {
		return toApi(toEco(*this).expandMacro(shape, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}
*/

ECO_API AModule AModule::to(std::string device) {
	try {
		return toApi(toEco(*this).to(device));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API std::string AModule::__str__() const {
	try {
		return toEco(*this).__str__();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::desc(string* p_name, string* p_builtin, VDict* p_option, VShape* p_in_shape, VShape* p_out_shape, int64* p_pmsize) const {
	try {
		toEco(*this).desc(p_name, p_builtin, p_option, p_in_shape, p_out_shape, p_pmsize);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::train() {
	try {
		toEco(*this).train();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::eval() {
	try {
		toEco(*this).eval();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::append_child(AModule child) {
	try {
		toEco(*this).appendChild(toEco(child));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API ATensor AModule::__call__(ATensor x) {
	try {
		return toApi(toEco(*this).__call__(toEco(x)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API api::TensorDict AModule::__call__(api::TensorDict xs) {
	try {
		return toApi(toEco(*this).__call__(toEco(xs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API ATensor AModule::predict(ATensor x) {
	try {
		return toApi(toEco(*this).predict(toEco(x)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API api::TensorDict AModule::predict(api::TensorDict xs) {
	try {
		return toApi(toEco(*this).predict(toEco(xs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API AParameters AModule::parameters() {
	try {
		return toApi(toEco(*this).parameters());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

/*
ECO_API VDict AModule::state_dict() const {
	try {
		return toEco(*this).state_dict();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}
*/

ECO_API AModule AModule::create_clone() {
	try {
		return toApi(toEco(*this).createClone());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API AModule AModule::fetch_child(string name) const {
	try {
		return toApi(toEco(*this).fetch_child(name));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API AModule AModule::seek_layer(string name) const {
	try {
		return toApi(toEco(*this).seekLayer(name));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API AModule AModule::nth_child(int nth) const {
	try {
		return toApi(toEco(*this).nthChild(nth));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::init_parameters() {
	try {
		toEco(*this).init_parameters();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

/*
ECO_API bool AModule::load_parameters(string filePath, string mode) {
	try {
		return toEco(*this).loadParameters(filePath, mode);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API bool AModule::load_parameters(string root, string filePath, string mode) {
	try {
		return toEco(*this).loadParameters(root+"/"+filePath, mode);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}
*/

ECO_API void AModule::load_cfg_config(string cfg_path, string weight_path) {
	try {
		toEco(*this).load_cfg_config(cfg_path, weight_path);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::save(std::string path) const {
	try {
		toEco(*this).saveParameters(path);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::save(std::string root, std::string path) const {
	try {
		toEco(*this).saveParameters(root, path);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

//int add_forward_callback_handler(TCbForwardCallback* pCbFunc, VDict instInfo, VDict filters);
//int add_backward_callback_handler(TCbBackwardCallback* pCbFunc, VDict instInfo, VDict filters);

//void remove_callback_handler(int nId);

ECO_API void AModule::upload_data_index(VList dataIdx) {
	try {
		toEco(*this).upload_data_index(dataIdx);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

ECO_API void AModule::set_paramater(TensorDict paramTensors, string mode) {
	try {
		toEco(*this).setParamater(toEco(paramTensors), mode);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Module");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Module");
	}
}

//======================================================================================

ECO_API ALoss::Loss() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API ALoss::Loss(void* core) : api::Object(core) {
}

ECO_API ALoss::Loss(const Loss& a) : api::Object(NULL) {
	try {
		m_core = ((ELossCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}
}

ECO_API int ALoss::get_engine_id() {
	return toEco(*this).nn().getEngineObjId(toEco(*this).cloneCore());
}

ECO_API ALoss& ALoss::operator=(const Loss& a) {
	try {
		if (&a != this) {
			((ELossCore*)m_core)->destroy();
			m_core = ((ELossCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}

	return *this;
}

ECO_API ALoss::~Loss() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((ELossCore*)m_core)->destroy();
}

ECO_API ATensor ALoss::__call__(Tensor pred, Tensor y, bool download_all) {
	try {
		return toApi(toEco(*this).__call__(toEco(pred), toEco(y), download_all));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}
}

ECO_API api::TensorDict ALoss::__call__(TensorDict preds, TensorDict ys, bool download_all) {
	try {
		return toApi(toEco(*this).__call__(toEco(preds), toEco(ys), download_all));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}
}

ECO_API api::Tensor ALoss::eval_accuracy(Tensor pred, Tensor y, bool download_all) {
	try {
		return toApi(toEco(*this).eval_accuracy(toEco(pred), toEco(y), download_all));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}
}

ECO_API api::TensorDict ALoss::eval_accuracy(TensorDict preds, TensorDict ys, bool download_all) {
	try {
		return toApi(toEco(*this).eval_accuracy(toEco(preds), toEco(ys), download_all));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}
}

ECO_API api::TensorDict ALoss::evaluate(TensorDict preds, TensorDict ys, bool download_all) {
	try {
		return toApi(toEco(*this).evaluate(toEco(preds), toEco(ys), download_all));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}
}

ECO_API void ALoss::backward() {
	try {
		toEco(*this).backward();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Loss");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Loss");
	}
}

//======================================================================================

ECO_API AOptimizer::Optimizer() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API AOptimizer::Optimizer(void* core) : api::Object(core) {
}

ECO_API api::Optimizer::Optimizer(const Optimizer& a) : api::Object(NULL) {
	try {
		m_core = ((EOptimizerCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API int AOptimizer::get_engine_id() {
	return toEco(*this).nn().getEngineObjId(toEco(*this).cloneCore());
}

ECO_API api::Optimizer& api::Optimizer::operator=(const Optimizer& a) {
	try {
		if (&a != this) {
			((EOptimizerCore*)m_core)->destroy();
			m_core = ((EOptimizerCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}

	return *this;
}

ECO_API api::Optimizer::~Optimizer() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EOptimizerCore*)m_core)->destroy();
}

ECO_API void api::Optimizer::set_option(VDict kwArgs) {
	try {
		toEco(*this).setOption(kwArgs);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API void api::Optimizer::zero_grad() {
	try {
		toEco(*this).zero_grad();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

ECO_API void api::Optimizer::step() {
	try {
		toEco(*this).step();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Optimizer");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Optimizer");
	}
}

//======================================================================================

ECO_API AParameters::Parameters() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API AParameters::Parameters(void* core) : api::Object(core) {
}

ECO_API AParameters::Parameters(const Parameters& a) : api::Object(NULL) {
	try {
		m_core = ((EParametersCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
}

ECO_API AParameters& AParameters::operator=(const Parameters& a) {
	try {
		if (&a != this) {
			((EParametersCore*)m_core)->destroy();
			m_core = ((EParametersCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}

	return *this;
}

ECO_API AParameters::~Parameters() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EParametersCore*)m_core)->destroy();
}

/*
ECO_API void AParameters::add(AParameters params) {
	try {
		toEco(*this).add(toEco(params));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
}
*/

ECO_API int AParameters::get_engine_id() {
	return toEco(*this).nn().getEngineObjId(toEco(*this).cloneCore());
}

ECO_API void AParameters::zero_grad() {
	try {
		toEco(*this).zero_grad();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
}

ECO_API VList AParameters::weightList(api::TensorDict& tensors) {
	try {
		ETensorDict dict;
		VList terms = toEco(*this).weightList(dict);
		tensors = toApi(dict);
		return terms;
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
	return {};
}

ECO_API VList AParameters::gradientList(api::TensorDict& tensors) {
	try {
		ETensorDict dict;
		VList terms = toEco(*this).gradientList(dict);
		tensors = toApi(dict);
		return terms;
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
	return {};
}

ECO_API api::TensorDict AParameters::weightDict() {
	try {
		return toApi(toEco(*this).weightDict());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
	return {};
}

ECO_API api::TensorDict AParameters::gradientDict() {
	try {
		return toApi(toEco(*this).gradientDict());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
	return {};
}

ECO_API void AParameters::initWeights() {
	try {
		toEco(*this).initWeights();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Parameters");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Parameters");
	}
}

//======================================================================================

ECO_API ATensor::Tensor() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API ATensor::Tensor(void* core) : api::Object(core) {
}

ECO_API ATensor::Tensor(const Tensor& a) : api::Object(NULL) {
	try {
		m_core = ((ETensorCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API int ATensor::get_engine_id() {
	return toEco(*this).nn().getEngineObjId(toEco(*this).cloneCore());
}

ECO_API ATensor& ATensor::operator=(const Tensor& a) {
	try {
		if (&a != this) {
			((ETensorCore*)m_core)->destroy();
			m_core = ((ETensorCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}

	return *this;
}

ECO_API ATensor::~Tensor() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((ETensorCore*)m_core)->destroy();
}

/*
ECO_API ATensor ATensor::load(NN nn, FILE* fin) {
	try {
		return toApi(ETensor::load(toEco(nn), fin));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API ATensor::operator int() const {
	try {
		return (int)toEco(*this);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor::operator int64() const {
	try {
		// 주의: VHTensor 형변환자와 혼동되므로 int64 형변환자를 직접 정의해 사용하면 안된다.
		//return (int64)toEco(*this);
		return toEco(*this).to_int64();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor::operator float() const {
	try {
		return (float)toEco(*this);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API int64 ATensor::len() const {
	try {
		return toEco(*this).len();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API int64 ATensor::size() const {
	try {
		return toEco(*this).size();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API int ATensor::device() const {
	try {
		return toEco(*this).device();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API int64 ATensor::byte_size() const {
	try {
		return toEco(*this).byteSize();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API VShape ATensor::shape() const {
	try {
		return toEco(*this).shape();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API VDataType ATensor::type() const {
	try {
		return toEco(*this).type();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API string ATensor::type_desc() const {
	try {
		return toEco(*this).type_desc();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API VDataType ATensor::name_to_type(string type_name) {
	try {
		return TpUtils::to_data_type(type_name);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API bool ATensor::has_no_data() {
	try {
		return toEco(*this).hasNoData();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::alloc_data() {
	try {
		toEco(*this).allocData();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::download_data() {
	try {
		toEco(*this).downloadData();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API void ATensor::download_data(ANN nn) {
	try {
		toEco(*this).downloadData(toEco(nn));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API void ATensor::copy_data(VShape shape, VDataType type, void* pData) {
	try {
		toEco(*this).copyData(shape, type, pData);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API api::Scalar ATensor::item() const {
	try {
		return toApi(toEco(*this).item());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::backward() {
	try {
		toEco(*this).backward();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::backward_with_gradient(ATensor grad) {
	try {
		toEco(*this).backwardWithGradient(toEco(grad));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::dump(std::string title, bool full) {
	try {
		toEco(*this).dump(title, full);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::copy_partial_data(Tensor src) {
	try {
		toEco(*this).copyPartialData(toEco(src));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::copy_row_from(int64 n, ATensor src, int64 nDataIdx) {
	try {
		toEco(*this).copyRowFrom(n, toEco(src), nDataIdx);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::shift_timestep_to_right(Tensor src, int64 steps) {
	try {
		toEco(*this).shift_timestep_to_right(toEco(src), steps);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::ones(VShape shape) {
	try {
		return toApi(ETensor::ones(toEco(*this), shape));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::zeros(VShape shape) {
	try {
		return toApi(ETensor::zeros(toEco(*this), shape));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::rand_uniform(VShape shape, float min, float max) {
	try {
		return toApi(ETensor::rand_uniform(toEco(*this), shape, min, max));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::rand_normal(VShape shape, float avg, float std) {
	try {
		return toApi(ETensor::rand_normal(toEco(*this), shape, avg, std));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::rand_onehot(VShape shape) {
	try {
		return toApi(ETensor::rand_onehot(toEco(*this), shape));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ANN::arange(VValue arg1, VValue arg2, VValue arg3, VDict kwArgs) {
	try {
		return toApi(ETensor::arange(toEco(*this), arg1, arg2, arg3, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API ATensor ATensor::matmul(ATensor x1, ATensor x2) {
	try {
		return toApi(ETensor::matmul(toEco(x1), toEco(x2)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::hconcat(ATensor x1, ATensor x2) {
	try {
		return toApi(ETensor::hconcat(toEco(x1), toEco(x2)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::vstack(ATensor x1, ATensor x2) {
	try {
		return toApi(ETensor::vstack(toEco(x1), toEco(x2)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API void ATensor::vstack_on(ATensor x1, ATensor x2) {
	try {
		toEco(*this).vstack_on(toEco(x1), toEco(x2));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API ATensor ATensor::tile(ATensor x, int64 rep) {
	try {
		return toApi(ETensor::tile(toEco(x), rep));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::repeat(ATensor x, int64 rep) {
	try {
		return toApi(ETensor::repeat(toEco(x), rep));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

*/

ECO_API ATensor ATensor::resize(VShape shape) {
	try {
		TpCuda cuda;
		return toApi(cuda.resize(toEco(*this), shape));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::resize_on(ATensor x) {
	try {
		TpCuda cuda;
		cuda.resize_on(toEco(*this), toEco(x));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::transpose_on(int64 axis1, int64 axis2) {
	try {
		TpCuda cuda;
		cuda.transpose_on(toEco(*this), axis1, axis2);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::load_jpeg_pixels(string filepath, bool chn_last, bool transpose, int code, float mix) {
	try {
		TpCuda cuda;
		cuda.load_jpeg_pixels(toEco(*this), filepath, chn_last, transpose, code, mix);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::shuffle() {
	try {
		toEco(*this).shuffle();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API void ATensor::rand_uniform_on(float min, float max) {
	try {
		toEco(*this).rand_uniform_on(min, max);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API void ATensor::fetch_idx_rows(ATensor src, int64* pnMap, int64 size) {
	try {
		toEco(*this).fetchIdxRows(toEco(src), pnMap, size);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::fetch_idx_rows(ATensor src, int* pnMap, int64 size) {
	try {
		toEco(*this).fetchIdxRows(toEco(src), pnMap, size);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::copy_into_row(int64 nthRow, Tensor src) {
	try {
		toEco(*this).copy_into_row(nthRow, toEco(src));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::to_device(int device) {
	try {
		return toApi(toEco(*this).toDevice(device));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API string ATensor::get_dump_str(string title, bool full) {
	try {
		return toEco(*this).get_dump_str(title, full);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::upload() {
	try {
		toEco(*this).upload();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API void ATensor::free_dump_str(char* pData) {
	try {
		toEco(*this).free_dump_str(pData);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::dump_arr_feat(int nth, string title) {
	try {
		toEco(*this).dump_arr_feat(nth, title);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API void ATensor::reset() {
	try {
		toEco(*this).reset();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

//ECO_API void save(TpStreamOut* fout);

//ECO_API static ETensor load(NN nn, TpStreamIn* fin);
//ECO_API static ETensor load(NN nn, FILE* fin);

/*
ECO_API VIntList ATensor::find_element(VValue element) {
	try {
		return toEco(*this).findElement(element);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API void ATensor::set_element(int64 pos, VValue value) {
	try {
		toEco(*this).setElement(pos, value);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::set_element(VList pos, VValue value) {
	try {
		toEco(*this).setElement(pos, value);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void ATensor::set_slice(VList index, VDataType type, int64 datSize, void* value_ptr) {
	try {
		toEco(*this).setSlice(index, type, datSize, value_ptr);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API VValue ATensor::get_element(int64 pos) {
	try {
		return toEco(*this).getElement(pos);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API VValue ATensor::get_element(VList pos) {
	try {
		return toEco(*this).getElement(pos);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::get_slice(VList sliceIndex) {
	try {
		return toApi(toEco(*this).getSlice(sliceIndex));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API void ATensor::set_debug_name(string name) {
	try {
		toEco(*this).setDebugName(name);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API ATensor ATensor::sum() {
	try {
		return toApi(toEco(*this).sum());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::argmax(int64 axis) {
	try {
		return toApi(toEco(*this).argmax(axis));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::mean() {
	try {
		return toApi(toEco(*this).mean());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::abs() {
	try {
		return toApi(toEco(*this).abs());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::sigmoid() {
	try {
		return toApi(toEco(*this).sigmoid());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::square() {
	try {
		return toApi(toEco(*this).square());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::max(ATensor rhs) {
	try {
		return toApi(toEco(*this).max(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::transpose() {
	try {
		return toApi(toEco(*this).transpose());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API ATensor ATensor::set_type(VDataType type, string option) {
	try {
		return toApi(toEco(*this).set_type(type, option));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

ECO_API ATensor ATensor::to_type(VDataType type, string option) {
	try {
		return toApi(toEco(*this).to_type(type, option));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::to_type(string type_name, string option) {
	try {
		VDataType type = TpUtils::to_data_type(type_name);
		return toApi(toEco(*this).to_type(type, option));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::reshape(VShape shape) {
	try {
		return toApi(toEco(*this).reshape(shape));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator ==(ATensor other) {
	try {
		return toApi(toEco(*this) == toEco(other));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator ==(int value) {
	try {
		return toApi(toEco(*this).operator ==(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator ==(int64 value) {
	try {
		return toApi(toEco(*this).operator ==(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator ==(float value) {
	try {
		return toApi(toEco(*this).operator ==(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator !=(ATensor other) {
	try {
		return toApi(toEco(*this) != toEco(other));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator !=(int value) {
	try {
		return toApi(toEco(*this).operator !=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator !=(int64 value) {
	try {
		return toApi(toEco(*this).operator !=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator !=(float value) {
	try {
		return toApi(toEco(*this).operator !=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >(ATensor other) {
	try {
		return toApi(toEco(*this) > toEco(other));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >(int value) {
	try {
		return toApi(toEco(*this).operator >(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >(int64 value) {
	try {
		return toApi(toEco(*this).operator >(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >(float value) {
	try {
		return toApi(toEco(*this).operator >(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >=(ATensor other) {
	try {
		return toApi(toEco(*this) >= toEco(other));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >=(int value) {
	try {
		return toApi(toEco(*this).operator >=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >=(int64 value) {
	try {
		return toApi(toEco(*this).operator >=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator >=(float value) {
	try {
		return toApi(toEco(*this).operator >=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <(ATensor other) {
	try {
		return toApi(toEco(*this) < toEco(other));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <(int value) {
	try {
		return toApi(toEco(*this).operator <(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <(int64 value) {
	try {
		return toApi(toEco(*this).operator <(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <(float value) {
	try {
		return toApi(toEco(*this).operator <(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <=(ATensor other) {
	try {
		return toApi(toEco(*this) <= toEco(other));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <=(int value) {
	try {
		return toApi(toEco(*this).operator <=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <=(int64 value) {
	try {
		return toApi(toEco(*this).operator <=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator <=(float value) {
	try {
		return toApi(toEco(*this).operator <=(value));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator &&(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator &&(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator ||(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator ||(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator &(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator &(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator |(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator |(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator *(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator *(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator *(float rhs) {
	try {
		return toApi(toEco(*this).operator *(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator *(int rhs) {
	try {
		return toApi(toEco(*this).operator *(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator *(int64 rhs) {
	try {
		return toApi(toEco(*this).operator *(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator /(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator /(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator /(float rhs) {
	try {
		return toApi(toEco(*this).operator /(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator /(int rhs) {
	try {
		return toApi(toEco(*this).operator /(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator /(int64 rhs) {
	try {
		return toApi(toEco(*this).operator /(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator +(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator +(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator +(float rhs) {
	try {
		return toApi(toEco(*this).operator +(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator +(int rhs) {
	try {
		return toApi(toEco(*this).operator +(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator +(int64 rhs) {
	try {
		return toApi(toEco(*this).operator +(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator -(ATensor rhs) {
	try {
		return toApi(toEco(*this).operator -(toEco(rhs)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator -(float rhs) {
	try {
		return toApi(toEco(*this).operator -(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator -(int rhs) {
	try {
		return toApi(toEco(*this).operator -(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator -(int64 rhs) {
	try {
		return toApi(toEco(*this).operator -(rhs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator[](int64 index) {
	try {
		return toApi(toEco(*this)[index]);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::operator[](ATensor index) {
	try {
		return toApi(toEco(*this)[toEco(index)]);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API void* ATensor::void_ptr() {
	try {
		return toEco(*this).void_ptr();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API int* ATensor::int_ptr() {
	try {
		return toEco(*this).int_ptr();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API int64* ATensor::int64_ptr() {
	try {
		return toEco(*this).int64_ptr();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API float* ATensor::float_ptr() {
	try {
		return toEco(*this).float_ptr();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API unsigned char* ATensor::uchar_ptr() {
	try {
		return toEco(*this).uchar_ptr();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API unsigned char* ATensor::bool_ptr() {
	try {
		return toEco(*this).bool_ptr();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

/*
ECO_API ATensor ATensor::sin() {
	try {
		return toApi(toEco(*this).sin());
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::evaluate(string expression, VList args) {
	try {
		return toApi(ETensor::evaluate(expression, args));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::linspace(float from, float to, int count, VDict kwArgs) {
	try {
		return toApi(ETensor::linspace(from, to, count, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}

ECO_API ATensor ATensor::randh(VShape shape, VDict kwArgs) {
	try {
		return toApi(ETensor::randh(shape, kwArgs));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Tensor");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Tensor");
	}
}
*/

//======================================================================================

ECO_API AFunction::Function() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API AFunction::Function(void* core) : api::Object(core) {
}

ECO_API AFunction::Function(const Function& a) : api::Object(NULL) {
	try {
		m_core = ((EFunctionCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Function");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Function");
	}
}

ECO_API int AFunction::get_engine_id() {
	return toEco(*this).nn().getEngineObjId(toEco(*this).cloneCore());
}

ECO_API AFunction& AFunction::operator=(const Function& a) {
	try {
		if (&a != this) {
			((EFunctionCore*)m_core)->destroy();
			m_core = ((EFunctionCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Function");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Function");
	}

	return *this;
}

ECO_API AFunction::~Function() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EFunctionCore*)m_core)->destroy();
}

ECO_API AFunction::Function(ANN nn, string sBuiltin, string sName, VDict kwArgs) : api::Object(NULL) {
	EFunction function(toEco(nn), sBuiltin, sName, kwArgs);
	m_core = function.createApiClone();
}

ECO_API string AFunction::get_inst_name() {
	try {
		return toEco(*this).getInstName();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Function");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Function");
	}
}

/*
ECO_API void registUserDefinedFunction(Function* pInst);

virtual Tensor forward(int nInst, Tensor x, VDict opArgs);
virtual Tensor forward(int nInst, TensorList operands, VDict opArgs);

virtual Tensor backward(int nInst, Tensor ygrad, Tensor x, VDict opArgs);
virtual Tensor backward(int nInst, Tensor ygrad, int nth, ETensorList operands, VDict opArgs);
*/

//======================================================================================

ECO_API AMetric::Metric() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API AMetric::Metric(void* core) : api::Object(core) {
}

ECO_API AMetric::Metric(const Metric& a) : api::Object(NULL) {
	try {
		m_core = ((EMetricCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}
}

ECO_API int AMetric::get_engine_id() {
	return toEco(*this).nn().getEngineObjId(toEco(*this).cloneCore());
}

ECO_API AMetric& AMetric::operator=(const Metric& a) {
	try {
		if (&a != this) {
			((EMetricCore*)m_core)->destroy();
			m_core = ((EMetricCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}

	return *this;
}

ECO_API AMetric::~Metric() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EMetricCore*)m_core)->destroy();
}

ECO_API ATensor AMetric::__call__(Tensor pred) {
	try {
		return toApi(toEco(*this).__call__(toEco(pred)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}
}

ECO_API api::TensorDict AMetric::__call__(TensorDict preds) {
	try {
		return toApi(toEco(*this).__call__(toEco(preds)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}
}

ECO_API api::TensorDict AMetric::evaluate(TensorDict preds) {
	try {
		return toApi(toEco(*this).evaluate(toEco(preds)));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Metric");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Metric");
	}
}

//======================================================================================

ECO_API AScalar::Scalar() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API AScalar::Scalar(void* core) : api::Object(core) {
}

ECO_API api::Scalar::Scalar(const Scalar& a) : api::Object(NULL) {
	try {
		m_core = ((EScalarCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Scalar");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Scalar");
	}
}

ECO_API api::Scalar& api::Scalar::operator=(const Scalar& a) {
	try {
		if (&a != this) {
			((EScalarCore*)m_core)->destroy();
			m_core = ((EScalarCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Scalar");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Scalar");
	}

	return *this;
}

ECO_API api::Scalar::~Scalar() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EScalarCore*)m_core)->destroy();
}

/*
ECO_API api::Scalar::Scalar(ANN nn, ATensor tensor, float value) : api::Object(NULL) {
	EScalar scalar(toEco(nn), toEco(tensor), value);
	m_core = scalar.createApiClone();
}
*/

ECO_API api::Scalar::operator float() const {
	return (float)toEco(*this);
}

//======================================================================================

ECO_API AAccount::Account() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API AAccount::Account(void* core) : api::Object(core) {
}

ECO_API AAccount::Account(const Account& a) : api::Object(NULL) {
	try {
		m_core = ((EAccountCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API AAccount& api::Account::operator=(const Account& a) {
	try {
		if (&a != this) {
			((EAccountCore*)m_core)->destroy();
			m_core = ((EAccountCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}

	return *this;
}

ECO_API AAccount::~Account() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EAccountCore*)m_core)->destroy();
}

ECO_API void AAccount::login(string username, string password) {
	try {
		toEco(*this).login(username, password);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::logout() {
	try {
		toEco(*this).logout();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::registrate(string username, string password, string email) {
	try {
		toEco(*this).registrate(username, password, email);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API VList AAccount::get_user_list() {
	try {
		return toEco(*this).getUserList();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API VDict AAccount::get_user_info(string username) {
	try {
		return toEco(*this).getUserInfo(username);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::set_user_info(VDict userInfo, string username) {
	try {
		toEco(*this).setUserInfo(userInfo, username);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

/*
ECO_API void AAccount::close_account() {
	try {
		toEco(*this).closeAccount();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}
*/

ECO_API void AAccount::remove_user(string username) {
	try {
		toEco(*this).removeUser(username);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API VList AAccount::get_roles() {
	try {
		return toEco(*this).getRoles();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API VList AAccount::get_user_roles(string username) {
	try {
		return toEco(*this).getUserRoles(username);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API VList AAccount::get_role_permissions(string rolename) {
	try {
		return toEco(*this).getRolePermissions(rolename);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API VList AAccount::get_user_permissions(string username) {
	try {
		return toEco(*this).getUserPermissions(username);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::add_role(string rolename) {
	try {
		toEco(*this).addRole(rolename);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::rem_role(string rolename, bool force) {
	try {
		toEco(*this).remRole(rolename, force);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::add_user_role(string username, string rolename) {
	try {
		toEco(*this).addUserRole(username, rolename);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::rem_user_role(string username, string rolename) {
	try {
		toEco(*this).remUserRole(username, rolename);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::add_role_permission(string rolename, string permission) {
	try {
		toEco(*this).addRolePermission(rolename, permission);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::rem_role_permission(string rolename, string permission) {
	try {
		toEco(*this).remRolePermission(rolename, permission);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API void AAccount::regist_model(AModule model, string name, string desc, bool is_public) {
	try {
		toEco(*this).registModel(toEco(model), name, desc, is_public);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API VList AAccount::get_model_list() {
	try {
		return toEco(*this).getModelList();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API AModule AAccount::fetch_model(int mid) {
	try {
		return toApi(toEco(*this).fetchModel(mid));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

ECO_API AModule AAccount::fetch_model(string name) {
	try {
		return toApi(toEco(*this).fetchModel(name));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Account");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Account");
	}
}

//======================================================================================

ECO_API VValue AUtil::parse_json_file(string filePath) {
	try {
		JsonParser parser;
		return parser.parse_file(filePath);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Util");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Util");
	}
}

ECO_API VList AUtil::parse_jsonl_file(string filePath) {
	try {
		JsonParser parser;
		return parser.parse_jsonl_file(filePath);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Util");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Util");
	}
}

ECO_API VStrList AUtil::read_file_lines(string filePath) {
	try {
		return TpUtils::read_file_lines(filePath);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Util");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Util");
	}
}

ECO_API VValue AUtil::load_data(ANN nn, string filePath) {
	try {
		TpStreamIn fin(toEco(nn), filePath, true);
		return fin.load_value();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Util");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Util");
	}
}

ECO_API void AUtil::save_data(ANN nn, VValue value, string filePath) {
	try {
		TpStreamOut fout(toEco(nn), filePath);
		return fout.save_value(value);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "Util");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "Util");
	}
}

//======================================================================================

ECO_API AAudioSpectrumReader::AudioSpectrumReader() : api::Object(NULL) {
	m_core = NULL;
}

ECO_API AAudioSpectrumReader::AudioSpectrumReader(void* core) : api::Object(core) {
}

ECO_API AAudioSpectrumReader::AudioSpectrumReader(const AudioSpectrumReader& a) : api::Object(NULL) {
	try {
		m_core = ((EAudioFileReaderCore*)a.m_core)->clone_core();
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "AudioSpectrumReader");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "AudioSpectrumReader");
	}
}

ECO_API AAudioSpectrumReader& api::AudioSpectrumReader::operator=(const AudioSpectrumReader& a) {
	try {
		if (&a != this) {
			((EAudioFileReaderCore*)m_core)->destroy();
			m_core = ((EAudioFileReaderCore*)a.m_core)->clone_core();
		}
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "AudioSpectrumReader");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "AudioSpectrumReader");
	}

	return *this;
}

ECO_API AAudioSpectrumReader::~AudioSpectrumReader() {
	// 기반 클래스인 Object 파괴자에서 파괴하므로 여기서 파괴하면 중복 파괴로 메모리 접근 문제 발생
	//((EAudioFileReaderCore*)m_core)->destroy();
}

ECO_API bool AAudioSpectrumReader::add_file(string filePath) {
	try {
		return toEco(*this).addFile(filePath);
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "AudioSpectrumReader");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "AudioSpectrumReader");
	}
}

ECO_API ATensor AAudioSpectrumReader::get_fft_spectrums(bool ment) {
	try {
		return toApi(toEco(*this).get_fft_spectrums(ment));
	}
	catch (TpException ex) {
		throw EcoException(ex, __func__, "AudioSpectrumReader");
	}
	catch (...) {
		throw EcoException(NULL, __func__, "AudioSpectrumReader");
	}
}

//======================================================================================
