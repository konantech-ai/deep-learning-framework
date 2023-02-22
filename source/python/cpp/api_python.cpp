#include <eco_api.h>
#include "data_handler.h"
#include "exception_handler.h"
#include "inner_proc.h"
#include "api_python.h"

konan::eco::TensorDict m_PyDictToTensorDict(PyObject* py_dict) {
	VDict vdict;
	konan::eco::TensorDict tdict;

	kpy::ConvertPyDictToVDict(py_dict, &vdict);

	for (auto& it : vdict) {
		konan::eco::Tensor tensor((void*)(int64)it.second);
		tdict[it.first] = tensor;
	}

	return tdict;
}
////////////////////////////////////////////////////////////////
// Session and Device
////////////////////////////////////////////////////////////////

void* OpenSession(PyObject* server_url_str, PyObject* client_url_str)
{
	try {
		std::string server_url = PyUnicode_AsUTF8(server_url_str);
		std::string client_url = PyUnicode_AsUTF8(client_url_str);
		void* session_ptr = konan::eco::NN(server_url, client_url).clone_core();
		return session_ptr;
	}
	catch (...) {		
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool CloseSession(void* session_ptr)
{
	try {
		if (session_ptr == NULL)
			return false;

		konan::eco::Object::free(session_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void SrandTime(void)
{
	try {
		std::srand((unsigned int)std::time(NULL));
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void SrandSeed(void* session_ptr, PyObject* seed_int)
{
	try {
		int seed = PyLong_AsLong(seed_int);
		konan::eco::NN(session_ptr).srand((int64)seed);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

bool IsCUDAAvailable(void* session_ptr)
{
	try {
		return konan::eco::NN(session_ptr).cuda_is_available();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

int64 GetCUDADeviceCount(void* session_ptr)
{
	try {
		return konan::eco::NN(session_ptr).cuda_get_device_count();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return 0;
}

void SetNoGrad(void* session_ptr)
{
	try {
		konan::eco::NN(session_ptr).set_no_grad();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void UnsetNoGrad(void* session_ptr)
{
	try {
		konan::eco::NN(session_ptr).unset_no_grad();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

////////////////////////////////////////////////////////////////
// Module
////////////////////////////////////////////////////////////////

void* CreateModule(void* session_ptr, PyObject* module_name_str, PyObject* args_dict)
{
	try {		
		// Get a name of target module
		std::string str = PyUnicode_AsUTF8(module_name_str);
		//std::string str = TpUtils::seekDict(get_args, "module_name_str", "not defined");

		// Convert to lower case
		for (int i=0; i<str.size(); ++i)
			str[i] = tolower(str[i]);

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		// Declare an object of Module
		void* ptr = NULL;

		// Create a module
		// Refer to ms_builtinLayer, in the "konanai_engine/src/api_objects/vmodule.cpp" file.
		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		// No longer supported
		//if (str.compare("none") == 0)
		//	ptr = new konan::eco::Module;

		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		if (str.compare("linear") == 0)
			ptr = konan::eco::NN(session_ptr).Linear(args).clone_core();
		else if (str.compare("addbias") == 0)
			ptr = konan::eco::NN(session_ptr).AddBias(args).clone_core();
		else if (str.compare("dense") == 0)
			ptr = konan::eco::NN(session_ptr).Dense(args).clone_core();
		else if (str.compare("flatten") == 0)
			ptr = konan::eco::NN(session_ptr).Flatten(args).clone_core();
		else if (str.compare("reshape") == 0)
			ptr = konan::eco::NN(session_ptr).Reshape(args).clone_core();
		else if (str.compare("transpose") == 0)
			ptr = konan::eco::NN(session_ptr).Transpose(args).clone_core();
		else if (str.compare("concat") == 0)
			ptr = konan::eco::NN(session_ptr).Concat(args).clone_core();

		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		else if (str.compare("conv") == 0) {
			// This keyword is not included in the ms_builtinLayer.
			ptr = konan::eco::NN(session_ptr).Conv(args).clone_core();
		}
		else if (str.compare("conv1d") == 0)
			ptr = konan::eco::NN(session_ptr).Conv1D(args).clone_core();
		else if (str.compare("conv2d") == 0)
			ptr = konan::eco::NN(session_ptr).Conv2D(args).clone_core();
		else if (str.compare("conv2d_transposed") == 0)
			ptr = konan::eco::NN(session_ptr).Conv2D_Transposed(args).clone_core();
		else if (str.compare("conv2d_dilated") == 0)
			ptr = konan::eco::NN(session_ptr).Conv2D_Dilated(args).clone_core();
		else if (str.compare("conv2d_separable") == 0) {
			ptr = NULL;
			std::string errlog = std::string("EModule \"") + str + std::string("\" is not yet implemented");
			throw std::runtime_error(errlog.c_str());
		}
		else if (str.compare("conv2d_depthwise_separable") == 0
			|| str.compare("conv2d_pointwise") == 0
			|| str.compare("conv2d_grouped") == 0
			|| str.compare("conv2d_degormable") == 0) {
			ptr = NULL;
			std::string errlog = std::string("EModule \"") + str + std::string("\" is not yet implemented");
			throw std::runtime_error(errlog.c_str());
		}
		else if (str.compare("deconv") == 0) {
			// This keyword is not included in the ms_builtinLayer.
			ptr = konan::eco::NN(session_ptr).Deconv(args).clone_core();
		}
		else if (str.compare("max") == 0)
			ptr = konan::eco::NN(session_ptr).Max(args).clone_core();
		else if (str.compare("avg") == 0)
			ptr = konan::eco::NN(session_ptr).Avg(args).clone_core();
		else if (str.compare("globalavg") == 0)
			ptr = konan::eco::NN(session_ptr).GlobalAvg(args).clone_core();
		else if (str.compare("adaptiveavg") == 0)
			ptr = konan::eco::NN(session_ptr).AdaptiveAvg(args).clone_core();
		else if (str.compare("layernorm") == 0)
			ptr = konan::eco::NN(session_ptr).Layernorm(args).clone_core();
		else if (str.compare("batchnorm") == 0)
			ptr = konan::eco::NN(session_ptr).Batchnorm(args).clone_core();
		else if (str.compare("upsample") == 0)
			ptr = konan::eco::NN(session_ptr).Upsample(args).clone_core();
		else if (str.compare("pass") == 0)
			ptr = konan::eco::NN(session_ptr).Pass(args).clone_core();

		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		else if (str.compare("rnn") == 0)
			ptr = konan::eco::NN(session_ptr).Rnn(args).clone_core();
		else if (str.compare("lstm") == 0)
			ptr = konan::eco::NN(session_ptr).Lstm(args).clone_core();
		else if (str.compare("gru") == 0)
			ptr = konan::eco::NN(session_ptr).Gru(args).clone_core();

		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		else if (str.compare("embedding") == 0)
			ptr = konan::eco::NN(session_ptr).Embed(args).clone_core();
		else if (str.compare("dropout") == 0)
			ptr = konan::eco::NN(session_ptr).Dropout(args).clone_core();
		else if (str.compare("extract") == 0)
			ptr = konan::eco::NN(session_ptr).Extract(args).clone_core();
		else if (str.compare("mh_attention") == 0)
			ptr = konan::eco::NN(session_ptr).MultiHeadAttention(args).clone_core();

		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		else if (str.compare("activate") == 0)
			ptr = konan::eco::NN(session_ptr).Activate(args).clone_core();
		else if (str.compare("relu") == 0)
			ptr = konan::eco::NN(session_ptr).ReLU(args).clone_core();
		else if (str.compare("gelu") == 0)
			ptr = konan::eco::NN(session_ptr).Gelu(args).clone_core();
		else if (str.compare("softmax") == 0)
			ptr = konan::eco::NN(session_ptr).Softmax(args).clone_core();
		else if (str.compare("sigmoid") == 0)
			ptr = konan::eco::NN(session_ptr).Sigmoid(args).clone_core();
		else if (str.compare("tanh") == 0)
			ptr = konan::eco::NN(session_ptr).Tanh(args).clone_core();
		else if (str.compare("mish") == 0)
			ptr = konan::eco::NN(session_ptr).Mish(args).clone_core();
		else if (str.compare("swish") == 0)
			ptr = konan::eco::NN(session_ptr).Swish(args).clone_core();
		else if (str.compare("leaky_relu") == 0)
			ptr = konan::eco::NN(session_ptr).Leaky(args).clone_core();

		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		else if (str.compare("noise") == 0)
			ptr = konan::eco::NN(session_ptr).Noise(args).clone_core();
		else if (str.compare("random") == 0) {
			// This keyword is not included in the ms_builtinLayer.
			ptr = konan::eco::NN(session_ptr).Random(args).clone_core();
		}
		else if (str.compare("round") == 0)
			ptr = konan::eco::NN(session_ptr).Round(args).clone_core();
		else if (str.compare("codeconv") == 0)
			ptr = konan::eco::NN(session_ptr).CodeConv(args).clone_core();
		else if (str.compare("cosinesim") == 0)
			ptr = konan::eco::NN(session_ptr).CosineSim(args).clone_core();
		else if (str.compare("selectntop") == 0)
			ptr = konan::eco::NN(session_ptr).SelectNTop(args).clone_core();
		else if (str.compare("selectntoparg") == 0)
			ptr = konan::eco::NN(session_ptr).SelectNTopArg(args).clone_core();
		else if (str.compare("formula") == 0)
			ptr = konan::eco::NN(session_ptr).Formula(args).clone_core();

		////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////

		else {
			ptr = NULL;
			std::string errlog = std::string("Unsupported module(\"") + str + std::string("\")");
			throw std::runtime_error(errlog.c_str());
		}

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteModule(void* module_ptr)
{
	try {
		if (module_ptr == NULL)
			return false;

		konan::eco::Object::free(module_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void* CreateContainer(void* session_ptr, PyObject* container_name_str, PyObject* module_ptr_list, PyObject* args_dict)
{
	try {
		// Get a name of target module
		std::string str = PyUnicode_AsUTF8(container_name_str);
		//std::string str = TpUtils::seekDict(get_args, "container_name_str", "not defined");

		// Convert to lower case
		for (int i=0; i<str.size(); ++i)
			str[i] = tolower(str[i]);

		// Convert the PyList of the module pointers to the std::vector<void*>
		std::vector<void*> temp_vector;
		temp_vector.clear();
		kpy::ConvertPyListToVector(module_ptr_list, &temp_vector);

		// Convert the vector to ModuleList
		std::vector<konan::eco::Module> module_list;
		module_list.clear();
		for (int i=0; i<temp_vector.size(); ++i)
			module_list.push_back(konan::eco::Module(temp_vector[i]));

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		// Declare an object of Module
		void* ptr = NULL;

		////////////////////////////////////////////////////////////////
		// Normal containers
		////////////////////////////////////////////////////////////////

		if (str.compare("sequential") == 0)
			ptr = konan::eco::NN(session_ptr).Sequential(module_list, args).clone_core();

		////////////////////////////////////////////////////////////////
		// Original containers
		////////////////////////////////////////////////////////////////

		else if (str.compare("add") == 0)
			ptr = konan::eco::NN(session_ptr).Add(module_list, args).clone_core();
		else if (str.compare("residual") == 0)
			ptr = konan::eco::NN(session_ptr).Residual(module_list, args).clone_core();
		else if (str.compare("parallel") == 0)
			ptr = konan::eco::NN(session_ptr).Parallel(module_list, args).clone_core();
		else if (str.compare("pruning") == 0)
			ptr = konan::eco::NN(session_ptr).Pruning(module_list, args).clone_core();
		else if (str.compare("stack") == 0)
			ptr = konan::eco::NN(session_ptr).Stack(module_list, args).clone_core();
		else if (str.compare("squeezeexcitation") == 0)
			ptr = konan::eco::NN(session_ptr).SqueezeExcitation(module_list, args).clone_core();

		////////////////////////////////////////////////////////////////

		else {
			ptr = NULL;
			std::string errlog = std::string("Unsupported module(\"") + str + std::string("\")");
			throw std::runtime_error(errlog.c_str());
		}

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteContainer(void* container_module_ptr)
{
	try {
		if (container_module_ptr == NULL)
			return false;

		konan::eco::Object::free(container_module_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void RegisterMacro(void* session_ptr, PyObject* macro_name_str, void* module_ptr, PyObject* args_dict)
{
	try
	{
		// Get a key to create macro
		std::string macro_name = PyUnicode_AsUTF8(macro_name_str);
		//string macro_name = TpUtils::seekDict(get_args, "key", "not defined");

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		// Regist a key to create macro
		konan::eco::NN(session_ptr).regist_macro(macro_name, konan::eco::Module(module_ptr), args);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void* CreateMacro(void* session_ptr, PyObject* macro_name_str, PyObject* args_dict)
{
	try {
		// Get a key to create macro
		std::string macro_name = PyUnicode_AsUTF8(macro_name_str);
		//string macro_name = TpUtils::seekDict(get_args, "key", "not defined");

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		// Create a macro
		return konan::eco::NN(session_ptr).Macro(macro_name, args).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* ExpandModule(void* module_ptr, PyObject* data_shape_ptr, PyObject* args_dict)
{
	try {
		// Convert a data shape from PyList to VList
		VList temp_list;
		if (PyList_Check(data_shape_ptr)) {
			kpy::ConvertPyListToVList(data_shape_ptr, &temp_list);
		}
		else if (PyTuple_Check(data_shape_ptr)) {
			kpy::ConvertPyTupleToVList(data_shape_ptr, &temp_list);
		}
		else {
			throw std::runtime_error("bad shape: data shape argument must be list or tuple ");
		}

		// Convert a data shape from VList to VShape
		VShape data_shape;
		for (int i=0; i<temp_list.size(); ++i)
			data_shape = data_shape.append((int64)temp_list[i]);
	
		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		// Expand
		return konan::eco::Module(module_ptr).expand(data_shape, args).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* SetModuleToDevice(void* module_ptr, PyObject* device_name_str)
{
	try {
		std::string device_name = PyUnicode_AsUTF8(device_name_str);
		return konan::eco::Module(module_ptr).to(device_name).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* GetModuleShape(void* module_ptr)
{
	try {
		PyObject* module_shape_str = PyUnicode_FromString( konan::eco::Module(module_ptr).__str__().c_str() );
		return module_shape_str;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	PyObject* no_str = PyUnicode_FromString( std::string("").c_str() );

	return no_str;
}

void ModuleTrain(void* module_ptr)
{
	try {
		konan::eco::Module(module_ptr).train();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void ModuleEval(void* module_ptr)
{
	try {
		konan::eco::Module(module_ptr).eval();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void* ModuleCall(void* module_ptr, void* tensor_x_ptr)
{
	try {
		return konan::eco::Module(module_ptr).__call__(konan::eco::Tensor(tensor_x_ptr)).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* ModuleCallDict(void* module_ptr, PyObject* tensor_dict)
{
	try {
		konan::eco::TensorDict xs = m_PyDictToTensorDict(tensor_dict);

		konan::eco::TensorDict ys = konan::eco::Module(module_ptr).__call__(xs);

		VDict yHandles;

		for (auto& it : ys) {
			yHandles[it.first] = (int64)it.second.get_core();
		}
		
		PyObject* ptr = kpy::ConvertVDictToPyDict(yHandles);

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* ModulePredict(void* module_ptr, void* tensor_x_ptr)
{
	try {
		return konan::eco::Module(module_ptr).predict(konan::eco::Tensor(tensor_x_ptr)).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* ModulePredictDict(void* module_ptr, PyObject* tensor_dict)
{
	try {
		konan::eco::TensorDict xs = m_PyDictToTensorDict(tensor_dict);

		konan::eco::TensorDict ys = konan::eco::Module(module_ptr).predict(xs);

		VDict yHandles;

		for (auto& it : ys) {
			yHandles[it.first] = (int64)it.second.get_core();
		}

		PyObject* ptr = kpy::ConvertVDictToPyDict(yHandles);

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void ModuleAppendChild(void* module_ptr, void* child_ptr)
{
	try {
		konan::eco::Module child(child_ptr);
		konan::eco::Module(module_ptr).append_child(child);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void* ModuleNthChild(void* module_ptr, PyObject* nth_ptr)
{
	try {
		int nth = PyLong_AsLong(nth_ptr);
		return konan::eco::Module(module_ptr).nth_child(nth).clone_core();;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	return NULL;
}

KPY_API void* ModuleFetchChild(void* module_ptr, PyObject* name_str)
{
	try {
		string name = PyUnicode_AsUTF8(name_str);
		return konan::eco::Module(module_ptr).fetch_child(name).clone_core();;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	return NULL;
}

////////////////////////////////////////////////////////////////
// Loss
////////////////////////////////////////////////////////////////

PyObject* m_TensorDictToPyDict(konan::eco::TensorDict tdict) {
	VDict vdict;

	for (auto& it : tdict) {
		vdict[it.first] = (int64)it.second.get_core();
	}

	PyObject* ptr = kpy::ConvertVDictToPyDict(vdict);

	return ptr;
}

void* CreateLoss(void* session_ptr, PyObject* loss_name_str, PyObject* est_str, PyObject* ans_str, PyObject* args_dict)
{
	try {
		// Convert the string
		std::string loss_name = PyUnicode_AsUTF8(loss_name_str);
		std::string est = PyUnicode_AsUTF8(est_str);
		std::string ans = PyUnicode_AsUTF8(ans_str);

		// Convert to lower case
		for (int i=0; i<loss_name.size(); ++i)
			loss_name[i] = tolower(loss_name[i]);
		for (int i=0; i<est.size(); ++i)
			est[i] = tolower(est[i]);
		for (int i=0; i<ans.size(); ++i)
			ans[i] = tolower(ans[i]);

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		// Declare an object of Loss
		void* ptr = NULL;

		// Create an instance of ELoss (Refer to vloss.cpp)
		if (loss_name.compare("mse") == 0)
			ptr = konan::eco::NN(session_ptr).MSELoss(args, est, ans).clone_core();
		else if (loss_name.compare("crossentropy") == 0)
			ptr = konan::eco::NN(session_ptr).CrossEntropyLoss(args, est, ans).clone_core();
		else if (loss_name.compare("binary_crossentropy") == 0)
			ptr = konan::eco::NN(session_ptr).BinaryCrossEntropyLoss(args, est, ans).clone_core();
		else if (loss_name.compare("crossentropy_sigmoid") == 0)
			ptr = konan::eco::NN(session_ptr).CrossEntropySigmoidLoss(args, est, ans).clone_core();
		else if (loss_name == "crossentropy_pos_idx")
			ptr = konan::eco::NN(session_ptr).CrossEntropyPositiveIdxLoss(args, est, ans).clone_core();
		else if (loss_name.compare("custom") == 0) {
			ptr = NULL;
			std::string errlog = std::string("ELoss \"") + loss_name + std::string("\" is not yet implemented");
			throw std::runtime_error(errlog.c_str());
		}
		else {
			ptr = NULL;
			std::string errlog = std::string("Unsupported loss(\"") + loss_name + std::string("\")");
			throw std::runtime_error(errlog.c_str());
		}

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* CreateMultipleLoss(void* session_ptr, PyObject* children_dict)
{
	try {
		konan::eco::LossDict losses;

		// Convert the datatype of the arguments from PyDict to VDict
		VDict dict;
		kpy::ConvertPyDictToVDict(children_dict, &dict);

		for (auto& it : dict) {
			losses[it.first] = konan::eco::Loss((void*)(int64)it.second);
		}

		void* ptr = konan::eco::NN(session_ptr).MultipleLoss(losses).clone_core();

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* CreateCustomLoss(void* session_ptr, PyObject* loss_terms_dict, PyObject* static_dict, PyObject* args_dict)
{
	try
	{
		VDict loss_terms;
		VDict args;

		kpy::ConvertPyDictToVDict(loss_terms_dict, &loss_terms);
		kpy::ConvertPyDictToVDict(args_dict, &args);

		konan::eco::TensorDict statistics = m_PyDictToTensorDict(static_dict);

		return konan::eco::NN(session_ptr).CustomLoss(loss_terms, statistics, args).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteLoss(void* loss_ptr)
{
	try {
		if (loss_ptr == NULL)
			return false;

		konan::eco::Object::free(loss_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void* LossEvaluate(void* loss_ptr, void* tensor_pred_ptr, void* tensor_y_ptr, PyObject* download_all_flag)
{
	try {
		bool download_all = (download_all_flag == Py_True);

		konan::eco::Tensor tensor_pred(tensor_pred_ptr);
		konan::eco::Tensor tensor_y(tensor_y_ptr);

		return konan::eco::Loss(loss_ptr).__call__(tensor_pred, tensor_y, download_all).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* LossEvaluateDict(void* loss_ptr, PyObject* pred_dict, PyObject* y_dict, PyObject* download_all_flag)
{
	try {
		konan::eco::TensorDict preds = m_PyDictToTensorDict(pred_dict);
		konan::eco::TensorDict ys = m_PyDictToTensorDict(y_dict);
		
		bool download_all = (download_all_flag == Py_True);

		if (0) {
			printf("KP1\n");
			for (auto& it : preds) {
				printf("KP1: preds[%s] = T#%d\n", it.first.c_str(), it.second.get_engine_id());
			}
			for (auto& it : ys) {
				printf("KP1: ys[%s] = T#%d\n", it.first.c_str(), it.second.get_engine_id());
			}
		}

		konan::eco::TensorDict losses = konan::eco::Loss(loss_ptr).__call__(preds, ys, download_all);

		return m_TensorDictToPyDict(losses);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	return NULL;
}

void* LossEvalAccuracy(void* loss_ptr, void* tensor_pred_ptr, void* tensor_y_ptr, PyObject* download_all_flag)
{
	try {
		konan::eco::Tensor tensor_pred(tensor_pred_ptr);
		konan::eco::Tensor tensor_y(tensor_y_ptr);
		
		bool download_all = (download_all_flag == Py_True);

		return konan::eco::Loss(loss_ptr).eval_accuracy(tensor_pred, tensor_y, download_all).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* LossEvalAccuracyDict(void* loss_ptr, PyObject* pred_dict, PyObject* y_dict, PyObject* download_all_flag)
{
	try {
		konan::eco::TensorDict preds = m_PyDictToTensorDict(pred_dict);
		konan::eco::TensorDict ys = m_PyDictToTensorDict(y_dict);

		bool download_all = (download_all_flag == Py_True);

		if (0) {
			printf("KP2\n");
			for (auto& it : preds) {
				printf("KP2: preds[%s] = T#%d\n", it.first.c_str(), it.second.get_engine_id());
			}
			for (auto& it : ys) {
				printf("KP2: ys[%s] = T#%d\n", it.first.c_str(), it.second.get_engine_id());
			}
		}

		konan::eco::TensorDict accs = konan::eco::Loss(loss_ptr).eval_accuracy(preds, ys, download_all);

		return m_TensorDictToPyDict(accs);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	return NULL;
}

KPY_API void LossBackward(void* loss_ptr) {
	try {
		konan::eco::Loss(loss_ptr).backward();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

////////////////////////////////////////////////////////////////
// Metric
////////////////////////////////////////////////////////////////

void* CreateFormulaMetric(void* session_ptr, PyObject* metric_name_str, PyObject* formula_str, PyObject* args_dict)
{
	try {
		// Convert the string
		std::string metric_name = PyUnicode_AsUTF8(metric_name_str);
		std::string formula = PyUnicode_AsUTF8(formula_str);

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		return konan::eco::NN(session_ptr).FormulaMetric(metric_name, formula, args).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* CreateMultipleMetric(void* session_ptr, PyObject* children_dict, PyObject* args_dict)
{
	try {
		konan::eco::MetricDict metrices;

		// Convert the datatype of the arguments from PyDict to VDict
		VDict dict;
		kpy::ConvertPyDictToVDict(children_dict, &dict);

		for (auto& it : dict) {
			metrices[it.first] = konan::eco::Metric((void*)(int64)it.second);
		}

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		return konan::eco::NN(session_ptr).MultipleMetric(metrices, args).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* CreateCustomMetric(void* session_ptr, PyObject* metric_terms_dict, PyObject* static_dict, PyObject* args_dict)
{
	try
	{
		VDict metric_terms;
		VDict args;

		kpy::ConvertPyDictToVDict(metric_terms_dict, &metric_terms);
		kpy::ConvertPyDictToVDict(args_dict, &args);

		konan::eco::TensorDict statistics = m_PyDictToTensorDict(static_dict);

		return konan::eco::NN(session_ptr).CustomMetric(metric_terms, statistics, args).clone_core();
	}
	catch (konan::eco::EcoException ex) {
		printf("Temp Error:\n%s", ex.get_error_message(4).c_str());
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteMetric(void* metric_ptr)
{
	try {
		if (metric_ptr == NULL)
			return false;

		konan::eco::Object::free(metric_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void* MetricEvaluate(void* metric_ptr, void* tensor_pred_ptr)
{
	try {
		konan::eco::Tensor tensor_pred(tensor_pred_ptr);
		return konan::eco::Metric(metric_ptr).__call__(tensor_pred).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* MetricEvaluateDict(void* metric_ptr, PyObject* pred_dict)
{
	try {
		konan::eco::TensorDict preds = m_PyDictToTensorDict(pred_dict);
		konan::eco::TensorDict metrices = konan::eco::Metric(metric_ptr).__call__(preds);

		return m_TensorDictToPyDict(metrices);
	}
	catch (konan::eco::EcoException ex) {
		printf("Temp Error:\n%s", ex.get_error_message(4).c_str());
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	return NULL;
}

////////////////////////////////////////////////////////////////
// Parameters
////////////////////////////////////////////////////////////////

void* CreateParameters(void* module_ptr)
{
	try {
		return konan::eco::Module(module_ptr).parameters().clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteParameters(void* parameters_ptr)
{
	try {
		if (parameters_ptr == NULL)
			return false;

		konan::eco::Object::free(parameters_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

PyObject* GetParametersDump(void* parameters_ptr, bool is_full)
{
	try {
		std::string get_str = "";
		for (auto& it : konan::eco::Parameters(parameters_ptr).weightDict()) {
			get_str += it.second.get_dump_str(it.first, is_full);
		}
		PyObject* dump_str = PyUnicode_FromString(get_str.c_str());

		return dump_str;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return PyUnicode_FromString(std::string("").c_str());
}

PyObject* GetParameterWeightDict(void* parameters_ptr)
{
	try {
		konan::eco::TensorDict weights = konan::eco::Parameters(parameters_ptr).weightDict();

		VDict wHandles;

		for (auto& it : weights) {
			wHandles[it.first] = (int64)it.second.get_core();
		}

		PyObject* ptr = kpy::ConvertVDictToPyDict(wHandles);

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return PyUnicode_FromString(std::string("").c_str());
}

PyObject* GetParameterGradientDict(void* parameters_ptr)
{
	try {
		konan::eco::TensorDict weights = konan::eco::Parameters(parameters_ptr).gradientDict();

		VDict wHandles;

		for (auto& it : weights) {
			wHandles[it.first] = (int64)it.second.get_core();
		}

		PyObject* ptr = kpy::ConvertVDictToPyDict(wHandles);

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return PyUnicode_FromString(std::string("").c_str());
}

////////////////////////////////////////////////////////////////
// Optimizer
////////////////////////////////////////////////////////////////

void* CreateOptimizer(void* session_ptr, void* parameters_ptr, PyObject* optimizer_name_str, PyObject* args_dict)
{
	try {
		// Convert the string
		std::string name = PyUnicode_AsUTF8(optimizer_name_str);

		// Convert the datatype of the arguments from PyDict to VDict
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		// Declare an object of Optimizer
		void* ptr = NULL;

		ptr = konan::eco::NN(session_ptr).createOptimizer(name, konan::eco::Parameters(parameters_ptr), args).clone_core();

		return ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteOptimizer(void* optimizer_ptr)
{
	try {
		if (optimizer_ptr == NULL)
			return false;

		konan::eco::Object::free(optimizer_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void OptimizerSetup(void* optimizer_ptr, PyObject* args_dict_str) {
	try {
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict_str, &args);
		konan::eco::Optimizer(optimizer_ptr).set_option(args);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

bool OptimizerZeroGrad(void* optimizer_ptr)
{
	try {
		konan::eco::Optimizer(optimizer_ptr).zero_grad();
		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

bool OptimizerStep(void* optimizer_ptr)
{
	try {
		konan::eco::Optimizer(optimizer_ptr).step();
		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

////////////////////////////////////////////////////////////////
// Tensor
////////////////////////////////////////////////////////////////

void* CreateEmptyTensor(void)
{
	try {
		konan::eco::Tensor empty_tensor;
		return empty_tensor.clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void* CreateTensor(void* session_ptr, PyObject* shape_tuple, PyObject* type_str, PyObject* init) {
	try {
		VShape shape;
		kpy::ConvertPyTupleToVShape(shape_tuple, &shape);

		std::string type_name = PyUnicode_AsUTF8(type_str);

		VDataType type = konan::eco::Tensor::name_to_type(type_name);

		VValue init_val;
		kpy::ConvertPyObjectToVValue(init, &init_val);

		if (init_val.is_none()) {
			return konan::eco::NN(session_ptr).createTensor(shape, type).clone_core();
		}
		else if (init_val.is_string()) {
			return konan::eco::NN(session_ptr).createTensor(shape, type, (string)init_val).clone_core();
		}
		else if (init_val.is_list()) {
			return konan::eco::NN(session_ptr).createTensor(shape, type, (VList)init_val).clone_core();
		}
		else {
			throw std::runtime_error("improper init value for tensor creating");
		}
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void* CreateTensorFromNumPy(void* session_ptr, void* numpy_ptr, PyObject* numpy_type_int, PyObject* numpy_shape_list)
{
	try {
		// Convert a shape of the NumPy object from PyList to VList
		VList numpy_shape;
		kpy::ConvertPyListToVList(numpy_shape_list, &numpy_shape);

		// Convert a shape of the NumPy object from VList to VShape
		int64 tensor_dim = numpy_shape.size();
		int64* tensor_size = new int64[tensor_dim];

		if (tensor_size == NULL) throw std::runtime_error("Memory allocation failure");

		for (int idx=0; idx<tensor_dim; ++idx) {
			int64 element = numpy_shape[idx];
			tensor_size[idx] = element;
		}

		VShape tensor_shape(tensor_dim, tensor_size);

		delete [] tensor_size;
		tensor_size = NULL;

		// Convert a type of the NumPy object from PyInt to int64
		int64 numpy_type = PyLong_AsLongLong(numpy_type_int);

		// Create a Tensor object
		return konan::eco::NN(session_ptr).createTensor(tensor_shape, (VDataType)numpy_type, numpy_ptr ).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteTensor(void* tensor_ptr)
{
	try {
		if (tensor_ptr == NULL)
			return false;

		konan::eco::Object::free(tensor_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void DumpTensor(void* tensor_ptr, PyObject* tensor_name_str, PyObject* full_flag)
{
	try {
		std::string tensor_name = PyUnicode_AsUTF8(tensor_name_str);
		bool is_full = (full_flag == Py_True);
		konan::eco::Tensor(tensor_ptr).dump(tensor_name, is_full);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

PyObject* GetTensorDump(void* tensor_ptr, PyObject* tensor_name_str, PyObject* full_flag)
{
	try {
		std::string tensor_name = PyUnicode_AsUTF8(tensor_name_str);
		bool is_full = (full_flag == Py_True);

		std::string dump = konan::eco::Tensor(tensor_ptr).get_dump_str(tensor_name, is_full);

		return PyUnicode_FromString(dump.c_str());
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return PyUnicode_FromString(std::string("").c_str());
}

int64 GetTensorLength(void* tensor_ptr)
{
	try {
		return konan::eco::Tensor(tensor_ptr).len();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return 0;
}

int64 GetTensorSize(void* tensor_ptr)
{
	try {
		int64 size = konan::eco::Tensor(tensor_ptr).size();
		return size;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return 0;
}

PyObject* GetTensorShape(void* tensor_ptr)
{
	try {
		VShape tensor_shape = konan::eco::Tensor(tensor_ptr).shape();
		PyObject* tensor_shape_tuple = kpy::ConvertVShapeToPyTuple(&tensor_shape);
		return tensor_shape_tuple;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void CopyNumPyToTensor(void* tensor_ptr, void* numpy_ptr, PyObject* numpy_type_int, PyObject* numpy_shape_list)
{
	try {
		// Convert a shape of the NumPy object from PyList to VList
		VList numpy_shape;
		kpy::ConvertPyListToVList(numpy_shape_list, &numpy_shape);

		// Convert a shape of the NumPy object from VList to VShape
		int64 tensor_dim = numpy_shape.size();
		int64* tensor_size = new int64[tensor_dim];

		if (tensor_size == NULL) throw std::runtime_error("Memory allocation failure");

		for (int idx=0; idx<tensor_dim; ++idx) {
			int64 element = numpy_shape[idx];
			tensor_size[idx] = element;
		}

		VShape tensor_shape(tensor_dim, tensor_size);

		delete [] tensor_size;

		// Convert a type of the NumPy object from PyInt to int64
		int64 numpy_type = PyLong_AsLongLong(numpy_type_int);

		// Get an object of the Tensor
		konan::eco::Tensor tensor(tensor_ptr);
		
		switch (numpy_type) {
			case (int64)(VDataType::float32) :
				tensor.copy_data(tensor_shape, VDataType::float32, (float*)numpy_ptr);
				break;
			case (int64)(VDataType::int32) :
				tensor.copy_data(tensor_shape, VDataType::int32, (int*)numpy_ptr);
				break;
			case (int64)(VDataType::int64) :
				tensor.copy_data(tensor_shape, VDataType::int64, (int64*)numpy_ptr);
				break;
			case (int64)(VDataType::uint8) :
				tensor.copy_data(tensor_shape, VDataType::uint8, (unsigned char*)numpy_ptr);
				break;
			case (int64)(VDataType::bool8) :
				tensor.copy_data(tensor_shape, VDataType::bool8, (bool*)numpy_ptr);
				break;
			case (int64)(VDataType::float64) :
				tensor.copy_data(tensor_shape, VDataType::float64, (bool*)numpy_ptr);
				break;
			default: {
				throw std::runtime_error("Unsupported NumPy data type");
				break;
			}
		}
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

PyObject* ConvertTensorToNumPy(void* tensor_ptr)
{
	// Ensure that the current thread is ready to call the Python C API
	// regardless of the current state of Python, or of the global interpreter lock.
	PyGILState_STATE GILState = PyGILState_Ensure();

	try {
		// Declare an object of the Tensor
		konan::eco::Tensor tensor(tensor_ptr);

		// Check the validation
		if (tensor.has_no_data())
			throw std::runtime_error("Tensor has no data");		

		// Check the type of the Tensor
		// enum : float32, int32, int64, uint8, bool8 (float64 is not yet supported)
		VDataType get_type = tensor.type();

		// Get a data_type and a pointer of Tensor data
		int64 data_type = 0;
		void* data_ptr = NULL;

		switch (get_type) {
			case VDataType::float32: {
				data_type = (int64)(VDataType::float32);
				data_ptr = tensor.float_ptr();
				break;
			}
			case VDataType::int32: {
				data_type = (int64)(VDataType::int32);
				data_ptr = tensor.int_ptr();
				break;
			}
			case VDataType::int64: {
				data_type = (int64)(VDataType::int64);
				data_ptr = tensor.int64_ptr();
				break;
			}
			case VDataType::uint8: {
				data_type = (int64)(VDataType::uint8);
				data_ptr = tensor.uchar_ptr();
				break;
			}
			case VDataType::bool8: {
				data_type = (int64)(VDataType::bool8);
				data_ptr = tensor.bool_ptr();
				break;
			}
			default: {
				throw std::runtime_error("Unsupported ETensor data type");
				break;
			}
		}

		VShape get_shape = tensor.shape();
		int64 get_byte_size = tensor.byte_size();

		// Info[0] : Type
		// Info[1] : Pointer
		// Info[3 ~ N] is Shape
		PyObject* tensor_info = PyList_New(2 + get_shape.size());

		// Info[0] Set : Type
		PyList_SET_ITEM(tensor_info, 0, PyLong_FromLong((int)data_type));

		// Info[1] Set : Pointer
		PyList_SET_ITEM(tensor_info, 1, PyLong_FromVoidPtr(data_ptr));

		// Info[2,:] Set : Shape
		for (int idx=0; idx<get_shape.size(); ++idx)
			PyList_SET_ITEM(tensor_info, 2 + idx, PyLong_FromLong((int)(get_shape[idx])));

		PyGILState_Release(GILState);

		return tensor_info;
	}
	catch (...) {
		PyGILState_Release(GILState);
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

float ConvertTensorToScalar(void* tensor_ptr)
{
	try {
		float scalar = konan::eco::Tensor(tensor_ptr).item();
		return scalar;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return 0.0f;
}

void* GetTensorArgmax(void* tensor_ptr, int64 axis)
{
	try {
		return konan::eco::Tensor(tensor_ptr).argmax(axis).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void TensorBackward(void* tensor_ptr)
{
	try {
		konan::eco::Tensor(tensor_ptr).backward();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void TensorBackwardWithGrad(void* tensor_ptr, void* tensor_grad_ptr)
{
	try {
		konan::eco::Tensor(tensor_ptr).backward_with_gradient( konan::eco::Tensor(tensor_grad_ptr) );
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void* ApplySigmoidToTensor(void* tensor_ptr)
{
	try {
		// If you want to self-convert like blow,
		// *(ETensor*)pTensor = ((ETensor*)pTensor)->sigmoid();
		// you need to make other function.
		return konan::eco::Tensor(tensor_ptr).sigmoid().clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void ModuleLoadCfgWeight(void* module_ptr, PyObject* cfg_path_ptr, PyObject* weight_path_ptr) {
	try {
		std::string cfg_path = PyUnicode_AsUTF8(cfg_path_ptr);
		std::string weight_path = PyUnicode_AsUTF8(weight_path_ptr);

		konan::eco::Module(module_ptr).load_cfg_config(cfg_path, weight_path);
	}
	catch (konan::eco::EcoException ex) {
		printf("Temp Error:\n%s", ex.get_error_message(4).c_str());
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void ModuleSave(void* module_ptr, PyObject* path_ptr) {
	try {
		std::string path = PyUnicode_AsUTF8(path_ptr);

		konan::eco::Module(module_ptr).save(path);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void ModuleInitParameters(void* module_ptr) {
	try {
		konan::eco::Module(module_ptr).init_parameters();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void* TensorToType(void* tensor_ptr, PyObject* type_name_str, PyObject* option_str) {
	try {
		std::string type_name = PyUnicode_AsUTF8(type_name_str);
		std::string option = PyUnicode_AsUTF8(option_str);

		return konan::eco::Tensor(tensor_ptr).to_type(type_name, option).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void* TensorIndexedByInt(void* tensor_ptr, PyObject* index_ptr) {
	try {
		int index = PyLong_AsLong(index_ptr);

		return konan::eco::Tensor(tensor_ptr)[index].clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void* TensorIndexedByTensor(void* tensor_ptr, void* index_ptr) {
	try {
		konan::eco::Tensor index = konan::eco::Tensor(index_ptr);

		return konan::eco::Tensor(tensor_ptr)[index].clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void* TensorIndexedBySlice(void* tensor_ptr, PyObject* index_ptr) {
	try {
		VList index;
		int64 dataSize = 0;

		VShape shape = konan::eco::Tensor(tensor_ptr).shape();

		kpy::ConvertPySliceIndexToVList(index_ptr, shape, &index, dataSize);

		return konan::eco::Tensor(tensor_ptr).get_slice(index).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API PyObject* ValueIndexedBySlice(void* tensor_ptr, PyObject* index_int) {
	try {
		int64 index = PyLong_AsLongLong(index_int);

		konan::eco::Tensor tensor(tensor_ptr);
		
		switch (tensor.type()) {
		case VDataType::float32:
		{
			float* ptr = tensor.float_ptr();
			float value = ptr[index];
			return PyFloat_FromDouble((double)value);
		}
		case VDataType::int32:
		{
			int* ptr = tensor.int_ptr();
			int value = ptr[index];
			return PyLong_FromLong(value);
		}
		case VDataType::int64:
		{
			int64* ptr = tensor.int64_ptr();
			int64 value = ptr[index];
			return PyLong_FromLongLong(value);
		}
		default:
			throw std::runtime_error("ValueIndexedBySlice(): uncovered type yet");
		}
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void TensorSetElementByTensor(void* tensor_ptr, PyObject* index_ptr, void* src_tensor_ptr) {
	try {
		VList index;
		konan::eco::Tensor tensor(src_tensor_ptr);

		VShape shape = konan::eco::Tensor(tensor_ptr).shape();
		int64 dataSize = tensor.shape().total_size();
		void* value_ptr = tensor.void_ptr();

		kpy::ConvertPySliceIndexToVList(index_ptr, shape, &index, dataSize);
		VDataType dataType = tensor.type();

		konan::eco::Tensor(tensor_ptr).set_slice(index, dataType, dataSize, value_ptr);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorSetElementByArray(void* tensor_ptr, PyObject* index_ptr, int64 dataSize, PyObject* value_or_type, void* value_ptr) {
	try {
		VList index;

		VShape shape = konan::eco::Tensor(tensor_ptr).shape();

		kpy::ConvertPySliceIndexToVList(index_ptr, shape, &index, dataSize);
		int64 numpy_type = PyLong_AsLongLong(value_or_type);

		konan::eco::Tensor(tensor_ptr).set_slice(index, (VDataType)numpy_type, dataSize, value_ptr);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorSetElementByValue(void* tensor_ptr, PyObject* index_ptr, PyObject* value_or_type) {
	try {
		VList index;

		VShape shape = konan::eco::Tensor(tensor_ptr).shape();

		VValue value;

		kpy::ConvertPySliceIndexToVList(index_ptr, shape, &index, 1);
		kpy::ConvertPyObjectToVValue(value_or_type, &value);

		konan::eco::Tensor(tensor_ptr).set_element(index, value);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorPickupRows(void* tensor_ptr, void* src_tensor_ptr, int64 batch_size, void* idx_ptr) {
	try {
		konan::eco::Tensor src = konan::eco::Tensor(src_tensor_ptr);
		
		int* pnMap = (int*)idx_ptr;

		konan::eco::Tensor(tensor_ptr).fetch_idx_rows(src, pnMap, batch_size);

	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorCopyIntoRow(void* tensor_ptr, int64 nth, void* src_tensor_ptr) {
	try {
		konan::eco::Tensor src = konan::eco::Tensor(src_tensor_ptr);
		konan::eco::Tensor me = konan::eco::Tensor(tensor_ptr);

		if (src.type() != me.type()) throw std::runtime_error("unmatched tensor type for copy row");
		if (src.shape() != me.shape().remove_head()) throw std::runtime_error("unmatched tensor shape for copy row");

		int64 batch_size = me.shape()[0];

		if (nth < 0 || nth >= batch_size) throw std::runtime_error("bad row index for copy row");

		konan::eco::Tensor(tensor_ptr).copy_into_row(nth, src);

	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorSetZero(void* tensor_ptr) {
	try {
		konan::eco::Tensor(tensor_ptr).reset();

	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorCopyData(void* tensor_ptr, void* src_tensor_ptr) {
	try {
		konan::eco::Tensor src = konan::eco::Tensor(src_tensor_ptr);

		konan::eco::Tensor(tensor_ptr).copy_partial_data(src);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorShiftTimestepToRight(void* tensor_ptr, void* src_tensor_ptr, int64 steps) {
	try {
		konan::eco::Tensor src = konan::eco::Tensor(src_tensor_ptr);

		konan::eco::Tensor(tensor_ptr).shift_timestep_to_right(src, steps);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void* TensorSquare(void* tensor_ptr) {
	try {
		return konan::eco::Tensor(tensor_ptr).square().clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void* TensorSum(void* tensor_ptr) {
	try {
		return konan::eco::Tensor(tensor_ptr).sum().clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* GetTensorTypeName(void* tensor_ptr)
{
	try {
		string tensor_type = konan::eco::Tensor(tensor_ptr).type_desc();
		PyObject* name_ptr = PyUnicode_FromString(tensor_type.c_str());
		return name_ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void* TensorResize(void* tensor_ptr, PyObject* shape_ptr) {
	try {
		VList list;
		kpy::ConvertPyListToVList(shape_ptr, &list);
		VShape shape(list);

		return konan::eco::Tensor(tensor_ptr).resize(shape).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void TensorResizeOn(void* tensor_ptr, void* src_ptr) {
	try {
		konan::eco::Tensor src(src_ptr);
		konan::eco::Tensor(tensor_ptr).resize_on(src);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorTransposeOn(void* tensor_ptr, PyObject* axis1_ptr, PyObject* axis2_ptr) {
	try {
		int64 axis1 = PyLong_AsLongLong(axis1_ptr);
		int64 axis2 = PyLong_AsLongLong(axis2_ptr);

		konan::eco::Tensor(tensor_ptr).transpose_on(axis1, axis2);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API void TensorLoadJpegPixels(void* tensor_ptr, PyObject* filepath_p, PyObject* chn_last_p, PyObject* row_first_p, PyObject* code_p, PyObject* mix_p) {
	try {
		string filepath = PyUnicode_AsUTF8(filepath_p);
		bool chn_last = (chn_last_p == Py_True);
		bool transpose = (row_first_p == Py_True);
		int code = PyLong_AsLong(code_p);
		float mix = (float)PyFloat_AsDouble(mix_p);

		konan::eco::Tensor(tensor_ptr).load_jpeg_pixels(filepath, chn_last, transpose, code, mix);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}
 
////////////////////////////////////////////////////////////////
// Utility Service Routines
////////////////////////////////////////////////////////////////

KPY_API PyObject* UtilParseJsonFile(PyObject* filepath_ptr) {
	try {
		std::string filepath = PyUnicode_AsUTF8(filepath_ptr);

		VValue jsons = konan::eco::Util::parse_json_file(filepath);

		PyObject* pyobject_ptr = kpy::ConvertVValueToPyObject(jsons);

		return pyobject_ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API PyObject* UtilParseJsonlFile(PyObject* filepath_ptr) {
	try {
		std::string filepath = PyUnicode_AsUTF8(filepath_ptr);

		VList jsons = konan::eco::Util::parse_jsonl_file(filepath);

		PyObject* list_ptr = kpy::ConvertVListToPyList(jsons);

		return list_ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API PyObject* UtilReadFileLines(PyObject* filepath_ptr) {
	try {
		std::string filepath = PyUnicode_AsUTF8(filepath_ptr);

		VStrList lines = konan::eco::Util::read_file_lines(filepath);

		PyObject* list_ptr = kpy::ConvertVStrListToPyList(lines);

		return list_ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API PyObject* UtilLoadData(void* session_ptr, PyObject* filepath_ptr) {
	try {
		std::string filepath = PyUnicode_AsUTF8(filepath_ptr);

		VValue data = konan::eco::Util::load_data(konan::eco::NN(session_ptr), filepath);

		PyObject* pyobject_ptr = kpy::ConvertVValueToPyObject(data);

		return pyobject_ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

KPY_API void UtilSaveData(void* session_ptr, PyObject* term_ptr, PyObject* filepath_ptr) {
	try {
		VValue value;

		kpy::ConvertPyObjectToVValue(term_ptr, &value);

		std::string filepath = PyUnicode_AsUTF8(filepath_ptr);

		konan::eco::Util::save_data(konan::eco::NN(session_ptr), value, filepath);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

KPY_API int64 UtilPositiveElementCount(void* tensor_ptr) {
	try {
		konan::eco::Tensor tensor(tensor_ptr);

		if (tensor.type() != VDataType::int32) {
			throw VERR_UNDEFINED;
		}
		
		int* ptr = tensor.int_ptr();
		int64 size = tensor.size();

		int64 pos_count = 0;

		for (int64 n = 0; n < size; n++) {
			if (ptr[n] > 0) pos_count++;
		}

		return pos_count;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

////////////////////////////////////////////////////////////////
// AudioSpectrumReader
////////////////////////////////////////////////////////////////

KPY_API void* CreateAudioSpectrumReader(void* session_ptr, PyObject* args_dict) {
	try {
		VDict args;
		kpy::ConvertPyDictToVDict(args_dict, &args);

		return konan::eco::NN(session_ptr).createAudioSpectrumReader(args).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool DeleteAudioSpectrumReader(void* reader_ptr)
{
	try {
		if (reader_ptr == NULL)
			return false;

		konan::eco::Object::free(reader_ptr);

		return true;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

bool AudioSpectrumReaderAddFile(void* reader_ptr, PyObject* filepath_str)
{
	try {
		std::string filepath = PyUnicode_AsUTF8(filepath_str);

		return konan::eco::AudioSpectrumReader(reader_ptr).add_file(filepath);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

void* AudioSpectrumReaderExtractSpectrums(void* reader_ptr)
{
	try {
		return konan::eco::AudioSpectrumReader(reader_ptr).get_fft_spectrums(false).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

////////////////////////////////////////////////////////////////
// Sample methods for developing new features
////////////////////////////////////////////////////////////////

PyObject* ExecutePythonFunction(PyObject* pyobject_callable, PyObject* pyobject_dict)
{
	PyGILState_STATE gil_state = PyGILState_Ensure();
	PyObject* tuple = NULL;

	try {
		Py_ssize_t arg_size = 1;

		// Create a new tuple
		tuple = PyTuple_New(arg_size);

		if (!tuple) {
			throw std::logic_error("Unable to allocate memory for Python tuple.");
		}

		// Check the validation
        if (!PyCallable_Check(pyobject_callable)) {
			throw std::logic_error("The first parameter must be callable.");
        }
		if (!PyDict_Check(pyobject_dict)) {
			throw std::logic_error("The type of the input parameter is not PyDict.");
		}

		// Set function arguments as the tuple
		PyTuple_SET_ITEM(tuple, 0, pyobject_dict);

		// Execute the python function
		PyObject* result = PyObject_CallObject(pyobject_callable, tuple);

		// Release a tuple object
		if (tuple)
			Py_DECREF(tuple);

		PyGILState_Release(gil_state);
		return result;
	}
	catch (...) {
		if (tuple)
			Py_DECREF(tuple);
		PyGILState_Release(gil_state);
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
