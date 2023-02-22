#include <eco_api.h>
#include "data_handler.h"
#include "exception_handler.h"
#include "inner_proc.h"

////////////////////////////////////////////////////////////////
// Inner procedure - added by taebin.heo on 2023.01.06
////////////////////////////////////////////////////////////////

PyObject* GetEngineVersion(void* session_ptr)
{
	try {
		// Get engine version str
		std::string str = konan::eco::NN(session_ptr).get_engine_version();

		PyObject* pyobject_str = PyUnicode_FromString(str.c_str());

		return pyobject_str;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void SetNoTracer(void* session_ptr)
{
	try {
		konan::eco::NN(session_ptr).set_no_tracer();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void UnsetNoTracer(void* session_ptr)
{
	try {
		konan::eco::NN(session_ptr).unset_no_tracer();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

PyObject* GetBuiltinNames(void* session_ptr)
{
	try
	{
		VDict get_dict = konan::eco::NN(session_ptr).get_builtin_names();

		PyObject* pyobject_dict = kpy::ConvertVDictToPyDict(get_dict);
		
		return pyobject_dict;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

bool IsBuiltinName(void* session_ptr, PyObject* domain_str, PyObject* builtin_str)
{
	try {
		// Get domain str
		std::string get_domain_str = PyUnicode_AsUTF8(domain_str);
		// Get builtin str
		std::string get_builtin_str = PyUnicode_AsUTF8(builtin_str);

		return konan::eco::NN(session_ptr).is_builtin_name(get_domain_str, get_builtin_str);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return false;
}

PyObject* GetLayerFormula(void* session_ptr, PyObject* layer_name_str)
{
	try {
		// Get layer name str
		std::string get_layer_name_str = PyUnicode_AsUTF8(layer_name_str);
		// Get layer formula str
		std::string get_str = konan::eco::NN(session_ptr).get_layer_formula(get_layer_name_str);

		PyObject* pyobject_str = PyUnicode_FromString(get_str.c_str());
		return pyobject_str;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

PyObject* GetLeakInfo(void* session_ptr, bool flag_bool)
{
	try
	{
		PyGILState_STATE gil_state = PyGILState_Ensure();

		VDict get_dict = konan::eco::NN(session_ptr).get_leak_info(flag_bool);

		PyObject* pyobject_dict = kpy::ConvertVDictToPyDict(get_dict);

		return pyobject_dict;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void DumpLeakInfo(void* session_ptr, bool flag_bool)
{
	try {
		konan::eco::NN(session_ptr).dump_leak_info(flag_bool);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void* CreateUserDefinedLayer(void* session_ptr, PyObject* name_str, PyObject* formula_str, PyObject* parameter_dict, PyObject* args_dict)
{
	try {
		// Get name str
		std::string get_name_str = PyUnicode_AsUTF8(name_str);
		// Get formula str
		std::string get_formula_str = PyUnicode_AsUTF8(formula_str);
		// Get padameter dict
		VDict get_parameter_dict;
		kpy::ConvertPyDictToVDict(parameter_dict, &get_parameter_dict);
		// Get args dict
		VDict get_args_dict;
		kpy::ConvertPyDictToVDict(args_dict, &get_args_dict);

		void* get_ptr = konan::eco::NN(session_ptr).create_user_defined_layer(get_name_str, get_formula_str, get_parameter_dict, get_args_dict).clone_core();

		return get_ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
	return NULL;
}

void* CreateModel(void* session_ptr, PyObject* model_name_str, PyObject* args_dict)
{
	try {
		// Get model name str
		std::string model_name = PyUnicode_AsUTF8(model_name_str);
		// Get args dict
		VDict get_args_dict;
		kpy::ConvertPyDictToVDict(args_dict, &get_args_dict);

		return konan::eco::NN(session_ptr).Model(model_name, get_args_dict).clone_core();
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}

void SaveModule(void* session_ptr, void* module_ptr, PyObject* file_name_str)
{
	try {
		// Get file name str
		std::string get_name_str = PyUnicode_AsUTF8(file_name_str);

		konan::eco::NN(session_ptr).save_module(konan::eco::Module(module_ptr), get_name_str);
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}
}

void* LoadModule(void* session_ptr, PyObject* file_name_str)
{
	try {
		// Get file name str
		std::string get_name_str = PyUnicode_AsUTF8(file_name_str);

		void* get_ptr = konan::eco::NN(session_ptr).load_module(get_name_str).clone_core();

		return get_ptr;
	}
	catch (...) {
		ExceptionHandler(__FILE__, __LINE__, __FUNCTION__);
	}

	return NULL;
}
