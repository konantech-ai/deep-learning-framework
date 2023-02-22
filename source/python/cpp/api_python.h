#pragma once

#ifdef KPY_EXPORTS
#ifdef KPY_API
#undef KPY_API
#endif
#define KPY_API __declspec(dllexport)
#else
#define KPY_API __declspec(dllimport)
#endif

#ifdef FOR_LINUX
#ifdef KPY_API
#undef KPY_API
#endif
#define KPY_API __attribute__((__visibility__("default")))
#endif

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <string.h>

#ifndef FOR_LINUX
#include <direct.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////
// Session and Device
////////////////////////////////////////////////////////////////
KPY_API void*     OpenSession(PyObject* server_url_str, PyObject* client_url_str);
KPY_API bool      CloseSession(void* session_ptr);

KPY_API void      SrandTime(void);
KPY_API void      SrandSeed(void* session_ptr, PyObject* seed_int);

KPY_API bool      IsCUDAAvailable(void* session_ptr);
KPY_API int64     GetCUDADeviceCount(void* session_ptr);

KPY_API void      SetNoGrad(void* session_ptr);
KPY_API void      UnsetNoGrad(void* session_ptr);

////////////////////////////////////////////////////////////////
// Module
////////////////////////////////////////////////////////////////
KPY_API void*     CreateModule(void* session_ptr, PyObject* module_name_str, PyObject* args_dict);
KPY_API bool      DeleteModule(void* module_ptr);
KPY_API void*     CreateContainer(void* session_ptr, PyObject* container_name_str, PyObject* module_ptr_list, PyObject* args_dict);
KPY_API bool      DeleteContainer(void* container_module_ptr);
KPY_API void      RegisterMacro(void* session_ptr, PyObject* macro_name_str, void* module_ptr, PyObject* args_dict);
KPY_API void*     CreateMacro(void* session_ptr, PyObject* macro_name_str, PyObject* args_dict);

KPY_API void*     ExpandModule(void* module_ptr, PyObject* data_shape_list, PyObject* args_dict);
KPY_API void*     SetModuleToDevice(void* module_ptr, PyObject* device_name_str);
KPY_API PyObject* GetModuleShape(void* module_ptr);

// Set the module to training mode or evaluation mode.
KPY_API void      ModuleTrain(void* module_ptr);
KPY_API void      ModuleEval(void* module_ptr);
KPY_API void      ModuleAppendChild(void* module_ptr, void* child_ptr);

KPY_API void*      ModuleNthChild(void* module_ptr, PyObject* nth);
KPY_API void*      ModuleFetchChild(void* module_ptr, PyObject* name_str);

// Predict an answer from input x. This function returns Tensor.
KPY_API void*     ModuleCall(void* module_ptr, void* tensor_x_ptr);
KPY_API PyObject* ModuleCallDict(void* module_ptr, PyObject* tensor_dict);
KPY_API void*     ModulePredict(void* module_ptr, void* tensor_x_ptr);
KPY_API PyObject* ModulePredictDict(void* module_ptr, PyObject* tensor_dict);

// Save the module
KPY_API void      ModuleSave(void* module_ptr, PyObject* path_ptr);

////////////////////////////////////////////////////////////////
// Loss
////////////////////////////////////////////////////////////////

KPY_API void*     CreateLoss(void* session_ptr, PyObject* loss_name_str, PyObject* est_str, PyObject* ans_str, PyObject* args_dict);
KPY_API void*     CreateMultipleLoss(void* session_ptr, PyObject* pyobject_dict);
KPY_API void*     CreateCustomLoss(void* session_ptr, PyObject* loss_terms_dict, PyObject* static_dict, PyObject* args_dict);
KPY_API bool      DeleteLoss(void* loss_ptr);
KPY_API void*     LossEvaluate(void* loss_ptr, void* tensor_pred_ptr, void* tensor_y_ptr, PyObject* download_all_flag);
KPY_API PyObject* LossEvaluateDict(void* loss_ptr, PyObject* pred_dict, PyObject* y_dict, PyObject* download_all_flag);
KPY_API PyObject* LossEvaluateDictFloat(void* loss_ptr, PyObject* pred_dict, PyObject* y_float_ptr, PyObject* download_all_flag);
KPY_API void*     LossEvalAccuracy(void* loss_ptr, void* tensor_pred_ptr, void* tensor_y_ptr, PyObject* download_all_flag);
KPY_API PyObject* LossEvalAccuracyDict(void* loss_ptr, PyObject* pred_dict, PyObject* y_dict, PyObject* download_all_flag);
KPY_API void      LossBackward(void* loss_ptr);

////////////////////////////////////////////////////////////////
// Metric
////////////////////////////////////////////////////////////////

KPY_API void*     CreateFormulaMetric(void* session_ptr, PyObject* metric_name_str, PyObject* formula_str, PyObject* args_dict);
KPY_API void*     CreateMultipleMetric(void* session_ptr, PyObject* pyobject_dict, PyObject* args_dict);
KPY_API void*     CreateCustomMetric(void* session_ptr, PyObject* loss_terms_dict, PyObject* static_dict, PyObject* args_dict);
KPY_API bool      DeleteMetric(void* loss_ptr);
KPY_API void*     MetricEvaluate(void* loss_ptr, void* tensor_pred_ptr);
KPY_API PyObject* MetricEvaluateDict(void* loss_ptr, PyObject* pred_dict);

////////////////////////////////////////////////////////////////
// Parameters
////////////////////////////////////////////////////////////////

KPY_API void*     CreateParameters(void* module_ptr);
KPY_API bool      DeleteParameters(void* parameters_ptr);
KPY_API PyObject* GetParametersDump(void* parameters_ptr, bool is_full = false);
KPY_API PyObject* GetParameterWeightDict(void* parameters_ptr);
KPY_API PyObject* GetParameterGradientDict(void* parameters_ptr);

////////////////////////////////////////////////////////////////
// Optimizer
////////////////////////////////////////////////////////////////

KPY_API void*     CreateOptimizer(void* session_ptr, void* parameters_ptr, PyObject* optimizer_name_str, PyObject* args_dict);
KPY_API bool      DeleteOptimizer(void* optimizer_ptr);
KPY_API void      OptimizerSetup(void* optimizer_ptr, PyObject* args_dict_str);
KPY_API bool      OptimizerZeroGrad(void* optimizer_ptr);
KPY_API bool      OptimizerStep(void* optimizer_ptr);

////////////////////////////////////////////////////////////////
// Tensor
////////////////////////////////////////////////////////////////

KPY_API void*     CreateEmptyTensor(void);
KPY_API void*     CreateTensor(void* session_ptr, PyObject* shape_tuple, PyObject* type_str, PyObject* init);
KPY_API void*     CreateTensorFromNumPy(void* session_ptr, void* numpy_ptr, PyObject* numpy_type_int, PyObject* numpy_shape_list);
KPY_API bool      DeleteTensor(void* tensor_ptr);

KPY_API void      ModuleLoadCfgWeight(void* module_ptr, PyObject* cfg_path_ptr, PyObject* weight_path_ptr);
KPY_API void      ModuleSave(void* module_ptr, PyObject* path_ptr);
KPY_API void      ModuleInitParameters(void* module_ptr);
KPY_API void*     TensorToType(void* tensor_ptr, PyObject* type_name_str, PyObject* option_str);
KPY_API void*     TensorIndexedByInt(void* tensor_ptr, PyObject* index_ptr);
KPY_API void*     TensorIndexedByTensor(void* tensor_ptr, void* index_ptr);
KPY_API PyObject* ValueIndexedBySlice(void* tensor_ptr, PyObject* index_int);
KPY_API void*     TensorIndexedBySlice(void* tensor_ptr, PyObject* index_ptr);
KPY_API void      TensorSetElementByTensor(void* tensor_ptr, PyObject* index_ptr, void* src_tensor_ptr);
KPY_API void      TensorSetElementByArray(void* tensor_ptr, PyObject* index_ptr, int64 dataSize, PyObject* value_or_type, void* value_ptr);
KPY_API void      TensorSetElementByValue(void* tensor_ptr, PyObject* index_ptr, PyObject* value_or_type);
KPY_API void      TensorPickupRows(void* tensor_ptr, void* src_tensor_ptr, int64 batch_size, void* idx_ptr);
KPY_API void      TensorCopyIntoRow(void* tensor_ptr, int64 nth, void* src_tensor_ptr);
KPY_API void      TensorSetZero(void* tensor_ptr);
KPY_API void      TensorCopyData(void* tensor_ptr, void* src_tensor_ptr);
KPY_API void      TensorShiftTimestepToRight(void* tensor_ptr, void* src_tensor_ptr, int64 steps);
KPY_API void*     TensorSquare(void* tensor_ptr);
KPY_API void*     TensorSum(void* tensor_ptr);
KPY_API void*     TensorResize(void* tensor_ptr, PyObject* shape_ptr);
KPY_API void      TensorResizeOn(void* tensor_ptr, void* src_ptr);
KPY_API void      TensorTransposeOn(void* tensor_ptr, PyObject* axis1_ptr, PyObject* axis2_ptr);
KPY_API PyObject* GetTensorTypeName(void* tensor_ptr);
KPY_API void      TensorLoadJpegPixels(void* tensor_ptr, PyObject* filepath, PyObject* chn_last, PyObject* transpose, PyObject* code, PyObject* mix);
KPY_API void      DumpTensor(void* tensor_ptr, PyObject* tensor_name_str, PyObject* full_flag);
KPY_API PyObject* GetTensorDump(void* tensor_ptr, PyObject* tensor_name_str, PyObject* full_flag);

KPY_API int64     GetTensorLength(void* tensor_ptr);
KPY_API int64     GetTensorSize(void* tensor_ptr);
KPY_API PyObject* GetTensorShape(void* tensor_ptr);

KPY_API void      CopyNumPyToTensor(void* tensor_ptr, void* numpy_ptr, PyObject* numpy_type_int, PyObject* numpy_shape_list);
KPY_API PyObject* ConvertTensorToNumPy(void* tensor_ptr);
KPY_API float     ConvertTensorToScalar(void* tensor_ptr);  // For 1-size Tensor only
KPY_API void*     GetTensorArgmax(void* tensor_ptr, int64 axis);

KPY_API void      TensorBackward(void* tensor_ptr);
KPY_API void      TensorBackwardWithGrad(void* tensor_ptr, void* tensor_grad_ptr);

KPY_API void*     ApplySigmoidToTensor(void* tensor_ptr);

////////////////////////////////////////////////////////////////
// Utility Service Routines
////////////////////////////////////////////////////////////////
KPY_API PyObject* UtilParseJsonFile(PyObject* filepath_ptr);
KPY_API PyObject* UtilParseJsonlFile(PyObject* filepath_ptr);
KPY_API PyObject* UtilReadFileLines(PyObject* filepath_ptr);
KPY_API PyObject* UtilLoadData(void* session_ptr, PyObject* filepath_ptr);
KPY_API void      UtilSaveData(void* session_ptr, PyObject* term_ptr, PyObject* filepath_ptr);
KPY_API int64     UtilPositiveElementCount(void* tensor_ptr);

////////////////////////////////////////////////////////////////
// AudioSpectrumReader
////////////////////////////////////////////////////////////////
KPY_API void*     CreateAudioSpectrumReader(void* session_ptr, PyObject* args_dict);
KPY_API bool      DeleteAudioSpectrumReader(void* reader_ptr);
KPY_API bool      AudioSpectrumReaderAddFile(void* reader_ptr, PyObject* filepath_str);
KPY_API void*     AudioSpectrumReaderExtractSpectrums(void* reader_ptr);

////////////////////////////////////////////////////////////////
// Sample methods for developing new features
////////////////////////////////////////////////////////////////
/**
* Execute a python function in C/C++.
* @param   a pointer of python function, a PyDict-type function argument.
* @returns NULL if it fails, else a return value of the function.
*/
KPY_API PyObject* ExecutePythonFunction(PyObject* pyobject_callable, PyObject* pyobject_dict);

////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif
