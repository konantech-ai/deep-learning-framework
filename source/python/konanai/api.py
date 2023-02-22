"""K
This file is part of Konantech AI Framework project.
It is subject to the license terms in the LICENSE file found in the top-level directory of this distribution.
"""

"""
Caution!
1) If you are using the virtual environment of 'conda', then DLL search path of python may be disabled.
   - Error message: FileNotFoundError: Could not find module
   - Reference: https://issueexplorer.com/issue/conda/conda/10897
2) In this case, you have to add the system variable CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1,
   change the way the function is called for DLL import.
   - Reference: https://docs.conda.io/projects/conda/en/latest/user-guide/troubleshooting.html
3) Solution 1 (command in conda command line) :
   > set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1
   > conda update -c defaults python
4) Solution 2 (recommended, use this following code, also 'winmode=0' must be included) :
   This function must be called before other routing functions(numpy, etc...).
   Enable to modify DLL search path of CONDA
   os.putenv('CONDA_DLL_SEARCH_MODIFICATION_ENABLE','1')
   module = ctypes.CDLL(lib_name, mode=ctypes.DEFAULT_MODE, handle=None, use_errno=True, use_last_error=False, winmode=0)
"""

# Enable to modify DLL search path of CONDA
# This function must be called before other routing functions(numpy, etc...).
import os
os.putenv('CONDA_DLL_SEARCH_MODIFICATION_ENABLE','1')

# Set library paths based on the Environment variables.
PACKAGE_INSTALL_TYPE_GENERAL_USER = 1 # Install command : 'pip install .'
PACKAGE_INSTALL_TYPE_DEVELOPER    = 2 # Install command : 'pip install -e .' (This installation type requires the environment variable 'KONANAI_PATH'.)
PACKAGE_INSTALL_TYPE = 2
if PACKAGE_INSTALL_TYPE == PACKAGE_INSTALL_TYPE_DEVELOPER:
    os.add_dll_directory(os.path.abspath(os.environ['KONANAI_PATH']+"/out/Release/"))

# Import other modules
import sys
import traceback
import ctypes, _ctypes
import numpy as np
import struct as st
from typing import Optional, Callable

# Check the current OS
lib_name = "konanai"
lib_ext = None
KAI_LIB = None
if sys.platform == "win32":
    lib_name = "konanai_python"
    lib_ext = ".dll" # for Windows
elif sys.platform == "linux" or sys.platform == "linux2":
    lib_name = "libkonanai_python"
    lib_ext = ".so" # for POSIX

# Call Library
from distutils.sysconfig import get_python_lib
if PACKAGE_INSTALL_TYPE == PACKAGE_INSTALL_TYPE_GENERAL_USER:
    lib_name = os.path.join( "konanai" , lib_name )
    lib_name = os.path.join( get_python_lib() , lib_name) # get_python_lib() : ~\\[INSTALLED_PACKAGE_NAME]\\Lib\\site-packages
if KAI_LIB == None:
    try:
        # If you are using the virtual environment of 'conda', then this function is unavailable.
        # Error message = FileNotFoundError: Could not find module
        KAI_LIB = ctypes.cdll.LoadLibrary(lib_name + lib_ext)
    except Exception as e:
        # Caution!
        # If the system variable CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1 is not defined before all path-specifying functions are called,
        # then these following functions also return the same error as the above function.
        try:
            KAI_LIB = ctypes.CDLL(lib_name + lib_ext, mode=ctypes.DEFAULT_MODE, handle=None, use_errno=True, use_last_error=False, winmode=0)
        except Exception as e:
            print(e)
            print("Error : Module '%s' load fail from File Path" % (lib_name + lib_ext))
            #print("Note : This error also occurs when other packages such as numpy are imported before this API.")
            exit(1)

# Global variables
TENSOR_TYPE_FLOAT32 = 0
TENSOR_TYPE_INT32   = 1
TENSOR_TYPE_INT64   = 2
TENSOR_TYPE_UINT8   = 3
TENSOR_TYPE_BOOL8   = 4
TENSOR_TYPE_FLOAT64 = 5

# Function Define
if KAI_LIB != None:

    def free_library():
        global KAI_LIB
        global lib_name
        global lib_ext

        if (KAI_LIB == None):
            print("Module '%s' is not loaded yet.\n" % (lib_name + lib_ext))
            return False
        else:
            if sys.platform == "win32":
                # for Windows
                _ctypes.FreeLibrary(KAI_LIB._handle)
            elif sys.platform == "linux" or sys.platform == "linux2":
                # for POSIX
                _ctypes.dlclose(KAI_LIB._handle)
                print("Module '%s' has been finalized.\n" % (lib_name + lib_ext))
            return True

    def exception_handler(e: Exception):
        #print(e.__context__)
        flag_print = False
        flag_exit = True

        if ( e.__context__ != None ):
            print("\nPython Exception (Error Code : {})".format(e.__context__))

        err_str = traceback.extract_stack()
        for idx in range( len(err_str) - 2 ) :
            print("  {}, in \"{}\":{}".format( err_str[idx].line, err_str[idx].filename, err_str[idx].lineno,));
            #print(err_str[idx].filename) # file name
            #print(err_str[idx].lineno) # file line number
            #print(err_str[idx].line) # file line string
        if (flag_exit): sys.exit(0)


    # TensorDict 인자를 전달하기 위해서는 python 단에서 아래 처리를 통해 핸들을 추출한 후 cpp 단에서 engine용 텐서로 재구성하도록 한다.
    def send_tensor_dict(x):
        xs = {}
        for key in x.keys():
            xs[key] = x[key].get_core()
        return xs;

    # TensorDict 결과값을 얻어오기 위해서는 cpp 단에서 핸들을 추출한 후 아래 처리를 통해 python 단에서 python용 텐서로 재구성하도록 한다.
    # 그런데 api.py에서 tensor.py를 접근하기 곤란해 Tensor 객체 생성이 곤란하므로 실제 처리는 상부 호출단에서 하기로 한다.
    #def recv_tensor_dict(x):
    #    xs = {}
    #    for key in x.keys():
    #        xs[key] = Tensor(x[key])
    #    return xs;

    ################################################################
    # Session and Device
    ################################################################

    lib_OpenSession = KAI_LIB.OpenSession
    lib_OpenSession.restype = ctypes.c_void_p
    lib_OpenSession.argtypes = [ctypes.py_object, ctypes.py_object]
    def OpenSession(server_url, client_url) -> ctypes.c_void_p:
        try :
            return lib_OpenSession(server_url, client_url)
        except Exception as e : exception_handler(e)

    lib_CloseSession = KAI_LIB.CloseSession
    lib_CloseSession.restype = ctypes.c_bool
    lib_CloseSession.argtypes = [ctypes.c_void_p]
    def CloseSession(session_ptr: ctypes.c_void_p) -> bool:
        try :
            return lib_CloseSession(session_ptr)
        except Exception as e : exception_handler(e)

    lib_SrandTime = KAI_LIB.SrandTime
    lib_SrandTime.restype = None
    lib_SrandTime.argtypes = []
    def SrandTime() -> None:
        try :
            lib_SrandTime()
        except Exception as e : exception_handler(e)

    lib_SrandSeed = KAI_LIB.SrandSeed
    lib_SrandSeed.restype = None
    lib_SrandSeed.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def SrandSeed(session_ptr: ctypes.c_void_p, seed: int) -> None:
        try :
            lib_SrandSeed(session_ptr, seed)
        except Exception as e : exception_handler(e)

    lib_IsCUDAAvailable = KAI_LIB.IsCUDAAvailable
    lib_IsCUDAAvailable.restype = ctypes.c_bool
    lib_IsCUDAAvailable.argtypes = [ctypes.c_void_p]
    def IsCUDAAvailable(session_ptr: ctypes.c_void_p) -> bool:
        try:
            return lib_IsCUDAAvailable(session_ptr)
        except Exception as e : exception_handler(e)

    lib_GetCUDADeviceCount = KAI_LIB.GetCUDADeviceCount
    lib_GetCUDADeviceCount.restype = ctypes.c_int64
    lib_GetCUDADeviceCount.argtypes = [ctypes.c_void_p]
    def GetCUDADeviceCount(session_ptr: ctypes.c_void_p) -> int:
        try:
            return lib_GetCUDADeviceCount(session_ptr)
        except Exception as e : exception_handler(e)

    lib_SetNoGrad = KAI_LIB.SetNoGrad
    lib_SetNoGrad.restype = None
    lib_SetNoGrad.argtypes = [ctypes.c_void_p]
    def SetNoGrad(session_ptr: ctypes.c_void_p) -> None:
        try:
            lib_SetNoGrad(session_ptr)
        except Exception as e : exception_handler(e)

    lib_UnsetNoGrad = KAI_LIB.UnsetNoGrad
    lib_UnsetNoGrad.restype = None
    lib_UnsetNoGrad.argtypes = [ctypes.c_void_p]
    def UnsetNoGrad(session_ptr: ctypes.c_void_p) -> None:
        try:
            lib_UnsetNoGrad(session_ptr)  
        except Exception as e : exception_handler(e)

    ################################################################
    # Module
    ################################################################

    lib_CreateModule = KAI_LIB.CreateModule
    lib_CreateModule.restype = ctypes.c_void_p
    lib_CreateModule.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def CreateModule(session_ptr: ctypes.c_void_p, module_name: str, args: Optional[dict] = None) -> ctypes.c_void_p:
        try:
            if args is None:
                args = {}
            return lib_CreateModule(session_ptr, module_name, args)
        except Exception as e : exception_handler(e)

    lib_DeleteModule = KAI_LIB.DeleteModule
    lib_DeleteModule.restype = ctypes.c_bool
    lib_DeleteModule.argtypes = [ctypes.c_void_p]
    def DeleteModule(module_ptr: ctypes.c_void_p) -> bool:
        try:
            return lib_DeleteModule(module_ptr)
        except Exception as e : exception_handler(e)

    lib_CreateContainer = KAI_LIB.CreateContainer
    lib_CreateContainer.restype = ctypes.c_void_p
    lib_CreateContainer.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def CreateContainer(session_ptr: ctypes.c_void_p, container_name: str, module_ptr_list: list, args: Optional[dict] = None) -> ctypes.c_void_p:
        try :
            if args is None:
                args = {}
            return lib_CreateContainer(session_ptr, container_name, module_ptr_list, args)
        except Exception as e : exception_handler(e)

    lib_DeleteContainer = KAI_LIB.DeleteContainer
    lib_DeleteContainer.restype = ctypes.c_bool
    lib_DeleteContainer.argtypes = [ctypes.c_void_p]
    # This API is equivalent to DeleteModule().
    def DeleteContainer(container_module_ptr: ctypes.c_void_p) -> bool:
        try:
            return lib_DeleteContainer(container_module_ptr)
        except Exception as e : exception_handler(e)

    lib_RegisterMacro = KAI_LIB.RegisterMacro
    lib_RegisterMacro.restype = None
    lib_RegisterMacro.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.c_void_p, ctypes.py_object]
    def RegisterMacro(session_ptr: ctypes.c_void_p, macro_name: str, module_ptr: ctypes.c_void_p, args: Optional[dict] = None) -> None:
        try :
            if args is None:
                args = {}
            lib_RegisterMacro(session_ptr, macro_name, module_ptr, args)
        except Exception as e : exception_handler(e)

    lib_CreateMacro = KAI_LIB.CreateMacro
    lib_CreateMacro.restype = ctypes.c_void_p
    lib_CreateMacro.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def CreateMacro(session_ptr: ctypes.c_void_p, macro_name: str, args: Optional[dict] = None) -> ctypes.c_void_p:
        try :
            if args is None:
                args = {}
            return lib_CreateMacro(session_ptr, macro_name, args)
        except Exception as e : exception_handler(e)

    lib_ExpandModule = KAI_LIB.ExpandModule
    lib_ExpandModule.restype = ctypes.c_void_p
    lib_ExpandModule.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def ExpandModule(module_ptr: ctypes.c_void_p, data_shape: list or tuple, args: Optional[dict] = None) -> ctypes.c_void_p:
        try :
            return lib_ExpandModule(module_ptr, data_shape, args)
        except Exception as e : exception_handler(e)

    lib_SetModuleToDevice = KAI_LIB.SetModuleToDevice
    lib_SetModuleToDevice.restype = ctypes.c_void_p
    lib_SetModuleToDevice.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def SetModuleToDevice(module_ptr: ctypes.c_void_p, device_name: str) -> ctypes.c_void_p:
        try :
            return lib_SetModuleToDevice(module_ptr, device_name)
        except Exception as e : exception_handler(e)

    lib_GetModuleShape = KAI_LIB.GetModuleShape
    lib_GetModuleShape.restype = ctypes.py_object
    lib_GetModuleShape.argtypes = [ctypes.c_void_p]
    def GetModuleShape(module_ptr: ctypes.c_void_p) -> str:
        try :
            return lib_GetModuleShape(module_ptr)
        except Exception as e : exception_handler(e)

    lib_ModuleTrain = KAI_LIB.ModuleTrain
    lib_ModuleTrain.restype = None
    lib_ModuleTrain.argtypes = [ctypes.c_void_p]
    def ModuleTrain(module_ptr: ctypes.c_void_p) -> None:
        """ Set the module to training mode. """
        try :
            lib_ModuleTrain(module_ptr)
        except Exception as e : exception_handler(e)

    lib_ModuleEval = KAI_LIB.ModuleEval
    lib_ModuleEval.restype = None
    lib_ModuleEval.argtypes = [ctypes.c_void_p]
    def ModuleEval(module_ptr: ctypes.c_void_p) -> None:
        """ Set the module to evaluation mode. """
        try :
            lib_ModuleEval(module_ptr)
        except Exception as e : exception_handler(e)
        
    lib_ModuleAppendChild = KAI_LIB.ModuleAppendChild
    lib_ModuleAppendChild.restype = None
    lib_ModuleAppendChild.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    def ModuleAppendChild(module_ptr: ctypes.c_void_p, child_ptr:ctypes.c_void_p) -> None:
        try :
            lib_ModuleAppendChild(module_ptr, child_ptr)
        except Exception as e : exception_handler(e)

    lib_ModuleCall = KAI_LIB.ModuleCall
    lib_ModuleCall.restype = ctypes.c_void_p
    lib_ModuleCall.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    def ModuleCall(module_ptr: ctypes.c_void_p, tensor_x_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        """ ModuleCall an answer from input x. This function returns Tensor. """
        try :
            return lib_ModuleCall(module_ptr, tensor_x_ptr)
        except Exception as e : exception_handler(e)

    
    lib_ModuleCallDict = KAI_LIB.ModuleCallDict
    lib_ModuleCallDict.restype = ctypes.py_object
    lib_ModuleCallDict.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def ModuleCallDict(module_ptr: ctypes.c_void_p, tensors: ctypes.py_object) -> ctypes.py_object:
        """ ModuleCallDict an answer from input x. This function returns Tensor. """
        try :
            return lib_ModuleCallDict(module_ptr, send_tensor_dict(tensors))
        except Exception as e : exception_handler(e)

    lib_ModulePredict = KAI_LIB.ModulePredict
    lib_ModulePredict.restype = ctypes.c_void_p
    lib_ModulePredict.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    def ModulePredict(module_ptr: ctypes.c_void_p, tensor_x_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        """ Predict an answer from input x. This function returns Tensor. """
        try :
            return lib_ModulePredict(module_ptr, tensor_x_ptr)
        except Exception as e : exception_handler(e)

    lib_ModulePredictDict = KAI_LIB.ModulePredictDict
    lib_ModulePredictDict.restype = ctypes.py_object
    lib_ModulePredictDict.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def ModulePredictDict(module_ptr: ctypes.c_void_p, tensors: ctypes.py_object) -> ctypes.py_object:
        """ ModulePredictDict an answer from input x. This function returns Tensor. """
        try :
            return lib_ModulePredictDict(module_ptr, send_tensor_dict(tensors))
        except Exception as e : exception_handler(e)

    lib_ModuleNthChild = KAI_LIB.ModuleNthChild
    lib_ModuleNthChild.restype = ctypes.c_void_p
    lib_ModuleNthChild.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def ModuleNthChild(module_ptr: ctypes.c_void_p, nth: ctypes.py_object) -> ctypes.py_object:
        """ ModuleNthChild an answer from input x. This function returns Tensor. """
        try :
            return lib_ModuleNthChild(module_ptr, nth)
        except Exception as e : exception_handler(e)

    lib_ModuleFetchChild = KAI_LIB.ModuleFetchChild
    lib_ModuleFetchChild.restype = ctypes.c_void_p
    lib_ModuleFetchChild.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def ModuleFetchChild(module_ptr: ctypes.c_void_p, name_str: ctypes.py_object) -> ctypes.py_object:
        """ ModuleFetchChild an answer from input x. This function returns Tensor. """
        try :
            return lib_ModuleFetchChild(module_ptr, name_str)
        except Exception as e : exception_handler(e)

    ################################################################
    # Loss
    ################################################################

    lib_CreateLoss = KAI_LIB.CreateLoss
    lib_CreateLoss.restype = ctypes.c_void_p
    lib_CreateLoss.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def CreateLoss(session_ptr: ctypes.c_void_p, loss_name: str, est: str, ans: str, args: Optional[dict] = {}) -> ctypes.c_void_p:
        try:
            if args is None:
                args = {}
            return lib_CreateLoss(session_ptr, loss_name, est, ans, args)
        except Exception as e : exception_handler(e)

    lib_CreateMultipleLoss = KAI_LIB.CreateMultipleLoss
    lib_CreateMultipleLoss.restype = ctypes.c_void_p
    lib_CreateMultipleLoss.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def CreateMultipleLoss(session_ptr: ctypes.c_void_p, children_ptrs: dict) -> ctypes.c_void_p:
        try:
            return lib_CreateMultipleLoss(session_ptr, children_ptrs)
        except Exception as e : exception_handler(e)

    lib_CreateCustomLoss = KAI_LIB.CreateCustomLoss
    lib_CreateCustomLoss.restype = ctypes.c_void_p
    lib_CreateCustomLoss.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def CreateCustomLoss(session_ptr: ctypes.c_void_p, loss_terms:dict, static_tensors:dict or None, args: Optional[dict] = {}) -> ctypes.c_void_p:
        try:
            return lib_CreateCustomLoss(session_ptr, loss_terms, send_tensor_dict(static_tensors), args)
        except Exception as e : exception_handler(e)

    lib_DeleteLoss = KAI_LIB.DeleteLoss
    lib_DeleteLoss.restype = ctypes.c_bool
    lib_DeleteLoss.argtypes = [ctypes.c_void_p]
    def DeleteLoss(loss_ptr: ctypes.c_void_p) -> bool:
        try:
            return lib_DeleteLoss(loss_ptr)
        except Exception as e : exception_handler(e)

    lib_LossEvaluate = KAI_LIB.LossEvaluate
    lib_LossEvaluate.restype = ctypes.c_void_p
    lib_LossEvaluate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.py_object]
    def LossEvaluate(loss_ptr: ctypes.c_void_p, tensor_pred_ptr: ctypes.c_void_p, tensor_y_ptr: ctypes.c_void_p, download_all:bool) -> ctypes.c_void_p:
        """ LossEvaluate the predicted answer against y. This function returns Loss. """
        try :
            return lib_LossEvaluate(loss_ptr, tensor_pred_ptr, tensor_y_ptr, download_all)
        except Exception as e : exception_handler(e)
        
    lib_LossEvaluateDict = KAI_LIB.LossEvaluateDict
    lib_LossEvaluateDict.restype = ctypes.py_object
    lib_LossEvaluateDict.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def LossEvaluateDict(loss_ptr: ctypes.c_void_p, pred_dict: ctypes.py_object, y_dict: ctypes.py_object, download_all:bool) -> ctypes.py_object:
        """ LossEvaluateDict the predicted answer against y. This function returns Loss. """
        try :
            return lib_LossEvaluateDict(loss_ptr, send_tensor_dict(pred_dict), send_tensor_dict(y_dict), download_all)
        except Exception as e : exception_handler(e)

    lib_LossEvalAccuracy = KAI_LIB.LossEvalAccuracy
    lib_LossEvalAccuracy.restype = ctypes.c_void_p
    lib_LossEvalAccuracy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    def LossEvalAccuracy(loss_ptr: ctypes.c_void_p, tensor_pred_ptr: ctypes.c_void_p, tensor_y_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        """ LossEvalAccuracy the predicted answer against y. This function returns Loss. """
        try :
            return lib_LossEvalAccuracy(loss_ptr, tensor_pred_ptr, tensor_y_ptr)
        except Exception as e : exception_handler(e)


    lib_LossEvalAccuracyDict = KAI_LIB.LossEvalAccuracyDict
    lib_LossEvalAccuracyDict.restype = ctypes.py_object
    lib_LossEvalAccuracyDict.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def LossEvalAccuracyDict(loss_ptr: ctypes.c_void_p, pred_dict: ctypes.py_object, y_dict: ctypes.py_object) -> ctypes.py_object:
        """ LossEvalAccuracyDict the predicted answer against y. This function returns Loss. """
        try :
            return lib_LossEvalAccuracyDict(loss_ptr, send_tensor_dict(pred_dict), send_tensor_dict(y_dict))
        except Exception as e : exception_handler(e)
        
    lib_LossBackward = KAI_LIB.LossBackward
    lib_LossBackward.restype = None
    lib_LossBackward.argtypes = [ctypes.c_void_p]
    def LossBackward(Loss_ptr: ctypes.c_void_p) -> None:
        try :
            lib_LossBackward(Loss_ptr)
        except Exception as e : exception_handler(e)
        
    ################################################################
    # Metric
    ################################################################

    lib_CreateFormulaMetric = KAI_LIB.CreateFormulaMetric
    lib_CreateFormulaMetric.restype = ctypes.c_void_p
    lib_CreateFormulaMetric.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def CreateFormulaMetric(session_ptr: ctypes.c_void_p, metric_name: str, formula: str, args: Optional[dict] = {}) -> ctypes.c_void_p:
        try:
            if args is None:
                args = {}
            return lib_CreateFormulaMetric(session_ptr, metric_name, formula, args)
        except Exception as e : exception_handler(e)

    lib_DeleteMetric = KAI_LIB.DeleteMetric
    lib_DeleteMetric.restype = ctypes.c_bool
    lib_DeleteMetric.argtypes = [ctypes.c_void_p]
    def DeleteMetric(metric_ptr: ctypes.c_void_p) -> bool:
        try:
            return lib_DeleteMetric(metric_ptr)
        except Exception as e : exception_handler(e)

    lib_CreateMultipleMetric = KAI_LIB.CreateMultipleMetric
    lib_CreateMultipleMetric.restype = ctypes.c_void_p
    lib_CreateMultipleMetric.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def CreateMultipleMetric(session_ptr: ctypes.c_void_p, children_ptrs: dict, args: Optional[dict] = {}) -> ctypes.c_void_p:
        try:
            return lib_CreateMultipleMetric(session_ptr, children_ptrs, args)
        except Exception as e : exception_handler(e)

    lib_CreateCustomMetric = KAI_LIB.CreateCustomMetric
    lib_CreateCustomMetric.restype = ctypes.c_void_p
    lib_CreateCustomMetric.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def CreateCustomMetric(session_ptr: ctypes.c_void_p, metric_terms:dict, static_tensors:dict or None, args: Optional[dict] = {}) -> ctypes.c_void_p:
        try:
            return lib_CreateCustomMetric(session_ptr, metric_terms, send_tensor_dict(static_tensors), args)
        except Exception as e : exception_handler(e)

    lib_MetricEvaluate = KAI_LIB.MetricEvaluate
    lib_MetricEvaluate.restype = ctypes.c_void_p
    lib_MetricEvaluate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    def MetricEvaluate(metric_ptr: ctypes.c_void_p, tensor_pred_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        """ MetricEvaluate the predicted answer against y. This function returns Metric. """
        try :
            return lib_MetricEvaluate(metric_ptr, tensor_pred_ptr)
        except Exception as e : exception_handler(e)

    lib_MetricEvaluateDict = KAI_LIB.MetricEvaluateDict
    lib_MetricEvaluateDict.restype = ctypes.py_object
    lib_MetricEvaluateDict.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def MetricEvaluateDict(metric_ptr: ctypes.c_void_p, pred_dict: ctypes.py_object) -> ctypes.py_object:
        """ MetricEvaluateDict the predicted answer against y. This function returns Metric. """
        try :
            return lib_MetricEvaluateDict(metric_ptr, send_tensor_dict(pred_dict))
        except Exception as e : exception_handler(e)
        
    ################################################################
    # Parameters
    ################################################################

    lib_CreateParameters = KAI_LIB.CreateParameters
    lib_CreateParameters.restype = ctypes.c_void_p
    lib_CreateParameters.argtypes = [ctypes.c_void_p]
    def CreateParameters(module_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        try :
            return lib_CreateParameters(module_ptr)
        except Exception as e : exception_handler(e)

    lib_DeleteParameters = KAI_LIB.DeleteParameters
    lib_DeleteParameters.restype = ctypes.c_bool
    lib_DeleteParameters.argtypes = [ctypes.c_void_p]
    def DeleteParameters(parameters_ptr: ctypes.c_void_p) -> bool:
        try :
            return lib_DeleteParameters(parameters_ptr)
        except Exception as e : exception_handler(e)

    lib_GetParametersDump = KAI_LIB.GetParametersDump
    lib_GetParametersDump.restype = ctypes.py_object
    lib_GetParametersDump.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    def GetParametersDump(parameters_ptr: ctypes.c_void_p, is_full: bool = False) -> str:
        try :
            return lib_GetParametersDump(parameters_ptr, is_full)
        except Exception as e : exception_handler(e)

    lib_GetParameterWeightDict = KAI_LIB.GetParameterWeightDict
    lib_GetParameterWeightDict.restype = ctypes.py_object
    lib_GetParameterWeightDict.argtypes = [ctypes.c_void_p]
    def GetParameterWeightDict(parameters_ptr: ctypes.c_void_p) -> dict:
        try :
            return lib_GetParameterWeightDict(parameters_ptr)
        except Exception as e : exception_handler(e)

    lib_GetParameterGradientDict = KAI_LIB.GetParameterGradientDict
    lib_GetParameterGradientDict.restype = ctypes.py_object
    lib_GetParameterGradientDict.argtypes = [ctypes.c_void_p]
    def GetParameterGradientDict(parameters_ptr: ctypes.c_void_p) -> dict:
        try :
            return lib_GetParameterGradientDict(parameters_ptr)
        except Exception as e : exception_handler(e)

    ################################################################
    # Optimizer
    ################################################################

    lib_CreateOptimizer = KAI_LIB.CreateOptimizer
    lib_CreateOptimizer.restype = ctypes.c_void_p
    lib_CreateOptimizer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def CreateOptimizer(session_ptr: ctypes.c_void_p, parameters_ptr: ctypes.c_void_p, optimizer_name: str, args: Optional[dict] = {}) -> ctypes.c_void_p:
        try :
            return lib_CreateOptimizer(session_ptr, parameters_ptr, optimizer_name, args)
        except Exception as e : exception_handler(e)

    lib_DeleteOptimizer = KAI_LIB.DeleteOptimizer
    lib_DeleteOptimizer.restype = ctypes.c_bool
    lib_DeleteOptimizer.argtypes = [ctypes.c_void_p]
    def DeleteOptimizer(optimizer_ptr: ctypes.c_void_p) -> bool:
        try :
            return lib_DeleteOptimizer(optimizer_ptr)
        except Exception as e : exception_handler(e)

    lib_OptimizerSetup = KAI_LIB.OptimizerSetup
    lib_OptimizerSetup.restype = None
    lib_OptimizerSetup.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def OptimizerSetup(optimizer_ptr: ctypes.c_void_p, args_dict:dict) -> None:
        try :
            return lib_OptimizerSetup(optimizer_ptr, args_dict)
        except Exception as e : exception_handler(e)

    lib_OptimizerZeroGrad = KAI_LIB.OptimizerZeroGrad
    lib_OptimizerZeroGrad.restype = ctypes.c_bool
    lib_OptimizerZeroGrad.argtypes = [ctypes.c_void_p]
    def OptimizerZeroGrad(optimizer_ptr: ctypes.c_void_p) -> bool:
        try :
            return lib_OptimizerZeroGrad(optimizer_ptr)
        except Exception as e : exception_handler(e)

    lib_OptimizerStep = KAI_LIB.OptimizerStep
    lib_OptimizerStep.restype = ctypes.c_bool
    lib_OptimizerStep.argtypes = [ctypes.c_void_p]
    def OptimizerStep(optimizer_ptr: ctypes.c_void_p) -> bool:
        try :
            return lib_OptimizerStep(optimizer_ptr)
        except Exception as e : exception_handler(e)

    ################################################################
    # Utility Service Routines
    ################################################################

    lib_UtilParseJsonFile = KAI_LIB.UtilParseJsonFile
    lib_UtilParseJsonFile.restype = ctypes.py_object
    lib_UtilParseJsonFile.argtypes = [ctypes.py_object]
    def UtilParseJsonFile(filepath: ctypes.py_object) -> dict or list:
        try :
            return lib_UtilParseJsonFile(filepath)
        except Exception as e : exception_handler(e)

    lib_UtilParseJsonlFile = KAI_LIB.UtilParseJsonlFile
    lib_UtilParseJsonlFile.restype = ctypes.py_object
    lib_UtilParseJsonlFile.argtypes = [ctypes.py_object]
    def UtilParseJsonlFile(filepath: ctypes.py_object) -> list:
        try :
            return lib_UtilParseJsonlFile(filepath)
        except Exception as e : exception_handler(e)

    lib_UtilReadFileLines = KAI_LIB.UtilReadFileLines
    lib_UtilReadFileLines.restype = ctypes.py_object
    lib_UtilReadFileLines.argtypes = [ctypes.py_object]
    def UtilReadFileLines(filepath: ctypes.py_object) -> list:
        try :
            return lib_UtilReadFileLines(filepath)
        except Exception as e : exception_handler(e)

    lib_UtilLoadData = KAI_LIB.UtilLoadData
    lib_UtilLoadData.restype = ctypes.py_object
    lib_UtilLoadData.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def UtilLoadData(session_ptr: ctypes.c_void_p, filepath: ctypes.py_object) -> dict:
        try :
            return lib_UtilLoadData(session_ptr, filepath)
        except Exception as e : exception_handler(e)

    lib_UtilSaveData = KAI_LIB.UtilSaveData
    lib_UtilSaveData.restype = None
    lib_UtilSaveData.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def UtilSaveData(session_ptr: ctypes.c_void_p, term:ctypes.py_object, filepath: ctypes.py_object) -> None:
        try :
            lib_UtilSaveData(session_ptr, term, filepath)
        except Exception as e : exception_handler(e)

    lib_UtilPositiveElementCount = KAI_LIB.UtilPositiveElementCount
    lib_UtilPositiveElementCount.restype = int
    lib_UtilPositiveElementCount.argtypes = [ctypes.c_void_p]
    def UtilPositiveElementCount(tensor_ptr: ctypes.c_void_p) -> int:
        try :
            return lib_UtilPositiveElementCount(tensor_ptr)
        except Exception as e : exception_handler(e)

    ################################################################
    # AudioSpectrumReader
    ################################################################

    lib_CreateAudioSpectrumReader = KAI_LIB.CreateAudioSpectrumReader
    lib_CreateAudioSpectrumReader.restype = ctypes.c_void_p
    lib_CreateAudioSpectrumReader.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def CreateAudioSpectrumReader(session_ptr: ctypes.c_void_p, args_dict:ctypes.py_object) -> ctypes.c_void_p:
        try :
            return lib_CreateAudioSpectrumReader(session_ptr, args_dict)
        except Exception as e : exception_handler(e)

    lib_DeleteAudioSpectrumReader = KAI_LIB.DeleteAudioSpectrumReader
    lib_DeleteAudioSpectrumReader.restype = ctypes.c_bool
    lib_DeleteAudioSpectrumReader.argtypes = [ctypes.c_void_p]
    def DeleteAudioSpectrumReader(tensor_ptr: ctypes.c_void_p) -> bool:
        try :
            return lib_DeleteAudioSpectrumReader(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_AudioSpectrumReaderAddFile = KAI_LIB.AudioSpectrumReaderAddFile
    lib_AudioSpectrumReaderAddFile.restype = ctypes.c_bool
    lib_AudioSpectrumReaderAddFile.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def AudioSpectrumReaderAddFile(tensor_ptr: ctypes.c_void_p, filepath_str:ctypes.py_object) -> bool:
        try :
            return lib_AudioSpectrumReaderAddFile(tensor_ptr, filepath_str)
        except Exception as e : exception_handler(e)

    lib_AudioSpectrumReaderExtractSpectrums = KAI_LIB.AudioSpectrumReaderExtractSpectrums
    lib_AudioSpectrumReaderExtractSpectrums.restype = ctypes.c_void_p
    lib_AudioSpectrumReaderExtractSpectrums.argtypes = [ctypes.c_void_p]
    def AudioSpectrumReaderExtractSpectrums(tensor_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        try :
            return lib_AudioSpectrumReaderExtractSpectrums(tensor_ptr)
        except Exception as e : exception_handler(e)

    ################################################################
    # Tensor
    ################################################################

    lib_CreateEmptyTensor = KAI_LIB.CreateEmptyTensor
    lib_CreateEmptyTensor.restype = ctypes.c_void_p
    lib_CreateEmptyTensor.argtypes = []
    def CreateEmptyTensor() -> ctypes.c_void_p:
        try :
            return lib_CreateEmptyTensor()
        except Exception as e : exception_handler(e)

    lib_CreateTensor = KAI_LIB.CreateTensor
    lib_CreateTensor.restype = ctypes.c_void_p
    lib_CreateTensor.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def CreateTensor(session_ptr: ctypes.c_void_p, shape:ctypes.py_object, type:ctypes.py_object, init:ctypes.py_object) -> ctypes.c_void_p:
        try :
            return lib_CreateTensor(session_ptr, shape, type, init)
        except Exception as e : exception_handler(e)

    lib_CreateTensorFromNumPy = KAI_LIB.CreateTensorFromNumPy
    lib_CreateTensorFromNumPy.restype = ctypes.c_void_p
    lib_CreateTensorFromNumPy.argtypes = [ctypes.c_void_p, np.ctypeslib.ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def CreateTensorFromNumPy(session_ptr: ctypes.c_void_p, numpy_obj: np.ndarray) -> ctypes.c_void_p:
        global TENSOR_TYPE_FLOAT32, TENSOR_TYPE_INT32, TENSOR_TYPE_INT64, TENSOR_TYPE_UINT8, TENSOR_TYPE_BOOL8, TENSOR_TYPE_FLOAT64
        try :
            if not isinstance(numpy_obj, np.ndarray):
                raise Exception("bad argument: Not a numpy object")

            if (numpy_obj.dtype == "float32"): numpy_type = TENSOR_TYPE_FLOAT32
            elif (numpy_obj.dtype == "int32"): numpy_type = TENSOR_TYPE_INT32
            elif (numpy_obj.dtype == "int64"): numpy_type = TENSOR_TYPE_INT64
            elif (numpy_obj.dtype == "uint8"): numpy_type = TENSOR_TYPE_UINT8
            elif (numpy_obj.dtype == "bool8"): numpy_type = TENSOR_TYPE_BOOL8
            elif (numpy_obj.dtype == "float64"): numpy_type = TENSOR_TYPE_FLOAT64
            else: raise Exception("exception : Unsupported NumPy data type")

            numpy_shape = []
            for idx in range(len(numpy_obj.shape)):
                numpy_shape.append(numpy_obj.shape[idx])
            '''
            print('numpy_obj.shape', numpy_obj.shape)
            print('numpy_obj.dtype', numpy_obj.dtype)
            print('numpy_obj.dim', numpy_obj.dim)
            print('numpy_obj.ndim', numpy_obj.ndim)
            numpy_shape = []
            for idx in range(numpy_obj.ndim):
                numpy_shape.append(numpy_obj.shape[idx])
            '''

            numpy_ptr = np.ctypeslib.as_ctypes(numpy_obj)
            return lib_CreateTensorFromNumPy(session_ptr, numpy_ptr, numpy_type, numpy_shape)
        except Exception as e : exception_handler(e)

    lib_DeleteTensor = KAI_LIB.DeleteTensor
    lib_DeleteTensor.restype = ctypes.c_bool
    lib_DeleteTensor.argtypes = [ctypes.c_void_p]
    def DeleteTensor(tensor_ptr: ctypes.c_void_p) -> bool:
        try :
            return lib_DeleteTensor(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_DumpTensor = KAI_LIB.DumpTensor
    lib_DumpTensor.restype = None
    lib_DumpTensor.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def DumpTensor(tensor_ptr: ctypes.c_void_p, tensor_name: str, is_full: bool = False) -> None:
        try :
            lib_DumpTensor(tensor_ptr, tensor_name, is_full)
        except Exception as e : exception_handler(e)

    lib_GetTensorDump = KAI_LIB.GetTensorDump
    lib_GetTensorDump.restype = ctypes.py_object
    lib_GetTensorDump.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def GetTensorDump(tensor_ptr: ctypes.c_void_p, tensor_name: str, is_full: bool = False) -> str:
        try :
            return lib_GetTensorDump(tensor_ptr, tensor_name, is_full)
        except Exception as e : exception_handler(e)

    lib_GetTensorLength = KAI_LIB.GetTensorLength
    lib_GetTensorLength.restype = ctypes.c_int64
    lib_GetTensorLength.argtypes = [ctypes.c_void_p]
    def GetTensorLength(tensor_ptr: ctypes.c_void_p) -> int:
        try :
            return lib_GetTensorLength(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_GetTensorSize = KAI_LIB.GetTensorSize
    lib_GetTensorSize.restype = int
    lib_GetTensorSize.argtypes = [ctypes.c_void_p]
    def GetTensorSize(tensor_ptr: ctypes.c_void_p) -> int:
        try :
            size = lib_GetTensorSize(tensor_ptr)
            return size
        except Exception as e : exception_handler(e)

    lib_GetTensorShape = KAI_LIB.GetTensorShape
    lib_GetTensorShape.restype = ctypes.py_object
    lib_GetTensorShape.argtypes = [ctypes.c_void_p]
    def GetTensorShape(tensor_ptr: ctypes.c_void_p) -> tuple:
        try :
            return lib_GetTensorShape(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_CopyNumPyToTensor = KAI_LIB.CopyNumPyToTensor
    lib_CopyNumPyToTensor.restype = None
    lib_CopyNumPyToTensor.argtypes = [ctypes.c_void_p, np.ctypeslib.ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def CopyNumPyToTensor(tensor_ptr: ctypes.c_void_p, numpy_obj: np.ndarray) -> None:
        global TENSOR_TYPE_FLOAT32, TENSOR_TYPE_INT32, TENSOR_TYPE_INT64, TENSOR_TYPE_UINT8, TENSOR_TYPE_BOOL8, TENSOR_TYPE_FLOAT64
        try :
            if (numpy_obj.dtype == "float32"): numpy_type = TENSOR_TYPE_FLOAT32
            elif (numpy_obj.dtype == "int32"): numpy_type = TENSOR_TYPE_INT32
            elif (numpy_obj.dtype == "int64"): numpy_type = TENSOR_TYPE_INT64
            elif (numpy_obj.dtype == "uint8"): numpy_type = TENSOR_TYPE_UINT8
            elif (numpy_obj.dtype == "bool8"): numpy_type = TENSOR_TYPE_BOOL8
            elif (numpy_obj.dtype == "float64"): numpy_type = TENSOR_TYPE_FLOAT64
            else: raise Exception("exception : Unsupported NumPy data type")

            numpy_shape = []
            for idx in range(numpy_obj.ndim):
                numpy_shape.append(numpy_obj.shape[idx])

            numpy_ptr = np.ctypeslib.as_ctypes(numpy_obj)

            lib_CopyNumPyToTensor(tensor_ptr, numpy_ptr, numpy_type, numpy_shape)
        except Exception as e : exception_handler(e)

    lib_ConvertTensorToNumPy = KAI_LIB.ConvertTensorToNumPy
    lib_ConvertTensorToNumPy.restype = ctypes.py_object 
    lib_ConvertTensorToNumPy.argtypes = [ctypes.c_void_p]
    def ConvertTensorToNumPy(tensor_ptr: ctypes.c_void_p) -> np.ndarray:
        global TENSOR_TYPE_FLOAT32, TENSOR_TYPE_INT32, TENSOR_TYPE_INT64, TENSOR_TYPE_UINT8, TENSOR_TYPE_BOOL8
        try :
            tensor_info = lib_ConvertTensorToNumPy(tensor_ptr)

            tensor_type = tensor_info[0]
            type_c = 0
            type_numpy = 0
            if(tensor_type == TENSOR_TYPE_FLOAT32):
                type_c = ctypes.c_float
                type_numpy = np.float32
            elif(tensor_type == TENSOR_TYPE_INT32):
                type_c = ctypes.c_int32
                type_numpy = np.int32
            elif(tensor_type == TENSOR_TYPE_INT64):
                type_c = ctypes.c_int64
                type_numpy = np.int64
            elif(tensor_type == TENSOR_TYPE_UINT8):
                type_c = ctypes.c_uint8
                type_numpy = np.uint8
            elif(tensor_type == TENSOR_TYPE_BOOL8):
                type_c = ctypes.c_bool
                type_numpy = np.bool8
            else:
                raise Exception("exception : Unsupported Tensor data type")
            
            tensor_ptr = tensor_info[1]
            array_size = 1
            for idx in range(2, len(tensor_info)):
                array_size *= tensor_info[idx]
            array_c = ctypes.cast(tensor_ptr, ctypes.POINTER(type_c * array_size)).contents

            result_numpy = np.ctypeslib.as_array(array_c).reshape(tensor_info[2:]).astype(type_numpy)

            return result_numpy
        except Exception as e : exception_handler(e)

    lib_ConvertTensorToScalar = KAI_LIB.ConvertTensorToScalar
    lib_ConvertTensorToScalar.restype = ctypes.c_float
    lib_ConvertTensorToScalar.argtypes = [ctypes.c_void_p]
    def ConvertTensorToScalar(tensor_ptr: ctypes.c_void_p) -> float:
        """ For 1-size Tensor only """
        try :
            return lib_ConvertTensorToScalar(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_GetTensorArgmax = KAI_LIB.GetTensorArgmax
    lib_GetTensorArgmax.restype = ctypes.c_void_p
    lib_GetTensorArgmax.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    def GetTensorArgmax(tensor_ptr: ctypes.c_void_p, axis:ctypes.c_int64) -> ctypes.c_void_p:
        try :
            return lib_GetTensorArgmax(tensor_ptr, axis)
        except Exception as e : exception_handler(e)

    lib_TensorBackward = KAI_LIB.TensorBackward
    lib_TensorBackward.restype = None
    lib_TensorBackward.argtypes = [ctypes.c_void_p]
    def TensorBackward(tensor_ptr: ctypes.c_void_p) -> None:
        try :
            lib_TensorBackward(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorBackwardWithGrad = KAI_LIB.TensorBackwardWithGrad
    lib_TensorBackwardWithGrad.restype = None
    lib_TensorBackwardWithGrad.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    def TensorBackwardWithGrad(tensor_ptr: ctypes.c_void_p, tensor_grad_ptr: ctypes.c_void_p) -> None:
        try :
            lib_TensorBackwardWithGrad(tensor_ptr, tensor_grad_ptr)
        except Exception as e : exception_handler(e)

    lib_ApplySigmoidToTensor = KAI_LIB.ApplySigmoidToTensor
    lib_ApplySigmoidToTensor.restype = ctypes.c_void_p
    lib_ApplySigmoidToTensor.argtypes = [ctypes.c_void_p]
    def ApplySigmoidToTensor(tensor_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        try :
            return lib_ApplySigmoidToTensor(tensor_ptr)
        except Exception as e : exception_handler(e)
        
    ################################################################
    # Tensor - dhyoon added
    ################################################################

    lib_ModuleLoadCfgWeight = KAI_LIB.ModuleLoadCfgWeight
    lib_ModuleLoadCfgWeight.restype = None
    lib_ModuleLoadCfgWeight.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def ModuleLoadCfgWeight(module_ptr, cfg_path, weight_path):
        try :
            lib_ModuleLoadCfgWeight(module_ptr, cfg_path, weight_path)
        except Exception as e : exception_handler(e)

    lib_ModuleSave = KAI_LIB.ModuleSave
    lib_ModuleSave.restype = None
    lib_ModuleSave.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def ModuleSave(module_ptr, filename):
        try :
            lib_ModuleSave(module_ptr, filename)
        except Exception as e : exception_handler(e)

    lib_ModuleInitParameters = KAI_LIB.ModuleInitParameters
    lib_ModuleInitParameters.restype = None
    lib_ModuleInitParameters.argtypes = [ctypes.c_void_p]
    def ModuleInitParameters(module_ptr):
        try :
            lib_ModuleInitParameters(module_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorToType = KAI_LIB.TensorToType
    lib_TensorToType.restype = ctypes.c_void_p
    lib_TensorToType.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def TensorToType(tensor_ptr: ctypes.c_void_p, type_name_ptr: ctypes.py_object, option_ptr: ctypes.py_object) -> ctypes.c_void_p:
        """ Create new tensor coverting an existing tensor to a new data type. """
        try :
            return lib_TensorToType(tensor_ptr, type_name_ptr, option_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorIndexedByInt = KAI_LIB.TensorIndexedByInt
    lib_TensorIndexedByInt.restype = ctypes.c_void_p
    lib_TensorIndexedByInt.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def TensorIndexedByInt(tensor_ptr: ctypes.c_void_p, index: int) -> ctypes.c_void_p:
        """ Create new tensor slicing a row from an existing tensor. """
        try :
            return lib_TensorIndexedByInt(tensor_ptr, index)
        except Exception as e : exception_handler(e)

    lib_TensorIndexedByTensor = KAI_LIB.TensorIndexedByTensor
    lib_TensorIndexedByTensor.restype = ctypes.c_void_p
    lib_TensorIndexedByTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    def TensorIndexedByTensor(tensor_ptr: ctypes.c_void_p, index_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        """ Create new tensor slicing indexed rows from an existing tensor. """
        try :
            return lib_TensorIndexedByTensor(tensor_ptr, index_ptr)
        except Exception as e : exception_handler(e)

    lib_ValueIndexedBySlice = KAI_LIB.ValueIndexedBySlice
    lib_ValueIndexedBySlice.restype = ctypes.py_object
    lib_ValueIndexedBySlice.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def ValueIndexedBySlice(tensor_ptr: ctypes.c_void_p, index_int: ctypes.py_object) -> ctypes.py_object:
        """ Create new tensor slicing indexed rows from an existing tensor. """
        try :
            return lib_ValueIndexedBySlice(tensor_ptr, index_int)
        except Exception as e : exception_handler(e)

    lib_TensorIndexedBySlice = KAI_LIB.TensorIndexedBySlice
    lib_TensorIndexedBySlice.restype = ctypes.c_void_p
    lib_TensorIndexedBySlice.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def TensorIndexedBySlice(tensor_ptr: ctypes.c_void_p, index_tuple: ctypes.py_object) -> ctypes.c_void_p:
        """ Create new tensor slicing indexed rows from an existing tensor. """
        try :
            return lib_TensorIndexedBySlice(tensor_ptr, index_tuple)
        except Exception as e : exception_handler(e)

    lib_TensorSetElementByTensor = KAI_LIB.TensorSetElementByTensor
    lib_TensorSetElementByTensor.restype = None
    lib_TensorSetElementByTensor.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.c_void_p]
    def TensorSetElementByTensor(tensor_ptr: ctypes.c_void_p, index_tuple: ctypes.py_object, src_tensor_ptr:ctypes.c_void_p) -> None:
        """ Create new tensor slicing indexed rows from an existing tensor. """
        try :
            return lib_TensorSetElementByTensor(tensor_ptr, index_tuple, src_tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorSetElementByArray = KAI_LIB.TensorSetElementByArray
    lib_TensorSetElementByArray.restype = None
    lib_TensorSetElementByArray.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.c_int64, ctypes.py_object, np.ctypeslib.ctypes.c_void_p]
    def TensorSetElementByArray(tensor_ptr: ctypes.c_void_p, index_tuple: ctypes.py_object, value:np.ndarray) -> None:
        """ Create new tensor slicing indexed rows from an existing tensor. """
        try :
            if (value.dtype == "float32"): numpy_type = TENSOR_TYPE_FLOAT32
            elif (value.dtype == "int32"): numpy_type = TENSOR_TYPE_INT32
            elif (value.dtype == "int64"): numpy_type = TENSOR_TYPE_INT64
            elif (value.dtype == "uint8"): numpy_type = TENSOR_TYPE_UINT8
            elif (value.dtype == "bool8"): numpy_type = TENSOR_TYPE_BOOL8
            elif (value.dtype == "float64"): numpy_type = TENSOR_TYPE_FLOAT64
            else: raise Exception("exception : Unsupported NumPy data type")

            numpy_size = value.size
            numpy_ptr = np.ctypeslib.as_ctypes(value)

            lib_TensorSetElementByArray(tensor_ptr, index_tuple, numpy_size, numpy_type, numpy_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorSetElementByValue = KAI_LIB.TensorSetElementByValue
    lib_TensorSetElementByValue.restype = None
    lib_TensorSetElementByValue.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def TensorSetElementByValue(tensor_ptr: ctypes.c_void_p, index_tuple: ctypes.py_object, value:ctypes.py_object) -> None:
        """ Create new tensor slicing indexed rows from an existing tensor. """
        try :
            lib_TensorSetElementByValue(tensor_ptr, index_tuple, value)
        except Exception as e : exception_handler(e)

    lib_TensorPickupRows = KAI_LIB.TensorPickupRows
    lib_TensorPickupRows.restype = None
    lib_TensorPickupRows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, np.ctypeslib.ctypes.c_void_p]
    def TensorPickupRows(tensor_ptr: ctypes.c_void_p, src_tensor_ptr: ctypes.c_void_p, idx:np.ndarray) -> None:
        try:
            if idx.dtype != "int32": raise Exception('bad idx type')
            idx_ptr = np.ctypeslib.as_ctypes(idx)
            lib_TensorPickupRows(tensor_ptr, src_tensor_ptr, idx.size, idx_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorCopyIntoRow = KAI_LIB.TensorCopyIntoRow
    lib_TensorCopyIntoRow.restype = None
    lib_TensorCopyIntoRow.argtypes = [ctypes.c_void_p, ctypes.c_int64, np.ctypeslib.ctypes.c_void_p]
    def TensorCopyIntoRow(tensor_ptr: ctypes.c_void_p, nth:int, src_tensor_ptr: ctypes.c_void_p) -> None:
        try:
            lib_TensorCopyIntoRow(tensor_ptr, nth, src_tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorSetZero = KAI_LIB.TensorSetZero
    lib_TensorSetZero.restype = None
    lib_TensorSetZero.argtypes = [ctypes.c_void_p]
    def TensorSetZero(tensor_ptr: ctypes.c_void_p) -> None:
        try:
            lib_TensorSetZero(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorCopyData = KAI_LIB.TensorCopyData
    lib_TensorCopyData.restype = None
    lib_TensorCopyData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    def TensorCopyData(tensor_ptr: ctypes.c_void_p, src_tensor_ptr: ctypes.c_void_p) -> None:
        """ Copy data from the source tensor. """
        try :
            lib_TensorCopyData(tensor_ptr, src_tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorShiftTimestepToRight = KAI_LIB.TensorShiftTimestepToRight
    lib_TensorShiftTimestepToRight.restype = None
    lib_TensorShiftTimestepToRight.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    def TensorShiftTimestepToRight(tensor_ptr: ctypes.c_void_p, src_tensor_ptr: ctypes.c_void_p, steps) -> None:
        """ Copy data from the source tensor. """
        try :
            TensorShiftTimestepToRight(tensor_ptr, src_tensor_ptr, steps)
        except Exception as e : exception_handler(e)

    lib_TensorSquare = KAI_LIB.TensorSquare
    lib_TensorSquare.restype = ctypes.c_void_p
    lib_TensorSquare.argtypes = [ctypes.c_void_p]
    def TensorSquare(tensor_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        """ Create new tensor with square values for each element in current tensor. """
        try :
            return lib_TensorSquare(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorSum = KAI_LIB.TensorSum
    lib_TensorSum.restype = ctypes.c_void_p
    lib_TensorSum.argtypes = [ctypes.c_void_p]
    def TensorSum(tensor_ptr: ctypes.c_void_p) -> ctypes.c_void_p:
        """ Create new tensor with size 1 for sthe sum of elements in current tensor. """
        try :
            return lib_TensorSum(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorResize = KAI_LIB.TensorResize
    lib_TensorResize.restype = ctypes.c_void_p
    lib_TensorResize.argtypes = [ctypes.c_void_p, ctypes.py_object]
    def TensorResize(tensor_ptr: ctypes.c_void_p, shape:ctypes.py_object) -> ctypes.c_void_p:
        """ Create new tensor with new shape. """
        try :
            return lib_TensorResize(tensor_ptr, shape)
        except Exception as e : exception_handler(e)

    lib_TensorResizeOn = KAI_LIB.TensorResizeOn
    lib_TensorResizeOn.restype = None
    lib_TensorResizeOn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    def TensorResizeOn(tensor_ptr: ctypes.c_void_p, src_ptr: ctypes.c_void_p) -> None:
        """ Create new tensor with new shape. """
        try :
            lib_TensorResizeOn(tensor_ptr, src_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorTransposeOn = KAI_LIB.TensorTransposeOn
    lib_TensorTransposeOn.restype = None
    lib_TensorTransposeOn.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object]
    def TensorTransposeOn(tensor_ptr: ctypes.c_void_p, axis1: ctypes.py_object, axis2: ctypes.py_object) -> None:
        """ Create new tensor with new shape. """
        try :
            lib_TensorTransposeOn(tensor_ptr, axis1, axis2)
        except Exception as e : exception_handler(e)

    lib_GetTensorTypeName = KAI_LIB.GetTensorTypeName
    lib_GetTensorTypeName.restype = ctypes.py_object
    lib_GetTensorTypeName.argtypes = [ctypes.c_void_p]
    def GetTensorTypeName(tensor_ptr: ctypes.c_void_p) -> str:
        try :
            return lib_GetTensorTypeName(tensor_ptr)
        except Exception as e : exception_handler(e)

    lib_TensorLoadJpegPixels = KAI_LIB.TensorLoadJpegPixels
    lib_TensorLoadJpegPixels.restype = None
    lib_TensorLoadJpegPixels.argtypes = [ctypes.c_void_p, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object]
    def TensorLoadJpegPixels(tensor_ptr: ctypes.c_void_p, filepath: str, chn_last:bool, transpose: bool, code: int, mix: float) -> None:
        try :
            lib_TensorLoadJpegPixels(tensor_ptr, filepath, chn_last, transpose, code, mix)
        except Exception as e : exception_handler(e)

    ################################################################
    # Sample methods for developing new features
    ################################################################

    lib_ExecutePythonFunction = KAI_LIB.ExecutePythonFunction
    lib_ExecutePythonFunction.restype = ctypes.py_object
    lib_ExecutePythonFunction.argtypes = [ctypes.py_object, ctypes.py_object]
    def ExecutePythonFunction(callback_function: Callable, arg: dict):
        try :
            return lib_ExecutePythonFunction(callback_function, arg)
        except Exception as e : exception_handler(e)

####################################################################################################

else:
    print("Module '%s' define function fail." % (lib_name + lib_ext))
