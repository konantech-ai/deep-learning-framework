# parameters.py
from .api import *
from .tensor import *


class Parameters:
    """
    This class is Parameters - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `model_ptr`: ctypes.c_void_p
        Target model pointer( C++ engine ).
    """

    def __init__(self, params_ptr: ctypes.c_void_p):
        self.__params_ptr = params_ptr

    def __del__(self):
        if DeleteParameters(self.__params_ptr) is False:
            raise Exception("error: Failed to delete the Parameters object.")

    def __str__(self):
        return GetParametersDump(self.__params_ptr)

    def get_core(self) -> ctypes.c_void_p:
        """
        Get void pointer from Parameters - Capsule Class( C++ engine ).
        This pointer serve to connect 'Deep Learning Framework Engine'.

        Returns
        -------
        ctypes.c_void_p
        """
        return self.__params_ptr

    def weights(self) -> dict:
        return self._recv_tensor_dict(GetParameterWeightDict(self.__params_ptr))
    
    def gradients(self) -> dict:
        return self._recv_tensor_dict(GetParameterGradientDict(self.__params_ptr))
    
    def _recv_tensor_dict(self, x):
        xs = {}
        for key in x.keys():
            xs[key] = Tensor(x[key])
        return xs;

