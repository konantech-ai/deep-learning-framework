from ..api import *
from .._global_variables import GlobalVariables as gvar
from ..parameters import *
from typing import Optional


class Optimizer:
    """
    This class is Optimizer - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Default class for create optimizer.

    Parameters
    ----------
    `params`: Parameters
        Parameters from neural network model.
    `optimizer_name`: str
        Optimizer name.
    `args`: dict
        Arguments for initialize optimizer.
    """

    def __init__(self, params: Parameters, optimizer_name: str, **kwargs):
        self.__session_ptr = gvar.get_session_ptr()
        self.__params_ptr = params.get_core()
        self.__optimizer_name = optimizer_name
        self.__args = kwargs
        self.__optimizer_ptr = CreateOptimizer(self.__session_ptr, self.__params_ptr, self.__optimizer_name, self.__args)

    def __del__(self):
        if DeleteOptimizer(self.__optimizer_ptr) is False:
            raise Exception("error: Failed to delete the Optimizer object.")

    def setup(self, **kwargs):
        for key in kwargs:
            self.__args[key] = kwargs[key]
        OptimizerSetup(self.__optimizer_ptr, kwargs)

    def get_core(self) -> ctypes.c_void_p:
        """
        Get void pointer from Optimizer - Capsule Class( C++ engine ).
        This pointer serve to connect 'Deep Learning Framework Engine'.

        Returns
        ----------
        ctypes.c_void_p
        """
        return self.__optimizer_ptr

    def zero_grad(self) -> bool:
        """
        Reset gradients computation.

        Returns
        ----------
        bool
        """
        return OptimizerZeroGrad(self.__optimizer_ptr)
    
    def step(self) -> bool:
        """
        Do optimizer step.

        Returns
        ----------
        bool
        """
        return OptimizerStep(self.__optimizer_ptr)
    
    def parameters(self) -> Parameters:
        return CreateParameters(self.__params_ptr)
