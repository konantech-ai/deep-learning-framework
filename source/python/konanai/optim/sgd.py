from .optimizer import Optimizer
from ..parameters import *
from typing import Optional
from .._global_variables import GlobalVariables as gvar


class SGD(Optimizer):
    """
    This class is SGD Optimizer - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `params`: Parameters
        Parameters from neural network model.
    `lr`: float
        learning rate.
    `args`: dict
        Arguments for initialize optimizer.
    """

    optimizer_name = 'sgd'    
    def __init__(self, params: Parameters, **kwargs):
        super(SGD, self).__init__(params, __class__.optimizer_name, **kwargs)

    def __del__(self):
        super(SGD, self).__del__()
