from .optimizer import Optimizer
from ..parameters import *
from typing import Optional


class Adam(Optimizer):
    """
    This class is Adam Optimizer - Capsule Class( Python ).
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

    optimizer_name = 'adam'
    def __init__(self, params: Parameters, **kwargs):
        super(Adam, self).__init__(params, __class__.optimizer_name, **kwargs)

    def __del__(self):
        super(Adam, self).__del__()
