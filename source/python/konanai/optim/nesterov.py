from .optimizer import Optimizer
from ..parameters import *
from typing import Optional


class Nesterov(Optimizer):
    """
    This class is Nesterov Optimizer - Capsule Class( Python ).
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

    optimizer_name = 'nesterov'
    def __init__(self, params: Parameters, **kwargs):
        super(Nesterov, self).__init__(params, __class__.optimizer_name, **kwargs)

    def __del__(self):
        super(Nesterov, self).__del__()
