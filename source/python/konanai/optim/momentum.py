from .optimizer import Optimizer
from ..parameters import *
from typing import Optional


class Momentum(Optimizer):
    """
    This class is Momentum Optimizer - Capsule Class( Python ).
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

    optimizer_name = 'momentum'
    def __init__(self, params: Parameters, **kwargs):
        super(Momentum, self).__init__(params, __class__.optimizer_name, **kwargs)

    def __del__(self):
        super(Momentum, self).__del__()
