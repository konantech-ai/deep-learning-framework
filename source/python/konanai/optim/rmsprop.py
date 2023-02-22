from .optimizer import Optimizer
from ..parameters import *
from typing import Optional


class RMSprop(Optimizer):
    """
    This class is RMSprop Optimizer - Capsule Class( Python ).
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

    optimizer_name = 'rmsprop'    
    def __init__(self, params: Parameters, **kwargs):
        args.update( {"lr":lr} )
        super(RMSprop, self).__init__(params, __class__.optimizer_name, **kwargs)

    def __del__(self):
        super(RMSprop, self).__del__()
