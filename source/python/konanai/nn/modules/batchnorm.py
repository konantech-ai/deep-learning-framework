from .module import Module


# Original
class BatchNorm(Module):
    """
    This class is Batch Normalization - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Batch Normalization function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize Batch Normalization function module.
    """
    module_name = "batchnorm"
    
    def __init__(self, **kwargs):
        super(BatchNorm, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(BatchNorm, self).__del__()

