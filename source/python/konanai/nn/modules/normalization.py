from .module import Module


class LayerNorm(Module):
    """
    This class is Layer Normalization - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Layer Normalization function module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "layernorm"
    
    def __init__(self, **kwargs):
        super(LayerNorm, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(LayerNorm, self).__del__()

