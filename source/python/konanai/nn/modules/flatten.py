from .module import Module


class Flatten(Module):
    """
    This class is Flatten module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Flatten function module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "flatten"
    
    def __init__(self, **kwargs):
        super(Flatten, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Flatten, self).__del__()

