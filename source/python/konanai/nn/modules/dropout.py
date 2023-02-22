from .module import Module


class Dropout(Module):
    """
    This class is Dropout module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Dropout function module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "dropout"
    
    def __init__(self, **kwargs):
        super(Dropout, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Dropout, self).__del__()

