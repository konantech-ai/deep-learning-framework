from .module import Module


class Upsample(Module):
    """
    This class is Upsample - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "upsample"
    
    def __init__(self, **kwargs):
        super(Upsample, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Upsample, self).__del__()

