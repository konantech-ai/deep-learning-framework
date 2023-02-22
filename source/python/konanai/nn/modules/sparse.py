from .module import Module


class Embedding(Module):
    """
    This class is Embedding - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "embedding"
    
    def __init__(self, **kwargs):
        super(Embedding, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Embedding, self).__del__()

