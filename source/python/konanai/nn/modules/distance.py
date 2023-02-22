from .module import Module


class CosineSimilarity(Module):
    """
    This class is CosineSimilarity module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create CosineSimilarity function module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "cosinesim"
    
    def __init__(self, **kwargs):
        super(CosineSimilarity, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(CosineSimilarity, self).__del__()

