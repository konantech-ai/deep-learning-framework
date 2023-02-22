from .module import Module


class Linear(Module):
    """
    This class is Linear module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Linear function module.

    Parameters
    ----------
    `in_width`: int
        Input width.
    `out_width`: int
        Output width.
    `args`: dict
        Arguments for initialize Linear function.
    `kwargs`:
        keyword arguments
    """
    module_name = 'linear'
    
    def __init__(self, **kwargs):
        #if in_width is not None:
        #    args.update( {'in_width':in_width} )
        #if out_width is not None:
        #    args.update( {'out_width':out_width} )
        super(Linear, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Linear, self).__del__()

