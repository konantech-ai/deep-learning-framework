from .module import Module


# Original
class Max(Module):
    """
    This class is Max Pooling - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `stride`: int
        Max Pooling stride value.
    `kwargs`:
        keyword arguments
    """
    module_name = "max"
    
    def __init__(self, **kwargs):
        #if stride is not None:
        #    args.update( {'stride':stride} )
        super(Max, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Max, self).__del__()


# Original
class Avg(Module):
    """
    This class is Average Pooling - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "avg"
    
    def __init__(self, **kwargs):
        super(Avg, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Avg, self).__del__()


# Original
class GlobalAvg(Module):
    """
    This class is Global Average Pooling - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "globalavg"
    
    def __init__(self, **kwargs):
        super(GlobalAvg, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(GlobalAvg, self).__del__()


# Original
class AdaptiveAvg(Module):
    """
    This class is Adaptive Average Pooling - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "adaptiveavg"
    
    def __init__(self, **kwargs):
        super(AdaptiveAvg, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(AdaptiveAvg, self).__del__()

