from .module import Module


class Conv(Module):
    """
    This class is Conv module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For convolution calculating module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "conv"
    
    def __init__(self, **kwargs):
        super(Conv, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Conv, self).__del__()


class Conv1d(Module):
    """
    This class is Conv1d module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For 1D convolution calculating module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "conv1d"
    
    def __init__(self, **kwargs):
        super(Conv1d, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Conv1d, self).__del__()


class Conv2d(Module):
    """
    This class is Conv2d module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For 2D convolution calculating module.

    Parameters
    ----------
    `xchn`: int
        Input channel size.
    `ychn`: int
        Output channel size.
    `ksize`: int
        Convolution kernel(=filter) size.
    `*args`: dict
        For creating module argument.
    `kwargs`:
        keyword arguments
    """
    module_name = "conv2d"
    
    def __init__(self, **kwargs):
        #if xchn is not None:
        #    args.update( {'xchn':xchn} )
        #if ychn is not None:
        #    args.update( {'ychn':ychn} )
        #if ksize is not None:
        #    args.update( {'ksize':ksize} )
        super(Conv2d, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Conv2d, self).__del__()


class ConvTranspose2d(Module):
    """
    This class is ConvTranspose2d module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For 2D convolution transpose module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "conv2d_transposed"
    
    def __init__(self, **kwargs):
        super(ConvTranspose2d, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(ConvTranspose2d, self).__del__()


# Original
class ConvDepthwise1d(Module):
    """
    This class is ConvDepthwise1d module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For 1D convolution Depthwise module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "depthwise_conv"
    
    def __init__(self, **kwargs):
        super(ConvDepthwise1d, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(ConvDepthwise1d, self).__del__()


# Original
class ConvDepthwise2d(Module):
    """
    This class is ConvDepthwise2d module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For 2D convolution Depthwise module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "conv2d_depthwise"
    
    def __init__(self, **kwargs):
        super(ConvDepthwise2d, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(ConvDepthwise2d, self).__del__()


# Original
class ConvDilated2d(Module):
    """
    This class is conv2d_dilated module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For 2D convolution dilated module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "conv2d_dilated"
    
    def __init__(self, **kwargs):
        super(ConvDilated2d, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(ConvDilated2d, self).__del__()


# Original
class Deconv(Module):
    """
    This class is Deconv module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For de-convolution module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "deconv"
    
    def __init__(self, **kwargs):
        super(Deconv, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Deconv, self).__del__()

