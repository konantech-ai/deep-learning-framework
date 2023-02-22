from .module import Module


# Original
class Activate(Module):
    """
    This class is Activate activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create non-linear activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "activate"
    
    def __init__(self, **kwargs):
        super(Activate, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Activate, self).__del__()


class ReLU(Module):
    """
    This class is ReLU activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create ReLU activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "relu"
    
    def __init__(self, **kwargs):
        super(ReLU, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(ReLU, self).__del__()


class Sigmoid(Module):
    """
    This class is Sigmoid activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Sigmoid activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "sigmoid"
    
    def __init__(self, **kwargs):
        super(Sigmoid, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Sigmoid, self).__del__()


class Tanh(Module):
    """
    This class is Tanh activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Tanh activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "tanh"
    
    def __init__(self, **kwargs):
        super(Tanh, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Tanh, self).__del__()


class Softmax(Module):
    """
    This class is Softmax activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Softmax activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "softmax"

    def __init__(self, **kwargs):
        super(Softmax, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Softmax, self).__del__()


class GELU(Module):
    """
    This class is GELU activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create GELU activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "gelu"
    
    def __init__(self, **kwargs):
        super(GELU, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(GELU, self).__del__()


# Original
class Swish(Module):
    """
    This class is Swish activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Swish activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "swish"
    
    def __init__(self, **kwargs):
        super(Swish, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Swish, self).__del__()


class Mish(Module):
    """
    This class is Mish activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create Mish activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "mish"
    
    def __init__(self, **kwargs):
        super(Mish, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Mish, self).__del__()


class LeakyReLU(Module):
    """
    This class is LeakyReLU activation function - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Create LeakyReLU activation function module.

    Parameters
    ----------
    `args`: dict
        Arguments for initialize activation function.
    """
    module_name = "leaky_relu"
    
    def __init__(self, **kwargs):
        super(LeakyReLU, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(LeakyReLU, self).__del__()

