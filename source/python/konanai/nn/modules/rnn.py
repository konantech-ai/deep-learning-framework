from .module import Module


class RNN(Module):
    """
    This class is RNN - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "rnn"
    
    def __init__(self, **kwargs):
        super(RNN, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(RNN, self).__del__()


class LSTM(Module):
    """
    This class is LSTM - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "lstm"
    
    def __init__(self, **kwargs):
        super(LSTM, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(LSTM, self).__del__()


class GRU(Module):
    """
    This class is GRU - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "gru"
    
    def __init__(self, **kwargs):
        super(GRU, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(GRU, self).__del__()

