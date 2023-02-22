from ...api import CreateMacro, RegisterMacro
from ..._global_variables import GlobalVariables as gvar
from .module import Module
from typing import Optional


# Original
def register_macro(macro_name: str, module: Module, args: Optional[dict] = {}) -> None:
    """
    Register macro module to 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `macro_name`: str
        Register macro name.
    `module`: Module
        Target macro module
    `args`: dict
        Arguments for initialize register macro module.
    """

    session_ptr = gvar.get_session_ptr()
    RegisterMacro(session_ptr, macro_name, module.get_core(), args)


# Original
class Macro(Module):
    """
    This class is Macro - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `macro_name`: str
        macro name.
    `args`: dict
        Arguments for initialize macro module.
    `kwargs`:
        keyword arguments
    """
    module_name = "macro"
    
    def __init__(self, macro_name: str,**kwargs):
        session_ptr = gvar.get_session_ptr()
        module_ptr = CreateMacro(session_ptr, macro_name, kwargs)
        super(Macro, self).__init__(session_ptr, module_ptr, __class__.module_name, kwargs)

    def __del__(self):
        super(Macro, self).__del__()


# Original
class AddBias(Module):
    """
    This class is AddBias - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "addbias"
    
    def __init__(self, **kwargs):
        super(AddBias, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(AddBias, self).__del__()


# Original
class Dense(Module):
    """
    This class is Dense - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "dense"
    
    def __init__(self, **kwargs):
        super(Dense, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Dense, self).__del__()


# Original
class Reshape(Module):
    """
    This class is Reshape - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "reshape"
    
    def __init__(self, **kwargs):
        super(Reshape, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Reshape, self).__del__()


# Original
class Transpose(Module):
    """
    This class is Transpose - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "transpose"
    
    def __init__(self, **kwargs):
        #if axes is not None:
        #    args.update( {'axes':axes} )
        super(Transpose, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Transpose, self).__del__()


# Original
class Concat(Module):
    """
    This class is Concat - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "concat"
    
    def __init__(self, **kwargs):
        super(Concat, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Concat, self).__del__()


# Original
class Pass(Module):
    """
    This class is Pass - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "pass"
    
    def __init__(self, **kwargs):
        super(Pass, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Pass, self).__del__()


# Original
class Extract(Module):
    """
    This class is Extract - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "extract"
    
    def __init__(self, **kwargs):
        super(Extract, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Extract, self).__del__()


# Original
class MultiHeadAttention(Module):
    """
    This class is MultiHeadAttention - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "mh_attention"
    
    def __init__(self, **kwargs):
        super(MultiHeadAttention, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(MultiHeadAttention, self).__del__()


# Original
class Noise(Module):
    """
    This class is Noise - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "noise"
    
    def __init__(self, **kwargs):
        super(Noise, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Noise, self).__del__()


# Original
class Random(Module):
    """
    This class is Random - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "random"
    
    def __init__(self, **kwargs):
        super(Random, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Random, self).__del__()


# Original
class Round(Module):
    """
    This class is Round - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "round"
    
    def __init__(self, **kwargs):
        super(Round, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Round, self).__del__()


# Original
class SelectNTop(Module):
    """
    This class is SelectNTop - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "selectntop"
    
    def __init__(self, **kwargs):
        super(SelectNTop, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(SelectNTop, self).__del__()


# Original
class SelectNTopArg(Module):
    """
    This class is SelectNTopArg - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "selectntoparg"
    
    def __init__(self, **kwargs):
        super(SelectNTopArg, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(SelectNTopArg, self).__del__()

# Original
class CodeConv(Module):
    """
    This class is CodeConv module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For code-convolution module.

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "codeconv"
    
    def __init__(self, **kwargs):
        super(CodeConv, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(CodeConv, self).__del__()

# Original
class Formula(Module):
    """
    This class is Formula module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For user defined layer using formula expressions .

    Parameters
    ----------
    `kwargs`:
        keyword arguments
    """
    module_name = "formula"
    
    def __init__(self, formula: str, **kwargs):
        kwargs["formula"] = formula
        super(Formula, self)._init_from_module_name(__class__.module_name, kwargs)

    def __del__(self):
        super(Formula, self).__del__()

