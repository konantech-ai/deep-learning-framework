from .module import Module


# added by dyhoon
def _append_children(module_list: list, item_list:list):
    for item in item_list:     
        if item is None:
            pass
        elif isinstance(item, Module):
            module_list.append(item)
        elif isinstance(item, list):
            _append_children(module_list, item)
        else:
            return True
    return False


class Container(Module):
    """
    This class is Container - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Default class for create Container module.

    Parameters
    ----------
    `module_name`: str
        Create module name.
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    def __init__(self, module_name: str, *args: Module or list or dict):
        tuple_size = len(args)

        module_list = []
        args_dict = {}
        is_failure = False
        
        for idx, item in enumerate(args):
            if item is None:
                pass
            elif isinstance(item, Module):
                module_list.append(item)
            elif isinstance(item, list):
                is_failure = _append_children(module_list, item)
            elif (idx == tuple_size - 1) and (isinstance(item, dict)):
                args_dict = item
            else:
                is_failure = True

        if is_failure is True:
            raise ValueError("Container arguments can only be entered as Module or PyDict types, and PyDict arguments can be entered only at the end.")

        super(Container, self)._init_from_module_list(module_name, module_list, args_dict)

    def __del__(self):
        super(Container, self).__del__()


class Sequential(Container):
    """
    This class is Sequential module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For making complexity module(=layer).

    Parameters
    ----------
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    module_name = "sequential"
    
    def __init__(self, *args: Module or list or dict):
        super(Sequential, self).__init__(__class__.module_name, *args)

    def __del__(self):
        super(Sequential, self).__del__()


class Add(Container):
    """
    This class is Add module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For Add operation module.

    Parameters
    ----------
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    module_name = "add"
    
    def __init__(self, *args: Module or list or dict):
        super(Add, self).__init__(__class__.module_name, *args)

    def __del__(self):
        super(Add, self).__del__()


class Residual(Container):
    """
    This class is Residual module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For Residual operation module.

    Parameters
    ----------
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    module_name = "residual"
    
    def __init__(self, *args: Module or list or dict):
        super(Residual, self).__init__(__class__.module_name, *args)

    def __del__(self):
        super(Residual, self).__del__()


class Parallel(Container):
    """
    This class is Parallel module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For Parallel operation module.

    Parameters
    ----------
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    module_name = "parallel"
    
    def __init__(self, *args: Module or list or dict):
        super(Parallel, self).__init__(__class__.module_name, *args)

    def __del__(self):
        super(Parallel, self).__del__()


class Pruning(Container):
    """
    This class is Pruning module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For Pruning operation module.

    Parameters
    ----------
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    module_name = "pruning"
    
    def __init__(self, *args: Module or list or dict):
        super(Pruning, self).__init__(__class__.module_name, *args)

    def __del__(self):
        super(Pruning, self).__del__()


class Stack(Container):
    """
    This class is Pruning module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For Stack operation module.

    Parameters
    ----------
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    module_name = "stack"

    def __init__(self, *args: Module or list or dict):
        super(Stack, self).__init__(__class__.module_name, *args)

    def __del__(self):
        super(Stack, self).__del__()


class SqueezeExcitation(Container):
    """
    This class is Pruning module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For SqueezeExcitation operation module.

    Parameters
    ----------
    `*args`: Module or list or dict
        If arguments is module, make module list.
        If arguments is list, consider multiple list for make module list.
        If arguments is dict, deliver dictionary argument.
    """
    module_name = "squeezeexcitation"

    def __init__(self, *args: Module or list or dict):
        super(SqueezeExcitation, self).__init__(__class__.module_name, *args)

    def __del__(self):
        super(SqueezeExcitation, self).__del__()

