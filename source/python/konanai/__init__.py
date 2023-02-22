# Package version
__version__ = '1.1'

# Import API module (Caution! This module must be imported first.)
from .api import free_library

# Delete the file objects from the list of available names
# del api


################################################################################
# To handle subpackages(folders)
################################################################################

# Import subpackages(folders)
# from konanai import comparison as comparison
from konanai import cuda as cuda
from konanai import datasets as datasets
from konanai import nn as nn
from konanai import optim as optim
#from konanai import utils as utils
# from konanai import samples as samples
# from konanai import tutorial as tutorial


################################################################################
# To handle modules(files) and utilities(functions)
################################################################################

# Import modules
# from ._global_variables import GlobalVariables
from .session import Session, get_target_session, set_target_session
from .dataloader import DataLoader
from .tensor import convert_numpy_to_tensor

# Delete the file objects from the list of available names
del session
#del dataloader
del tensor

# Allow imports by the '*' keyword
__all__ = [
    'version', 'get_optimizers'
]


################################################################################
# Define basic utilities
################################################################################

def version() -> str:
    """
    Get 'Deep Learning Framework Engine' Version.

    Returns
    -------
    string
    """
    return __version__  

def get_optimizers() -> list:
    """
    Get available optimizer list.

    Returns
    -------
    list
    """
    return ['Adagrad', 'Adam', 'SGD', 'Momentum', 'Nesterov', 'RMSprop']
