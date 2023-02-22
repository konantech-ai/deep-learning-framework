# Import modules
from ..api import *
from .functions import *

# Delete the file objects from the list of available names
del functions

# Allow imports by the '*' keyword
__all__ = [
    'basic_train',
    'basic_test_classify'
]
