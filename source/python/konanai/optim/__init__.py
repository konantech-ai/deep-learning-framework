# Import a base module
from .optimizer import Optimizer as Optimizer

# Import modules
from .adagrad import Adagrad as Adagrad
from .adam import Adam as Adam
from .sgd import SGD as SGD
from .momentum import Momentum as Momentum
from .nesterov import Nesterov as Nesterov
from .rmsprop import RMSprop as RMSprop

# Delete the file objects from the list of available names
del optimizer
del adagrad
del adam
del sgd
del momentum
del nesterov
del rmsprop

# Allow imports by the '*' keyword
__all__ = [
    'Adagrad', 'Adam', 'SGD', 'Momentum', 'Nesterov', 'RMSprop'
]
