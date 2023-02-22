from ..api import IsCUDAAvailable, GetCUDADeviceCount
from .._global_variables import GlobalVariables as gvar


def is_available() -> bool:
    """
    Check CUDA is available.
    If CUDA is available, return true.

    Returns
    -------
    bool
    """

    return IsCUDAAvailable(gvar.get_session_ptr())

def device_count() -> int:
    """
    Returns the number of available CUDA device.
    If CUDA is unavailable, return 0.

    Returns
    -------
    int
    """

    if is_available():
        return GetCUDADeviceCount(gvar.get_session_ptr())
    else:
        return 0

