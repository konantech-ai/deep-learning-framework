# session.py
from .api import *
from ._global_variables import GlobalVariables as gvar
from .tensor import Tensor, convert_numpy_to_tensor
from .tensor import AudioSpectrumReader
import numpy as np


class Session:
    """
    | This class is Session - Capsule Class( Python ).
    | This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    | ``server_url`` (`Type : str`)
    | ``client_url`` (`Type : str`)    
    """

    def __init__(self, server_url: str="127.0.0.1", client_url:str=""):
        self.__session_ptr = OpenSession(server_url, client_url)

    def __del__(self):
        if CloseSession(self.__session_ptr) is False:
            raise Exception("error: Failed to delete the Session object.")

    def get_core(self) -> ctypes.c_void_p:
        """
        | Get void pointer from Session - Capsule Class( C++ engine ).
        | This pointer serve to connect 'Deep Learning Framework Engine' and access 'Neural Network' algorithm.

        Parameters
        ----------
        | ``server_url`` (`Type : str`)
        | ``client_url`` (`Type : str`)
        
        Returns
        -------
        |ctypes.c_void_p
        """

        return self.__session_ptr

    def srand(self, seed: int) -> None:
        """
        | Set random seed to 'Deep Learning Framework Engine'.

        Parameters
        ----------
        | ``seed`` (`Type : int`)
        |   Random seed value.
        |     Default - None
        |     'time(NULL)' : make random value.
        |     'int' : fixed random value.

        Returns
        -------
        | None
        """
        np.random.seed(seed)
        SrandSeed(self.__session_ptr, seed)
    
    def downloadAsNdArray(self, tensor_capsule) -> np.ndarray:
        """
        | From Tensor capsule, download data as ndarray

        Parameters
        ----------
        | ``tensor_capsule`` (`Type : `)

        Returns
        -------
        | np.ndarray
        """

        tensor_ptr = TensorCapsuleDownload(self.__session_ptr, tensor_capsule)
        return ConvertTensorToNumPy(tensor_ptr)
    
    def downloadAsTenser(self, tensor_capsule) -> Tensor:
        """
        | From Tensor capsule, download data as tensor

        Parameters
        ----------
        | ``tensor_capsule`` (`Type : `)

        Returns
        -------
        | Tensor
        """

        tensor_ptr = TensorCapsuleDownload(self.__session_ptr, tensor_capsule)
        return tensor_ptr

    ################################################################
    # Utility service routines
    ################################################################

    def parse_json_file(self, filepath:str) -> dict or list:
        """
        | Pasrsing JSON file and return Python dictionary.

        Parameters
        ----------
        | ``filepath`` (`Type : str`)
        |   JSON File path

        Returns
        -------
        | Python dict or list
        """

        return UtilParseJsonFile(filepath)

    def parse_jsonl_file(self, filepath:str) -> list:
        """
        | Pasrsing JSON-line file and return Python list.

        Parameters
        ----------
        | ``filepath`` (`Type : str`)
        |   JSON-line File path

        Returns
        -------
        | Python list
        """

        return UtilParseJsonlFile(filepath)

    def read_file_lines(self, filepath:str) -> list:
        """
        | Pasrsing file-line and return Python list.

        Parameters
        ----------
        | ``filepath`` (`Type : str`)
        |   File-line File path

        Returns
        -------
        | Python list
        """

        return UtilReadFileLines(filepath)

    def load_data(self, filepath:str) -> dict:
        """
        | Read Data and return Python dict.

        Parameters
        ----------
        | ``filepath`` (`Type : str`)
        |   Data File path

        Returns
        -------
        | Python dict
        """

        return UtilLoadData(self.__session_ptr, filepath)

    def save_data(self, term, filepath:str) -> None:
        """
        | Save Data.

        Parameters
        ----------
        | ``filepath`` (`Type : str`)
        |   Data File path

        Returns
        -------
        | None
        """

        UtilSaveData(self.__session_ptr, term, filepath)

    ################################################################
    # Tensor
    ################################################################

    def createTensor(self, shape, type, init:list or str or None=None) -> Tensor:
        """
        | Create Tensor

        Parameters
        ----------
        | ``shape`` (`Type : tuple`)
        |   Tensor Shape
        | ``type`` (`Type : str`)
        |   Tensor Data Type
        | ``init`` (`Type : list or str or None`)
        |   Init value

        Returns
        -------
        | Tensor
        """

        return Tensor(CreateTensor(self.__session_ptr, shape, type, init))

    def createTensorFromArray(self, array:np.ndarray) -> Tensor:
        """
        | Create Tensor From NumPy Array

        Parameters
        ----------
        | ``array`` (`Type : np.ndarray`)
        |   Source NumPy Array

        Returns
        -------
        | Tensor
        """

        return convert_numpy_to_tensor(self.__session_ptr, array)

    ################################################################
    # AudioSpectrumReader
    ################################################################

    def createAudioSpectrumReader(self, args: Optional[dict] = {}) -> AudioSpectrumReader:
        """
        | Create AudioSpectrumReader

        Parameters
        ----------
        | ``args`` (`Type : Optional[dict]`)

        Returns
        -------
        | AudioSpectrumReader
        """

        return AudioSpectrumReader(CreateAudioSpectrumReader(self.__session_ptr, args))

    ################################################################
    # no_grad
    ################################################################

    class NoGradBlock:
        """
        | This class is Session - NoGradBlock Class( Python ).
        | This class serve no graddient block

        Parameters
        ----------
        | ``session`` (`Type : Session`)
        |   Set input session class.
        """

        def __init__(self, session):
            self.session = session

        def __enter__(self):
            self.session.set_no_grad()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.session.unset_no_grad()

    def no_grad(self) -> NoGradBlock:
        """
        | No gradients block.

        Parameters
        ----------
        | None

        Returns
        -------
        | NoGradBlock
        """

        return self.NoGradBlock(self)

    def set_no_grad(self) -> None:
        """
        | Set no gradients computation.

        Parameters
        ----------
        | None

        Returns
        -------
        | None
        """

        SetNoGrad(self.__session_ptr)

    def unset_no_grad(self) -> None:
        """
        | Unset no gradients computation.

        Parameters
        ----------
        | None

        Returns
        -------
        | None
        """

        UnsetNoGrad(self.__session_ptr)

################################################################
# Access the session pointer
################################################################

def get_target_session() -> ctypes.c_void_p:
    """
    | Get Tartget Session

    Parameters
    ----------
    | None

    Returns
    -------
    | ctypes.c_void_p
    """

    return gvar.get_session_ptr()

def set_target_session(session: Session) -> None:
    """
    | Set tartget session.

    Parameters
    ----------
    | ``session`` (`Type : Session`)
    |   Set input session class.

    Returns
    -------
    | None
    """

    gvar.set_session_ptr(session.get_core())
