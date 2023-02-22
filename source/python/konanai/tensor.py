# tensor.py
from .api import *
import numpy as np
from ._global_variables import GlobalVariables as gvar


class Tensor:
    """
    | This class is Tensor - Capsule Class( Python ).
    | This class serve to connect 'Deep Learning Framework Engine'.

    Parameters
    ----------
    | ``tensor_ptr`` (`Type : ctypes.c_void_p`)
    |   Tensor pointer( C++ engine ).
    |   With this pointer, make Python Tensor class.
    """

    def __init__(self, tensor_ptr: ctypes.c_void_p) -> None:
        self.__tensor_ptr = tensor_ptr

    def __del__(self):
        if DeleteTensor(self.__tensor_ptr) is False:
            raise Exception("error: Failed to delete the Tensor object.")

    def __str__(self):
        return GetTensorDump(self.__tensor_ptr, "")

    def __len__(self):
        return GetTensorLength(self.__tensor_ptr)

    def __getitem__(self, k):
        if isinstance(k, int):
            return Tensor(TensorIndexedByInt(self.__tensor_ptr, k))
        elif isinstance(k, np.ndarray):
            index = convert_numpy_to_tensor(gvar.get_session_ptr(), k)
            return Tensor(TensorIndexedByTensor(self.__tensor_ptr, index.__tensor_ptr))
        elif isinstance(k, tuple):
            index = self._one_element_picker_index(k)
            if index >= 0:
                return ValueIndexedBySlice(self.__tensor_ptr, index)
            else:
                return Tensor(TensorIndexedBySlice(self.__tensor_ptr, k))
        else:
            raise Exception(f"bad index for slicing a tensor: {k}")

    def _one_element_picker_index(self, k):
        if len(k) != len(self.shape):
            return -1
        index = 0
        for n in range(0, len(self.shape)):
            if not isinstance(k[n], int):
                return -1
            index *= self.shape[n];
            index += k[n]
        return index

    def __setitem__(self, idx, val):
        if isinstance(val, Tensor):
            TensorSetElementByTensor(self.__tensor_ptr, idx, val.__tensor_ptr)
        elif isinstance(val, list):
            val = np.asarray(val)
            TensorSetElementByArray(self.__tensor_ptr, idx, val)
        elif isinstance(val, np.ndarray):
            TensorSetElementByArray(self.__tensor_ptr, idx, val)
        else:
            TensorSetElementByValue(self.__tensor_ptr, idx, val)

    @property
    def shape(self) -> tuple:
        """
        | Get Tensor Shape.
        Parameters
        ----------
        | None
        Returns
        -------
        | Python tuple 
        """

        return GetTensorShape(self.__tensor_ptr)

    @property
    def size(self) -> int:
        """
        | Get Tensor Size.
        Parameters
        ----------
        | None
        Returns
        -------
        | Python int
        """

        return GetTensorSize(self.__tensor_ptr)

    @property
    def type(self) -> str:
        """
        | Get Tensor Type.
        Parameters
        ----------
        | None
        Returns
        -------
        | Python str
        """

        return GetTensorTypeName(self.__tensor_ptr)

    @property
    def dtype(self) -> str:
        """
        | Get Tensor Data Type
        Parameters
        ----------
        | None
        Returns
        -------
        | Python str
        """

        return GetTensorTypeName(self.__tensor_ptr)

    def get_core(self) -> ctypes.c_void_p:
        """
        | Get void pointer from Tensor - Capsule Class( C++ engine ).
        | This pointer serve to connect 'Deep Learning Framework Engine'.
        Parameters
        ----------
        | None
        Returns
        -------
        | ctypes.c_void_p
        """

        return self.__tensor_ptr

    def dump(self, tensor_name: str, is_full: bool = False) -> None:
        """
        | Show tensor data as string.

        Parameters
        ----------
        | ``tensor_name`` (`Type : str`)
        | ``is_full`` (`Type : bool`)
        |   If this flag is ture, show full data.

        Returns
        -------
        | None
        """

        DumpTensor(self.__tensor_ptr, tensor_name, is_full)

    def to_type(self, type_name:str, option:str = "") -> ...:
        """
        | Tensor tpye convert to other type.

        Parameters
        ----------
        | ``type_name`` (`Type : str`)
        | ``option`` (`Type : str`)
        |   Data normalize option.
        |     'unit' : -1 ~ 1
        |     'posunit' : 0 ~ 1

        Returns
        -------
        | Tensor
        """

        return Tensor(TensorToType(self.__tensor_ptr, type_name, option))

    @classmethod
    def from_shape_type(cls, shape, type, init: list or None = None) -> 'Tensor':
        return cls(CreateTensor(gvar.get_session_ptr(), shape, type, init))

    def from_ndarray(self, x: np.ndarray) -> None:
        """
        | Copy NumPy data to Tensor data.

        Parameters
        ----------
        | ``x`` (`Type : np.ndarray`)

        Returns
        -------
        | None
        """

        CopyNumPyToTensor(self.__tensor_ptr, x)

    def to_ndarray(self) -> np.ndarray:
        """
        | Copy Tensor data to NumPy data.

        Parameters
        ----------
        | None

        Returns
        -------
        | np.ndarray
        """

        return ConvertTensorToNumPy(self.__tensor_ptr)
    
    def to_scalar(self) -> float:
        """
        | For 1-size Tensor only.
        | Return Tensor data float data.

        Parameters
        ----------
        | None

        Returns
        -------
        | Python float
        """

        return ConvertTensorToScalar(self.__tensor_ptr)

    def item(self) -> float:
        """
        | For 1-size Tensor only.
        | Same Function : to_scalar().
        | Return float type.
        | For user convenience.

        Parameters
        ----------
        | None

        Returns
        -------
        | Python float
        """

        return self.to_scalar()
    
    def value(self) -> int:
        """
        | For 1-size Tensor only.
        | Same Function : to_scalar().
        | Return int type.
        | For user convenience.

        Parameters
        ----------
        | None

        Returns
        -------
        | Python int
        """

        return int(self.to_scalar())

    def argmax(self, axis=-1) -> ...:
        """
        | Get Tensor argument max

        Parameters
        ----------
        | ``axis`` (`Type : int`)
        |   Default : -1

        Returns
        -------
        | Tensor
        """

        return Tensor(GetTensorArgmax(self.__tensor_ptr, axis))
    
    def backward(self, ndgrad: Optional[np.ndarray] = None) -> None:
        """
        | Recently executed loss fucntion trace, and proceed backpropagation.
        | If 'ndgrad' exist, proceed backpropagation with gradients.

        Parameters
        ----------
        | ``ndgrad`` (`Type : np.ndarray`)
        |   Gradients NumPy data

        Returns
        -------
        | None
        """

        if ndgrad is None:
            TensorBackward(self.__tensor_ptr)
        else:
            grad = convert_numpy_to_tensor(gvar.get_session_ptr(), ndgrad)
            TensorBackwardWithGrad(self.__tensor_ptr, grad.get_core())

    def sigmoid(self) -> ...:
        """
        | Apply sigmoid function to Tensor data(each elements).

        Parameters
        ----------
        | None

        Returns
        -------
        | Tensor
        """

        return Tensor(ApplySigmoidToTensor(self.__tensor_ptr))

    def square(self) -> ...:
        """
        | Apply square function to Tensor data(each elements).

        Parameters
        ----------
        | None

        Returns
        -------
        | Tensor
        """

        return Tensor(TensorSquare(self.__tensor_ptr))

    def sum(self) -> ...:
        """
        | Apply sum function to Tensor data(each elements).
        | Return 1-size Tensor.

        Parameters
        ----------
        | None

        Returns
        -------
        | Tensor
        """

        return Tensor(TensorSum(self.__tensor_ptr))

    def pickupRows(self, src:..., idx:np.ndarray) -> None:
        """
        | Pick up - Tensor row data

        Parameters
        ----------
        | ``src`` (`Type : Tensor`)
        | ``idx`` (`Type : np.ndarray`)

        Returns
        -------
        | None
        """

        TensorPickupRows(self.__tensor_ptr, src.__tensor_ptr, idx)

    def copy_into_row(self, nth:int, src:...) -> None:
        """
        | Copy - Tensor into row

        Parameters
        ----------
        | ``nth`` (`Type : int`)
        | ``src`` (`Type : Tensor`)

        Returns
        -------
        | None
        """

        TensorCopyIntoRow(self.__tensor_ptr, nth, src.__tensor_ptr)

    def resize_on(self, src:...) -> None:
        """
        | Resize - Tensor from Tensor

        Parameters
        ----------
        | ``src`` (`Type : Tensor`)

        Returns
        -------
        | None
        """

        TensorResizeOn(self.__tensor_ptr, src.__tensor_ptr)

    def transpose_on(self, axis1:int, axis2:int):
        """
        | Transpose - Tensor

        Parameters
        ----------
        | ``axis1`` (`Type : int`)
        | ``axis2`` (`Type : int`)

        Returns
        -------
        | None
        """

        TensorTransposeOn(self.__tensor_ptr, axis1, axis2)

    def set_zero(self) -> None:
        """
        | Set zero - Tensor

        Parameters
        ----------
        | None

        Returns
        -------
        | None
        """

        TensorSetZero(self.__tensor_ptr)

    def resize(self, shape:list) -> ...:
        """
        | Resize Tensor shape to input shape.

        Parameters
        ----------
        | ``shape`` (`Type : list`)

        Returns
        -------
        | Tenosr
        """

        return Tensor(TensorResize(self.__tensor_ptr, shape))

    #def copy_data(self, src: 'Tensor') -> None:
    def copy_data(self, src:...) -> None:
        """
        | Copy Tensor data from input Tensor

        Parameters
        ----------
        | ``src`` (`Type : Tensor`)

        Returns
        -------
        | None
        """

        TensorCopyData(self.__tensor_ptr, src.__tensor_ptr)

    def shift_timestep_to_right(self, src:..., steps:int = 1) -> None:
        """
        | Shift Tensor data to right with timestep

        Parameters
        ----------
        | ``src`` (`Type : Tensor`)
        | ``steps`` (`Type : int`)
        |   Default : 1

        Returns
        -------
        | None
        """

        TensorShiftTimestepToRight(self.__tensor_ptr, src.__tensor_ptr, steps)

    def load_jpeg_pixels(self, filepath: str, chn_last: bool = False, transpose: bool = False, code: int = -1, mix: float = 1.0):
        """
        | Load jpeg pixcels 

        Parameters
        ----------
        | ``filepath`` (`Type : str`)
        | ``chn_last`` (`Type : bool`)
        |   Default : False
        | ``transpose`` (`Type : bool`)
        |   Default : False
        | ``code`` (`Type : int`)
        |   Default : -1
        | ``mix`` (`Type : float`)
        |   Default : 1.0

        Returns
        -------
        | None
        """

        TensorLoadJpegPixels(self.__tensor_ptr, filepath, chn_last, transpose, code, mix)

def convert_tensor_to_numpy(tensor_ptr: ctypes.c_void_p) -> np.ndarray:
    """
    | Convert Tensor data to NumPy data.
    | Avaliable Tensor Type : flaot32, int32, int64, uint8, bool8

    Parameters
    ----------
    | ``tensor_ptr`` (`Type : ctypes.c_void_p`)
    |   Target Tensor pointer( C++ engine ).

    Returns
    -------
    | NumPy
    """

    numpy_obj = ConvertTensorToNumPy(tensor_ptr)
    return numpy_obj

def convert_numpy_to_tensor(session_ptr: ctypes.c_void_p, x: np.ndarray) -> Tensor:
    """
    | Create Tensor from NumPy data.
    | Almost same Tensor.from_ndarray(), but this function create new Tensor class.
    | Avaliable Tensor Type : flaot32, int32, int64, uint8, bool8, float64

    Parameters
    ----------
    | ``session_ptr`` (`Type : ctypes.c_void_p`)
    |   Target session pointer( C++ engine ).
    | ``x`` (`Type : np.ndarray`)
    |   Target NumPy Data.


    Returns
    -------
    | Tensor
    """

    tensor_x_ptr = CreateTensorFromNumPy(session_ptr, x)
    return Tensor(tensor_x_ptr)

# 적절한 파일 저장 위치를 찾지 못해 임시로 더부살이 시킴
class AudioSpectrumReader:
    """
    | This class is AudioSpectrumReader Class( Python ).
    | This class serve to control AudioSpectrum.

    Parameters
    ----------
    | ``reader_ptr`` (`Type : ctypes.c_void_p`)
    """

    def __init__(self, reader_ptr: ctypes.c_void_p) -> None:
        self.__reader_ptr = reader_ptr

    def __del__(self):
        if DeleteAudioSpectrumReader(self.__reader_ptr) is False:
            raise Exception("error: Failed to delete the AudioSpectrumReader object.")

    def add_file(self, filepath:str) -> bool:
        return AudioSpectrumReaderAddFile(self.__reader_ptr, filepath)

    def get_extract_spectrums(self) -> Tensor:
        return Tensor(AudioSpectrumReaderExtractSpectrums(self.__reader_ptr))