from ..session import Session
from ..api import *
from .._global_variables import GlobalVariables as gvar
from ..tensor import Tensor, convert_numpy_to_tensor
from typing import Any, Callable, Optional


class Dataset:
    """
    Base class for making datasets.
    It is necessary to override the ``len`` method.
    
    Parameters
    ----------
    `transform`: Optional[Callable]
        Specify if additional work such as resizing is required.
        However, not yet supported.
    `args`: Optional[dict]
        In addition, it is used when additional parameters are required.
    """
    def __init__(self, session, name:str, train:bool, download:bool, data_root:str, cache_root:str, transform:Callable, verbose:bool, select:str, args:dict):
        self.session = session
        self.name = name
        self.train = train
        self.download = download
        self.data_root = data_root
        self.cache_root = cache_root
        self.transform = transform
        self.select = select
        self.args = args
        self.verbose = verbose
        
    def __del__(self):
        pass

    def __len__(self) -> int:
        return self.len

    @property
    def len(self) -> int:
        # In user defined base classes, abstract methods should raise this exception when they require derived classes to override the method.
        # This exception is derived from RuntimeError.
        # Reference : https://docs.python.org/3.9/library/exceptions.html#NotImplementedError
        raise NotImplementedError()

    def get_arg(self, key: str, default: Any = None):
        return self.args.get(key, default)
    
    def convert_numpy_to_tensor(self, x: np.ndarray) -> Tensor:
        return convert_numpy_to_tensor(self.session.get_core(), x)

    def setData(self):
        self._setData(self.train, self.download)

        # Post processing : dtype conversion, normalization, etc.
        # Issue (2023-01-11) :
        #   Dataset에서는 raw 데이터를 읽기만 하게 하고, convert/normalize/resize 같은 부가 기능들은
        #   유저가 외부 함수로서 만들고 DataLoader에 전달하도록(즉, 일반적인 사용 방식으로) 변경해야 함
        if hasattr(self, 'x') and isinstance(self.x, np.ndarray):
            self.x = self.convert_numpy_to_tensor(self.x)

        if hasattr(self, 'y') and isinstance(self.y, np.ndarray):
            self.y = self.convert_numpy_to_tensor(self.y)

        self.load_postproc(train=self.train)

        self.data_idx = np.arange(self.len, dtype="int")

        if self.select == "random":
            np.random.shuffle(self.data_idx)

    def _setData(self, train, download):
        # Set the path
        if self.name != "":
            data_path = os.path.join(self.data_root, self.name.lower())
            cache_path = os.path.join(self.cache_root, self.name.lower()) if self.cache_root is not None else None
        else:
            data_path = self.data_root
            cache_path = self.cache_root if self.cache_root is not None else None

        # Load the existing cache
        if cache_path is not None:
            try:
                if self.load_cache(cache_path, train):
                    return
            except Exception:
                pass
        
        # Download raw data from web site
        if download:
            self.download_data(data_path, train)
        
        self.load_data(data_path, train)

        # Save the data as cache
        if cache_path is not None:
            # Printout a log
            if self.verbose: print("Start to save the {:s} cache.".format('train' if train else 'test'))
            self.save_cache(cache_path, train)

    def load_data(self, data_path:str, traib:bool) -> None:
        # In user defined base classes, abstract methods should raise this exception when they require derived classes to override the method.
        # This exception is derived from RuntimeError.
        # Reference : https://docs.python.org/3.9/library/exceptions.html#NotImplementedError
        raise NotImplementedError()
    
    def download_data(self, data_path:str, traib:bool) -> None:
        # In user defined base classes, abstract methods should raise this exception when they require derived classes to override the method.
        # This exception is derived from RuntimeError.
        # Reference : https://docs.python.org/3.9/library/exceptions.html#NotImplementedError
        raise NotImplementedError()
    
    def load_cache(self, cache_path: str, train: bool) -> bool:
        postfix = '.train' if train else '.test'
        filename = cache_path + '/' + self.name.lower() + postfix + '.npz'
        
        # Check the validation
        if not os.path.exists(filename):
            return False
        
        # Printout a log
        if self.verbose: print("Start to load the {:s} cache.".format('train' if train else 'test'))
        
        # Load the cache
        data = np.load(filename)
        
        # Get data as numpy arrays
        self.x = data['x']
        self.y = data['y']

        return True

    def save_cache(self, cache_path: str, train: bool) -> None:
        # Make the directory
        os.makedirs(cache_path, exist_ok=True)
        
        postfix = '.train' if train else '.test'
        filename = cache_path + '/' + self.name.lower() + postfix + '.npz'
        
        # Save one or more numpy arrays as non-compressed *.npz file.
        # This procedure is faster than using pickle.
        x = self.x if isinstance(self.x, np.ndarray) else self.x.to_ndarray()
        y = self.y if isinstance(self.y, np.ndarray) else self.y.to_ndarray()

        np.savez(filename, x=x, y=y)

    def load_postproc(self, train:bool) -> None:
        # In user defined base classes, this can be replaced with useful method
        pass

    def getFieldInfo(self, is_x:bool) -> dict:
        data = self.x if is_x else self.y
        shape = data.shape
        shape = (1,) if len(shape) == 1 else shape[1:]
        if is_x:
            resize = self.get_arg("resize", 0)
            if resize > 0:
                if len(shape) < 3: raise Exception("resize is used only for image data")
                shape = shape[:-2] + (resize, resize)
        info = {"shape": shape, "type": data.dtype }
        return { "#": info }

    @property
    def x_aux_shapes(self):
        info = self.getFieldInfo(True)
        shapes = {}
        for key in info.keys():
            if key != "#": shapes[key] = info[key]["shape"]
        return shapes

    def createAuxBuffer(self, batch_size) -> Tensor or None:
        resize = self.get_arg("resize", 0)
        if resize > 0:
            shape = (batch_size,) + self.x.shape[1:]
            imgBuf = Tensor.from_shape_type(shape, "float32", None)
            return imgBuf
        return None

    def getDataPair(self, idx, xs_buf: dict[str,Tensor], ys_buf: dict[str,Tensor] or None, aux_buf: Tensor):
        resize = self.get_arg("resize", 0)
        if resize == 0:
            xs_buf["#"].pickupRows(self.x, idx)
        else:
            aux_buf.pickupRows(self.x, idx)
            xs_buf["#"].resize_on(aux_buf)

        if ys_buf is not None:
            ys_buf["#"].pickupRows(self.y, idx)

    def dump_data_distribution(self, title="data distribution"):
        dist = self.extract_data_distribution()
        print(f"*** {title} ***")
        for key, freq in dist.items():
            print(f" * {key}: {freq} items")

    def extract_data_distribution(self):
        raise Exception("Dataset.extract_data_distribution() should be implemented on derived class")
