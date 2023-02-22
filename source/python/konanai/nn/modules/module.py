from ...api import *
from ..._global_variables import GlobalVariables as gvar
from ...parameters import *
#from ...dataloader import *


class Module:
    """
    This class is Module - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Default class for create module.

    Parameters
    ----------
    `session_ptr`: ctypes.c_void_p
        Target session pointer( C++ engine ).
    `module_ptr`: ctypes.c_void_p
        Target module pointer( C++ engine ).
    `module_name`: str
        Target module name.
    `args`: dict
        Arguments for initialize module.
    """
    def __init__(self, session_ptr: ctypes.c_void_p, module_ptr: ctypes.c_void_p, module_name: str="", args: Optional[dict] = {}):
        self.__session_ptr = session_ptr
        self.__module_ptr = module_ptr
        self.__module_name = module_name
        self.__args = args
    
    @classmethod
    def from_module_name(cls, module_name: str, args: Optional[dict] = {}):
        """
        For general users

        Parameters
        ----------
        `module_name`: str
            Craete module name.
        `args`: dict
            Arguments for initialize module.
        """
        session_ptr = gvar.get_session_ptr()
        module_ptr = CreateModule(session_ptr, module_name, args)
        return cls(session_ptr, module_ptr, module_name, args)
    
    def _recv_tensor_dict(self, x):
        xs = {}
        for key in x.keys():
            xs[key] = Tensor(x[key])
        return xs;

    def _init_from_module_name(self, module_name: str, args: Optional[dict] = {}):
        """
        For initializing derived classes
        """
        self.__session_ptr = gvar.get_session_ptr()
        self.__module_ptr = CreateModule(self.__session_ptr, module_name, args)
        self.__module_name = module_name
        self.__args = args
        
    def _init_from_module_list(self, module_name: str, module_list: list['Module'], args: Optional[dict] = {}):
        """
        For initializing derived classes
        """
        self.__session_ptr = gvar.get_session_ptr()
        module_ptr_list = []
        for module in module_list:
            module_ptr_list.append(module.get_core())
        self.__module_ptr = CreateContainer(self.__session_ptr, module_name, module_ptr_list, args)
        self.__module_name = module_name
        self.__args = args

    def __del__(self):
        if DeleteModule(self.__module_ptr) is False:
            raise Exception("error: Failed to delete the Module object.")

    def get_core(self) -> ctypes.c_void_p:
        """
        Get void pointer from Module - Capsule Class( C++ engine ).
        This pointer serve to connect 'Deep Learning Framework Engine'.

        Returns
        ----------
        ctypes.c_void_p
        """
        return self.__module_ptr

    def expand(self, data_shape: list or tuple, **kwargs) -> ...:
        """
        Exapnd model.
        This function do not calculate neural network, just get model shape.

        Parameters
        ----------
        `data_shape`: list or tuple
            shape of input
        `kwargs`: 
            keyword arguments

        Returns
        ----------
        Module
        """
        module_ptr = ExpandModule(self.__module_ptr, data_shape, kwargs)
        self.__args.update(kwargs)
        return Module(self.__session_ptr, module_ptr, self.__module_name, self.__args)

    def to(self, device_name: str) -> ...:
        """
        Module to device.
        You can choose calculation device.

        Parameters
        ----------
        `device_name`: str
            - ``'cuda'``: Use Nvidia CUDA (use all GPU)
            - ``'cuda:#'``: Use Nvidia CUDA (select GPU number)
            - ``'cpu'``: Use CPU

        Returns
        ----------
        Module
        """
        module_ptr = SetModuleToDevice(self.__module_ptr, device_name)
        return Module(self.__session_ptr, module_ptr, self.__module_name, self.__args)

    def shape(self) -> str:
        """
        Get module shape.

        Returns
        ----------
        string
        """
        return GetModuleShape(self.__module_ptr)

    def __repr__(self) -> str:
        return self.shape()

    def __str__(self) -> str:
        return self.shape()

    def parameters(self) -> Parameters:
        """
        Get module parameters.

        Returns
        ----------
        Parameters
        """
        return Parameters(CreateParameters(self.__module_ptr))

    def __getitem__(self, k):
        child_ptr = ModuleNthChild(self.__module_ptr, k)
        return Module(self.__session_ptr, child_ptr)

    def fetch_child(self, name:str) -> ...:
        child_ptr = ModuleFetchChild(self.__module_ptr, name)
        return Module(self.__session_ptr, child_ptr)

    def nth_child(self, nth:int) -> ...:
        child_ptr = ModuleNthChild(self.__module_ptr, nth)
        return Module(self.__session_ptr, child_ptr)

    def seek_layer(self, name:str) -> ...:
        child_ptr = ModuleSeekLayer(self.__module_ptr, name)
        return Module(self.__session_ptr, child_ptr)

    def append_child(self, child:...) -> None :
        ModuleAppendChild(self.__module_ptr, child.get_core())

    def train(self) -> None:
        """
        Sets the module in training mode.
        """
        ModuleTrain(self.__module_ptr)

    def eval(self) -> None:
        """
        Sets the module in evaluation mode.
        """
        ModuleEval(self.__module_ptr)

    def evaluate(self, x: Tensor or np.ndarray or dict) -> Tensor or dict:
        return self.__call__(x)

    def forward(self, x: Tensor or np.ndarray or dict) -> Tensor or dict:
        return self.__call__(x)

    def __call__(self, x: Tensor or np.ndarray or dict) -> Tensor or dict:
        if isinstance(x, dict):
            return self._recv_tensor_dict(ModuleCallDict(self.__module_ptr, x))
        else:
            if type(x) == np.ndarray:
                x = Tensor(CreateTensorFromNumPy(self.__session_ptr, x.astype(np.float32)))
            tensor_ptr_y_hat = ModuleCall(self.__module_ptr, x.get_core())
            return Tensor(tensor_ptr_y_hat)

    def predict(self, x: Tensor or np.ndarray or dict) -> Tensor or dict:
        if isinstance(x, dict):
            return self._recv_tensor_dict(ModulePredictDict(self.__module_ptr, x))
        else:
            if type(x) == np.ndarray:
                x = Tensor(CreateTensorFromNumPy(self.__session_ptr, x.astype(np.float32)))
            tensor_ptr_y_hat = ModulePredict(self.__module_ptr, x.get_core())
            return Tensor(tensor_ptr_y_hat)

    def load_cfg_weight(self, cfg_path:str, weight_path:str):
        ModuleLoadCfgWeight(self.__module_ptr, cfg_path, weight_path)

    def save(self, filename:str) -> None:
        """
        Save model parameters.

        Parameters
        ----------
        `filename`: str
            Save file name.
        """
        ModuleSave(self.__module_ptr, filename)

    def init_parameters(self):
        ModuleInitParameters(self.__module_ptr)

    # 2023.01.12 : Not support : -->
    ## verbose is not implemented yet
    #def compile(self, optimizer: str, loss: str, verbose: str) -> None:
    #    optimizer = optimizer.encode("utf-8")
    #    loss = loss.encode("utf-8")
    #    verbose = verbose.encode("utf-8")
    #    ModuleCompile(self.__module_ptr, {"optimizer":optimizer, "loss":loss, "verbose":verbose})
    #
    #def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int, lr: float) -> None:
    #    # shuffle
    #    idx = np.arange(x.shape[0])
    #    np.random.shuffle(idx)
    #    x = x[idx].astype(np.float32)
    #    y = y[idx].astype(np.float32)
    #
    #    # fit
    #    ModuleFit(self.__session_ptr, self.__module_ptr, x, y, {"batch_size":batch_size, "epochs":epochs, "lr":lr})
    #
    #def fit_ext(self, tr_loader: DataLoader, te_loader: DataLoader, batch_size: int, epochs: int, batch_report: int, lr: float) -> None:
    #
    #    # fit
    #    ModuleFitExt(self.__session_ptr, self.__module_ptr, tr_loader.get_core(), te_loader.get_core(), {"batch_size":batch_size, "batch_report":batch_report, "epochs":epochs, "lr":lr})
    #
    #def saveParameters(self, filename:str, root:Optional[str]=".") -> None:
    #    ModuleSaveParameters(self.__module_ptr, root, filename)
    #
    #def loadParameters(self, filename:str, root:Optional[str]=".") -> bool:
    #    return ModuleLoadParameters(self.__module_ptr, root, filename)
    #
    #def register_backward_hook(self, func):
    #    ModuleRegistBackwardHook(self.__session_ptr, self.__module_ptr, func)
    # 2023.01.12 : Not support : <--
