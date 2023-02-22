from ...api import *
from ..._global_variables import GlobalVariables as gvar
from ...tensor import Tensor
from typing import Optional


class Loss:
    """
    This class is Loss - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Default class for create Loss function module.

    Parameters
    ----------
    `session_ptr`: ctypes.c_void_p
        Target session pointer( C++ engine ).
    `loss_ptr`: ctypes.c_void_p
        Target loss function pointer( C++ engine ).
    `loss_name`: str
        Target loss function name.
    `est`: str
        Target est.
    `ans`: str
        Target ans.
    `args`: dict
        Arguments for initialize Loss function.
    """
    def __init__(self, session_ptr: ctypes.c_void_p, loss_ptr: ctypes.c_void_p, loss_name: str, est: str, ans: str, args: Optional[dict] = {}):
        self.__session_ptr = session_ptr
        self.__loss_ptr = loss_ptr
        self.__loss_name = loss_name
        self.est = est
        self.ans = ans
        self.args = args
    
    @classmethod
    def from_loss_name(cls, loss_name: str, est: str, ans: str, args: Optional[dict] = {}):
        """
        For general users

        Parameters
        ----------
        `loss_name`: str
            Target loss function name.
        `est`: str
            Target est.
        `ans`: str
            Target ans.
        `args`: dict
            Arguments for initialize Loss function.
        """
        session_ptr = gvar.get_session_ptr()
        loss_ptr = CreateLoss(session_ptr, loss_name, est, ans, args)
        return cls(session_ptr, loss_ptr, loss_name, est, ans, args)
    
    def _init_from_loss_name(self, loss_name: str, est: str, ans: str, args: Optional[dict] = {}):
        """
        For initializing derived classes

        Parameters
        ----------
        `loss_name`: str
            Target loss function name.
        `est`: str
            Target est.
        `ans`: str
            Target ans.
        `args`: dict
            Arguments for initialize Loss function.
        """
        self.__session_ptr = gvar.get_session_ptr()
        self.__loss_ptr = CreateLoss(self.__session_ptr, loss_name, est, ans, args)
        self.__loss_name = loss_name
        self.est = est
        self.ans = ans
        self.args = args
        
    @classmethod
    def from_loss_dict(cls, children: dict[str,'Loss']):
        """
        For general users

        Parameters
        ----------
        `children`: dict
            Target children.
        """
        session_ptr = gvar.get_session_ptr()
        children_ptrs = {}
        for key in children.keys():
            children_ptrs.update( {key:children[key].get_core()} )            
        loss_ptr = CreateMultipleLoss(session_ptr, children_ptrs)
        return cls(session_ptr, loss_ptr, "", "", "")
    
    def _init_from_loss_dict(self, children: dict[str,'Loss']):
        """
        For initializing derived classes

        Parameters
        ----------
        `children`: dict
            Target children.
        """
        self.__session_ptr = gvar.get_session_ptr()
        self.children = children
        children_ptrs = {}
        for key in children.keys():
            children_ptrs.update( {key:children[key].get_core()} )            
        self.__loss_ptr = CreateMultipleLoss(self.__session_ptr, children_ptrs)
        self.__loss_name = ""
        self.est = ""
        self.ans = ""
        self.args = {}

    def __del__(self):
        if DeleteLoss(self.__loss_ptr) is False:
            raise Exception("error: Failed to delete the Loss object.")

    def get_core(self) -> ctypes.c_void_p:
        """
        Get void pointer from Loss - Capsule Class( C++ engine ).
        This pointer serve to connect 'Deep Learning Framework Engine'.

        Returns
        -------
        ctypes.c_void_p
        """
        return self.__loss_ptr

    def backward(self) -> None:
        LossBackward(self.__loss_ptr)

    def __call__(self, pred: Tensor or dict, y: Tensor or np.ndarray or dict) -> Tensor or dict:
        return self._evaluate(pred, y, False)

    def evaluate(self, pred: Tensor or dict, y: Tensor or np.ndarray or dict) -> Tensor or dict:
        return self._evaluate(pred, y, False)

    def evaluate_extend(self, pred: Tensor or dict, y: Tensor or np.ndarray or dict) -> Tensor or dict:
        result = self._evaluate(pred, y, True)
        if self.lossTermKeys is None:
            return (result, {})
        else:
            losses = {}
            subterms = {}
            for key in result.keys():
                if key in self.lossTermKeys:
                    losses[key] = result[key]
                else:
                    subterms[key] = result[key]
            return (losses, subterms)

    def _evaluate(self, pred: Tensor or dict, y: Tensor or np.ndarray or dict, download_all:bool) -> Tensor or dict:
        if isinstance(pred, dict) or isinstance(y, dict):
            if not isinstance(pred, dict): raise Exception("unmatching pair: pred is not a dict when y is a dict")
            if not isinstance(y, dict):
                if isinstance(y, float):
                    raise Exception("here: need to support float-y")
                else:
                    raise Exception("unmatching pair: pred is a dict when y is not a dict nor float constant")
            loss = LossEvaluateDict(self.__loss_ptr, pred, y, download_all)
        else:
            if type(y) == np.ndarray:
                y = Tensor(CreateTensorFromNumPy(self.__session_ptr, y))
            loss = LossEvaluate(self.__loss_ptr, pred.get_core(), y.get_core(), download_all)

        if isinstance(loss, dict):
            return self._recv_tensor_dict(loss)
        else:
            return Tensor(loss)

    def eval_accuracy(self, pred: Tensor, y: Tensor or np.ndarray) -> Tensor or dict:
        """
        Evaluate accuracy, prediction(Tensor) and answer(Tensor or NumPy).


        Returns
        -------
        Tensor
        """
        if isinstance(pred, dict) or isinstance(y, dict):
            if not isinstance(pred, dict): raise Exception("unmatching pair: pred is not a dict when y is a dict")
            if not isinstance(y, dict): raise Exception("unmatching pair: pred is a dict when y is not a dict")
            acc = LossEvalAccuracyDict(self.__loss_ptr, pred, y)
        else:
            if type(y) == np.ndarray:
                y = Tensor(CreateTensorFromNumPy(self.__session_ptr, y))
            acc = LossEvalAccuracy(self.__loss_ptr, pred.get_core(), y.get_core())

        if isinstance(acc, dict):
            return self._recv_tensor_dict(acc)
        else:
            return Tensor(acc)
    
    def _recv_tensor_dict(self, x):
        xs = {}
        for key in x.keys():
            xs[key] = Tensor(x[key])
        return xs;

    def get_proc_count(self, preds: dict, ys: dict) -> dict:
        #y = ys[self.ans] if self.ans != 'y' else ys["#"]
        y = self.seek_matched_tensor(self.ans, 'y', ys)
        return {"#":y.size}

    def seek_matched_tensor(self, name:str, def_name:str, tensors:dict) -> Tensor:
        if name == def_name: return tensors["#"]
        if name in tensors.keys(): return tensors[name]
        pos = name.find('::')
        if pos >= 0:
            core_name = name[pos+2:]
            if core_name in tensors.keys(): return tensors[core_name]
        raise Exception(f"cannot find the '{name}' term in the tensors used for loss evaluation")

class MSELoss(Loss):
    """
    This class is MSELoss - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For MSELoss calculating module.

    Parameters
    ----------
    `est`: str
        Target est.
    `ans`: str
        Target ans.
    `args`: dict
        Arguments for initialize Loss function.
    """
    loss_name = 'mse'
    
    def __init__(self, est: str = 'pred', ans: str = 'y', **kwargs):
        super(MSELoss, self)._init_from_loss_name(__class__.loss_name, est, ans, kwargs)

    def __del__(self):
        super(MSELoss, self).__del__()


class CrossEntropyLoss(Loss):
    """
    This class is MSELoss - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For MSELoss calculating module.

    Parameters
    ----------
    `est`: str
        Target est.
    `ans`: str
        Target ans.
    `args`: dict
        Arguments for initialize Loss function.
    """
    loss_name = 'crossentropy'
    
    def __init__(self, est: str = 'pred', ans: str = 'y', **kwargs):
        super(CrossEntropyLoss, self)._init_from_loss_name(__class__.loss_name, est, ans, kwargs)

    def __del__(self):
        super(CrossEntropyLoss, self).__del__()

    def get_proc_count(self, preds: dict, ys: dict) -> int:
        #pred = preds[self.est] if self.est != 'pred' else preds["#"]
        #y = ys[self.ans] if self.ans != 'y' else ys["#"]
        pred = self.seek_matched_tensor(self.est, 'pred', preds)
        y = self.seek_matched_tensor(self.ans, 'y', ys)
        ysize = y.size
        if pred.size != ysize: return {"#":ysize}
        return {"#": ysize / y.shape[-1]}


# Original
class BinaryCrossEntropyLoss(Loss):
    """
    This class is BinaryCrossEntropyLoss - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For BinaryCrossEntropyLoss calculating module.

    Parameters
    ----------
    `est`: str
        Target est.
    `ans`: str
        Target ans.
    `args`: dict
        Arguments for initialize Loss function.
    """
    loss_name = 'binary_crossentropy'
    
    def __init__(self, est: str = 'pred', ans: str = 'y', **kwargs):
        super(BinaryCrossEntropyLoss, self)._init_from_loss_name(__class__.loss_name, est, ans, kwargs)

    def __del__(self):
        super(BinaryCrossEntropyLoss, self).__del__()
        

# Original
class CrossEntropySigmoidLoss(Loss):
    """
    This class is CrossEntropySigmoidLoss - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For CrossEntropySigmoidLoss calculating module.

    Parameters
    ----------
    `est`: str
        Target est.
    `ans`: str
        Target ans.
    `args`: dict
        Arguments for initialize Loss function.
    """
    loss_name = 'crossentropy_sigmoid'
    
    def __init__(self, est: str = 'pred', ans: str = 'y', **kwargs):
        super(CrossEntropySigmoidLoss, self)._init_from_loss_name(__class__.loss_name, est, ans, kwargs)
        
    def __del__(self):
        super(CrossEntropySigmoidLoss, self).__del__()


# Original
class CrossEntropyPositiveIdxLoss(Loss):
    loss_name = 'crossentropy_pos_idx'

    def __init__(self, est: str = 'pred', ans: str = 'y', **kwargs):
        super(CrossEntropyPositiveIdxLoss, self)._init_from_loss_name(__class__.loss_name, est, ans, kwargs)
        
    def __del__(self):
        super(CrossEntropyPositiveIdxLoss, self).__del__()

    def get_proc_count(self, preds: dict, ys: dict) -> int:
        #y = ys[self.ans] if self.ans != 'y' else ys["#"]
        y = self.seek_matched_tensor(self.ans, 'y', ys)
        return {"#": UtilPositiveElementCount(y.get_core())}


# Original
class MultipleLoss(Loss):
    """
    This class is MultipleLoss - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For MultipleLoss calculating module.
    """
    loss_name = 'multiple'
    
    def __init__(self, children: dict):
        super(MultipleLoss, self)._init_from_loss_dict(children)
        
    def __del__(self):
        super(MultipleLoss, self).__del__()

    def get_proc_count(self, pred: Tensor, y: Tensor or np.ndarray) -> dict:
        proc_count = {}
        for key in self.children:
            proc_count[key] = self.children[key].get_proc_count(pred, y)["#"]
        return proc_count

# Original
class CustomLoss(Loss):
    loss_name = 'custom'

    def __init__(self, terms, static={}, **kwargs):
        session_ptr = gvar.get_session_ptr()
        loss_ptr = CreateCustomLoss(session_ptr, terms, static, kwargs)
        super(CustomLoss, self).__init__(session_ptr, loss_ptr, 'custom', '', '')
        self.lossTermKeys = terms.keys()
        
    def __del__(self):
        super(CustomLoss, self).__del__()

    def get_proc_count(self, preds: dict, ys: dict) -> int:
        proc_count = {}
        for key in self.lossTermKeys:
            tensor = self.seek_matched_tensor(key, key, preds)
            proc_count[key] = tensor.size
        return proc_count

