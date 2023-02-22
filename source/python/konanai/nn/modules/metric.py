from ...api import *
from ..._global_variables import GlobalVariables as gvar
from ...tensor import Tensor
from typing import Optional


class Metric:
    """
    This class is Metric - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    Default class for create Metric function module.

    Parameters
    ----------
    `session_ptr`: ctypes.c_void_p
        Target session pointer( C++ engine ).
    `metric_ptr`: ctypes.c_void_p
        Target metric function pointer( C++ engine ).
    `metric_name`: str
        Target metric function name.
    `est`: str
        Target est.
    `args`: dict
        Arguments for initialize Metric function.
    """
    def __init__(self, session_ptr: ctypes.c_void_p, metric_ptr: ctypes.c_void_p, metric_name: str, est: str, args: Optional[dict] = {}):
        self.__session_ptr = session_ptr
        self.__metric_ptr = metric_ptr
        self.__metric_name = metric_name
        self.est = est
        self.args = args
    
    @classmethod
    def from_metric_name(cls, metric_name: str, est: str, args: Optional[dict] = {}):
        """
        For general users

        Parameters
        ----------
        `metric_name`: str
            Target metric function name.
        `est`: str
            Target est.
        `args`: dict
            Arguments for initialize Metric function.
        """
        session_ptr = gvar.get_session_ptr()
        metric_ptr = CreateMetric(session_ptr, metric_name, est, args)
        return cls(session_ptr, metric_ptr, metric_name, est, args)
    
    def _init_from_metric_name(self, metric_name: str, est: str, args: Optional[dict] = {}):
        """
        For initializing derived classes

        Parameters
        ----------
        `metric_name`: str
            Target metric function name.
        `est`: str
            Target est.
        `args`: dict
            Arguments for initialize Metric function.
        """
        self.__session_ptr = gvar.get_session_ptr()
        self.__metric_ptr = CreateMetric(self.__session_ptr, metric_name, est, args)
        self.__metric_name = metric_name
        self.est = est
        self.args = args
        
    @classmethod
    def from_metric_dict(cls, children: dict[str,'Metric']):
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
        metric_ptr = CreateMultipleMetric(session_ptr, children_ptrs)
        return cls(session_ptr, metric_ptr, "", "", "")
    
    def _init_from_metric_dict(self, children: dict[str,'Metric']):
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
        self.__metric_ptr = CreateMultipleMetric(self.__session_ptr, children_ptrs)
        self.__metric_name = ""
        self.est = ""
        self.args = {}

    def __del__(self):
        if DeleteMetric(self.__metric_ptr) is False:
            raise Exception("error: Failed to delete the Metric object.")

    def get_core(self) -> ctypes.c_void_p:
        """
        Get void pointer from Metric - Capsule Class( C++ engine ).
        This pointer serve to connect 'Deep Learning Framework Engine'.

        Returns
        -------
        ctypes.c_void_p
        """
        return self.__metric_ptr

    def __call__(self, pred: Tensor or dict) -> Tensor or dict:
        return self._evaluate(pred)

    def evaluate(self, pred: Tensor or dict) -> Tensor or dict:
        return self._evaluate(pred)

    def evaluate_extend(self, pred: Tensor or dict) -> Tensor or dict:
        result = self._evaluate(pred)
        if self.metricTermKeys is None:
            return (result, {})
        else:
            metrices = {}
            subterms = {}
            for key in result.keys():
                if key in self.metricTermKeys:
                    metrices[key] = result[key]
                else:
                    subterms[key] = result[key]
            return (metrices, subterms)

    def _evaluate(self, pred: Tensor or dict) -> Tensor or dict:
        if isinstance(pred, dict):
            metric = MetricEvaluateDict(self.__metric_ptr, pred)
        else:
            metric = MetricEvaluate(self.__metric_ptr, pred.get_core())

        if isinstance(metric, dict):
            return self._recv_tensor_dict(metric)
        else:
            return Tensor(metric)
    
    def _recv_tensor_dict(self, x):
        xs = {}
        for key in x.keys():
            xs[key] = Tensor(x[key])
        return xs;

    def get_proc_count(self, preds: dict) -> dict:
        pred = self.seek_matched_tensor(self.est, 'pred', preds)
        return {"#":pred.size}

    def seek_matched_tensor(self, name:str, def_name:str, tensors:dict) -> Tensor:
        if name == def_name: return tensors["#"]
        if name in tensors.keys(): return tensors[name]
        pos = name.find('::')
        if pos >= 0:
            core_name = name[pos+2:]
            if core_name in tensors.keys(): return tensors[core_name]
        raise Exception(f"cannot find the '{name}' term in the tensors used for metric evaluation")

# Original
class FormulaMetric(Metric):
    metric_name = 'formula'
    
    def __init__(self, name:str, formula:str, **kwargs):
        metric_ptr = CreateFormulaMetric(session_ptr, name, formula, kwargs)
        super(FormulaMetric, self).__init__(session_ptr, metric_ptr, 'formula', '', '')
        
    def __del__(self):
        super(MultipleMetric, self).__del__()

    def get_proc_count(self, preds: dict, ys: dict) -> int:
        proc_count = {}
        for key in self.metricTermKeys:
            tensor = self.seek_matched_tensor(key, key, preds)
            proc_count[key] = tensor.size
        return proc_count

# Original
class MultipleMetric(Metric):
    """
    This class is MultipleMetric - Capsule Class( Python ).
    This class serve to connect 'Deep Learning Framework Engine'.
    For MultipleMetric calculating module.
    """
    metric_name = 'multiple'
    
    def __init__(self, children: dict):
        super(MultipleMetric, self)._init_from_metric_dict(children)
        
    def __del__(self):
        super(MultipleMetric, self).__del__()

    def get_proc_count(self, pred: Tensor, y: Tensor or np.ndarray) -> dict:
        proc_count = {}
        for key in self.children:
            proc_count[key] = self.children[key].get_proc_count(pred, y)["#"]
        return proc_count

# Original
class CustomMetric(Metric):
    metric_name = 'custom'

    def __init__(self, terms, static={}, **kwargs):
        session_ptr = gvar.get_session_ptr()
        metric_ptr = CreateCustomMetric(session_ptr, terms, static, kwargs)
        super(CustomMetric, self).__init__(session_ptr, metric_ptr, 'custom', '', '')
        self.metricTermKeys = terms.keys()
        
    def __del__(self):
        super(CustomMetric, self).__del__()

    def get_proc_count(self, preds: dict, ys: dict) -> int:
        proc_count = {}
        for key in self.metricTermKeys:
            tensor = self.seek_matched_tensor(key, key, preds)
            proc_count[key] = tensor.size
        return proc_count

