from .api import CreateTensor
from ._global_variables import GlobalVariables as gvar
from .datasets import Dataset
from .tensor import Tensor
from typing import Any, Optional
import numpy as np
import math


class DataLoader:
    """
    | This class is DataLoader Class ( Python ).
    | This class serve to control Dataset.

    Parameters
    ----------
    | ``data_src`` (`Type : Dataset or ...`)
    |   Input Dataset Class
    | ``batch_size`` (`Type : Optional[int]`)
    |   Default - 0
    | ``select`` : (`Type : str`)
    |   Default - "random"
    | ``shuffle`` : (`Type : Optional[bool]`)
    |   Default - "True"
    | ``use_count`` : (`Type : int`)
    |   Default - 0
    | ``ratio`` : (`Type : float`)
    |   Default - 1.0
    | ``data_balancing`` : (`Type : bool`)
    |   Default - False
    | ``args`` : (`Type : Optional[dict]`)
    |   Default - {}
    """
    def __init__(self, data_src: Dataset or ..., batch_size: Optional[int] = 0, select: str = "random", shuffle: Optional[bool] = True, 
                 use_count: int = 0, ratio:float=1, data_balancing: bool = False, args: Optional[dict] = {}):
        self.batch_size = batch_size
        self.use_count = use_count
        self.args = args

        if isinstance(data_src, Dataset):
            self.m_init_with_dataset(data_src)
        elif isinstance(data_src, DataLoader):
            self.m_init_with_dataloader(data_src)
        else:
            raise Exception("Unknown datasrc type for Dataloder")

        if select == "random":
            np.random.shuffle(self.data_idx)
        elif select != "sequential":
            raise Exception("invalid select option for dataloader")

        self.raw_data_count = int(self.full_count * ratio)
        if 0 < use_count < self.raw_data_count: self.raw_data_count = use_count

        if data_balancing:
            self.shuffle_idx = self._createBalancedIndex()
        else:
            self.shuffle_idx = np.arange(self.raw_data_count);

        self.xs_buf = None
        self.ys_buf = None

    def m_init_with_dataset(self, data_src):
        """
        | Init DataLoader with Dataset. 
        Parameters
        ----------
        | ``data_src`` (`Type : Dataset`)
        |   Input Dataset Class
        Returns
        -------
        | None
        """
        if self.batch_size <= 0: self.batch_size = 32

        self.dataset = data_src
        self.full_count = self.dataset.len
        self.data_idx = self.dataset.data_idx


    def m_init_with_dataloader(self, data_src):
        """
        | Init DataLoader with DataLoader. 
        Parameters
        ----------
        | ``data_src`` (`Type : DataLoader`)
        |   Input DataLoader Class
        Returns
        -------
        | None
        """
        if self.batch_size <= 0:
           self.batch_size = data_src.batch_size

        if data_src.full_count - data_src.raw_data_count <= 0:
           raise Exception("source dataloader has no unusing data")

        self.dataset = data_src.dataset
        self.full_count = data_src.full_count - data_src.raw_data_count
        self.data_idx = self.dataset.data_idx[data_src.raw_data_count:]

    def _createBalancedIndex(self):
        """
        | For data balancing, create data.
        Parameters
        ----------
        | None
        Returns
        -------
        | None
        """
        if hasattr(self.dataset,'y') == False or self.dataset.y is None:
            idxs = self.data_idx[:self.raw_data_count]
            return self.dataset.createBalancedIndex(idxs)

        distribution, y_idx = self._extract_data_distribution(True)
        
        curr_pos = {}
        max_freq = 0
        for key, freq in distribution.items():
            if freq > max_freq: max_freq = freq
            curr_pos[key] = 0

        balanced_idxs = []    
        pos = 0

        idxs = self.data_idx[:self.raw_data_count]
        y_width = self.dataset.y.shape[1]

        for nth, n in enumerate(idxs):
            y_elem = y_idx[int(n)].item()
            y_desc = self.dataset.yHeaders[int(y_elem)]
            freq = distribution[y_desc]
            quater = max_freq // freq
            rest = max_freq % freq
            if curr_pos[y_desc] < rest: quater += 1
            for k in range(quater):
                balanced_idxs.append(nth)
                pos += 1
            curr_pos[y_desc] += 1
            
        return balanced_idxs

    def dump_data_distribution(self, title="data distribution"):
        """
        | Dump Data Distribution.
        Parameters
        ----------
        | ``title`` (`Type : str`)
        |   Default - "data distribution"
        Returns
        -------
        | None
        """
        dist = self.extract_data_distribution()
        print(f"*** {title} ***")
        for key, freq in dist.items():
            print(f" * {key}: {freq} items")
    
    def extract_data_distribution(self):
        """
        | Extract Data Distribution.
        Parameters
        ----------
        | None
        Returns
        -------
        | Python dict
        """
        if hasattr(self.dataset,'y') == False or self.dataset.y is None:
            idxs = self.data_idx[self.shuffle_idx]
            return self.dataset.extract_data_distribution_for_idxs(idxs)

        distribution, _ = self._extract_data_distribution(False)
        return distribution
    
    def _extract_data_distribution(self, in_real_data):
        """
        | Extract Data Distribution.
        Parameters
        ----------
        | ``in_real_data`` (`Type : bool`)
        |   Default - None
        Returns
        -------
        | Python dict, Python list
        """
        if hasattr(self.dataset,'yHeaders') == False:
            raise Exception("header list for y values doesn't exist")
        if len(self.dataset.yHeaders) <= 1:
            raise Exception("header list for y values doesn't have multiple entrues")

        dict = {}
        y_width = self.dataset.y.shape[1]
        
        idxs = self.data_idx[:self.raw_data_count] if in_real_data else self.data_idx[self.shuffle_idx]

        y_idx = self.dataset.y if y_width == 1 else self.dataset.y.argmax(axis=1)
        for n in idxs:
            y_elem = y_idx[int(n)].item()
            y_desc = self.dataset.yHeaders[int(y_elem)]
            if y_desc in dict.keys():
                dict[y_desc] += 1
            else:
                dict[y_desc] = 1
        return dict, y_idx

    def __del__(self):
        pass

    def __len__(self) -> int:
        return self.batch_count
    
    @property
    def data_count(self):
        """
        | Get Data Count.
        Parameters
        ----------
        | None
        Returns
        -------
        | Python int 
        """

        return len(self.shuffle_idx)

    @property
    def batch_count(self):
        """
        | Get Batch Count.
        Parameters
        ----------
        | None
        Returns
        -------
        | Python int 
        """

        return math.ceil(len(self.shuffle_idx) / self.batch_size)

    def get_arg(self, key: str, default: Any = None):
        """
        | Get argument.
        Parameters
        ----------
        | ``key`` (`Type : str`)
        |   Default - None
        | ``default`` (`Type : Any`)
        |   Default - None
        Returns
        -------
        | Python dict value
        """

        return self.args.get(key, default)

    def shuffle(self):
        """
        | Dataset Index Shuffle
        Parameters
        ----------
        | None
        Returns
        -------
        | None
        """

        np.random.shuffle(self.shuffle_idx)

    #### Original code ################################
    # def shape(self, is_x: bool) -> tuple:
    #     if self.dataset is None:
    #         raise Exception("Dataset not found.")
        
    #     info = self.dataset.get_info(is_x)
    #     return info['shape']

    # def dtype(self, is_x: bool) -> str:
    #     if self.dataset is None:
    #         raise Exception("Dataset not found.")
        
    #     info = self.dataset.get_info(is_x)
    #     return info['dtype']
    ###################################################

    #### Fixed to new style ###########################
    def shape(self, is_x: bool) -> list:
        """
        | Get Shape
        Parameters
        ----------
        | ``is_x`` (`Type : bool`)
        |   Default - None
        Returns
        -------
        | Python List
        """
        if self.dataset is not None:
            fieldInfo = self.dataset.getFieldInfo(is_x)
            field = fieldInfo["#"]
            return field["shape"]
        else:
            data = self.x if is_x else self.y
            return data.shape[1:]

    def dtype(self, is_x: bool) -> str:
        """
        | Get Data Type
        Parameters
        ----------
        | ``is_x`` (`Type : bool`)
        |   Default - None
        Returns
        -------
        | Python str
        """
        if self.dataset is not None:
            fieldInfo = self.dataset.getFieldInfo(is_x)
            field = fieldInfo["#"]
            return field["type"]
        else:
            data = self.x if is_x else self.y
            return data.dtype()
    ###################################################

    def __iter__(self):
        # Initialize for iteration
        self.batch_curr_index = 0
    
        #### (Added) Fixed to new style ###################
        if self.xs_buf is None:
            self.xs_buf = self._create_buffer(self.batch_size, True)
            self.ys_buf = self._create_buffer(self.batch_size, False)
            self.aux_buf = self.dataset.createAuxBuffer(self.batch_size)
        ###################################################

        return self
    
    def __next__(self):
        # Check the end point
        if self.batch_count <= 0 or self.batch_curr_index >= self.batch_count:
            raise StopIteration

        # Get the batch data
        # Issue (2023-01-11) :
        #   to use Excution-Tracer, reuse same buffer for each minibatch data.
        #   마지막 배치는 형상이 다르므로 버퍼 재생성이 필요하지만,
        #   현재는 동일한 버퍼 크기로 작업하기 위해 맨 끝에서 batch_size 만큼을 잘라 마지막 배치로 사용중.
        #   이로 인해 accuracy가 원래 결과보다 조금 높게 나옴
        # Issue (2023-01-12) :
        #   __init__()에서 index만 shuffle하고, 배치 루프에서 데이터를 찾는 방식이기 때문에
        #   데이터 접근 주소가 sequential하지 못하여 성능 최적화가 필요함
        from_idx = self.batch_curr_index * self.batch_size
        to_idx = from_idx + self.batch_size
        if to_idx > self.data_count:
        #if to_idx > self.use_count:
            to_idx = self.data_count
            #to_idx = self.use_count
            from_idx = to_idx - self.batch_size
            
        #### Original code ################################
        # (x, y) = self.dataset.get_data(self.data_idx[from_idx:to_idx])

        # # Set the data to excution tracer.
        # # Issue (2023-01-11) :
        # #   x_piece 및 y_piece에 매번 copy_data()로 데이터를 담고 있어 성능 저하 원인이 됨
        # if self.x_piece is None or self.x_piece.shape != x.shape:
        #     if self.x_piece is not None:
        #         print('ours', self.x_piece.shape())
        #     self.x_piece = x
        # else:
        #     self.x_piece.copy_data(x)

        # if self.y_piece is None or self.y_piece.shape != y.shape:
        #     self.y_piece = y
        # else:
        #     self.y_piece.copy_data(y)

        # # Update the index
        # self.batch_curr_index += 1

        # return (self.x_piece, self.y_piece)
        ###################################################

        #### Fixed to new style ###########################
        idxes = self.data_idx[self.shuffle_idx[from_idx:to_idx]]
        
        self.dataset.getDataPair(idxes, self.xs_buf, self.ys_buf, self.aux_buf)

        self.batch_curr_index += 1

        return (self.xs_buf, self.ys_buf)
        ###################################################
    
    def get_random_data(self, batch_size=0):
        """
        | Get Random Data
        Parameters
        ----------
        | ``batch_size`` (`Type : int`)
        |   Default - 0
        Returns
        -------
        | Python dict
        """

        if batch_size <= 0: batch_size = self.batch_size
        idx = np.random.randint(0, self.data_count, [batch_size])
        data_idx = self.shuffle_idx[idx]

        xs_buf = self._create_buffer(batch_size, True)
        aux_buf = self.dataset.createAuxBuffer(batch_size)
        
        self.dataset.getDataPair(data_idx, xs_buf, None, aux_buf)

        return xs_buf

    #### (Added) Fixed to new style ###################
    def _create_buffer(self, batch_size, is_x):
        buf_dict = {}
        if self.dataset is not None:
            field_info = self.dataset.getFieldInfo(is_x)
            for key in field_info.keys():
                field = field_info[key]
                shape = (batch_size,) + field["shape"]
                buf_dict[key] = Tensor(CreateTensor(gvar.get_session_ptr(), shape, field["type"], None))
        else:
            data = self.x if is_x else self.y
            shape = data.shape.replace_head(batch_size)
            buf_dict["#"] = Tensor(CreateTensor(gvar.get_session_ptr(), shape, data.dtype(), None))
        
        return buf_dict
    ###################################################

