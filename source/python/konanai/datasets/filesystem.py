from .dataset import Dataset
from ..tensor import Tensor
from typing import Callable, Optional

import os
import fnmatch
import numpy as np
import cv2

class FileSystemDataset(Dataset):
    def __init__(self, session, name: str, filter: str, image_shape: tuple, train: bool = True, download: bool = False,
                 data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(FileSystemDataset, self).__init__(session, name, train, download, data_root, cache_root, transform, verbose, select, args)
        self.filter = filter
        self.image_shape = image_shape
        self.setData()
        self.y = None

    def __del__(self):
        pass

    @property
    def len(self):
        return len(self.filelist)

    @property
    def get_output_width(self):
        width = 0
        for domain, info in self.domain_info.items():
            width += len(info["categories"])
        return width

    @property
    def yHeaders(self):
        if len(self.domain_info) > 1:
            raise Exception("not yet")
        domain = self.domain_list[0]
        return self.domain_info[domain]["categories"]

    def get_category_count(self, domain):
        return len(self.domain_info[domain]["categories"])

    def load_data(self, data_path:str, train:bool) -> None:
        self.filelist = []
        self.domain_list = []
        self.domain_info = {}
        pieces = self.filter.split('/')
        self.collect_data(data_path, pieces, 0, {})

    def collect_data(self, data_path, pieces, depth, domcat_idx):
        piece = pieces[depth];
        
        if depth == len(pieces) - 1:
            for filename in os.listdir(data_path):
                if fnmatch.fnmatch(filename, piece):
                    subpath = os.path.join(data_path, filename)
                    if os.path.isfile(subpath):
                        self.filelist.append({"path":subpath, "domcat":domcat_idx.copy()})
                        for domain, category_idx in domcat_idx.items():
                            self.domain_info[domain]["file_counts"][category_idx] += 1
        elif piece[0] == '<':    # 도메인을 지정하는 중간 경로
            if piece[-1] != '>':
                raise Exception(f"unmatched path domain piece: {piece}")
            
            domain = piece[1:-1]
            
            if domain not in self.domain_info.keys():
                domain_idx = len(self.domain_list)
                self.domain_list.append(domain)
                self.domain_info[domain] = {"idx":domain_idx, "categories":[], "file_counts":[]}

            for filename in os.listdir(data_path):
                subpath = os.path.join(data_path, filename)
                if not os.path.isfile(subpath):
                    category = filename
                    if category in self.domain_info[domain]["categories"]:
                        category_idx = self.domain_info[domain]["categories"].index(category)
                        #print(f"category_idx:{category_idx}, category:{category}")
                        #raise Exception("here")
                    else:
                        category_idx = len(self.domain_info[domain]["categories"])
                        self.domain_info[domain]["categories"].append(category)
                        self.domain_info[domain]["file_counts"].append(0)
                    domcat_idx[domain] = category_idx
                    self.collect_data(subpath, pieces, depth+1, domcat_idx)
        elif piece.find('*') < 0:   # 단순경로: 통과해 지정된 폴더로 이동
            data_path = data_path + "/" + piece
            self.collect_data(data_path, pieces, depth+1, domcat_idx)
        else:
            raise Exception("unprepared case: intermediate pth cannot contain '*' in current version")

    def dump_data_distribution(self, title="data distribution"):
        for domain in self.domain_list:
            dist = self.extract_data_distribution(domain)
            print(f"*** [{domain}]: {title} ***")
            for key, freq in dist.items():
                print(f" * {key}: {freq} items")

    def extract_data_distribution(self, domain):
        dist = {}
        domain_info = self.domain_info[domain]
        for n, category in enumerate(domain_info['categories']):
            dist[category] = domain_info['file_counts'][n]
        return dist
    
    def extract_data_distribution_for_idxs(self, idxs):
        distribution = {}
        
        for n in idxs:
            y_domcat = self.filelist[n]["domcat"]
            y_descs = []
            for domain, cat_idx in y_domcat.items():
                catregories = self.domain_info[domain]["categories"]
                y_descs.append(catregories[cat_idx])
            y_desc = "/".join(y_descs)
            if y_desc in distribution.keys():
                distribution[y_desc] += 1
            else:
                distribution[y_desc] = 1
        return distribution
    
    def createBalancedIndex(self, idxs):
        if len(self.domain_info) > 1:
            raise Exception("not yet")
        
        distribution = self.extract_data_distribution_for_idxs(idxs)
        
        domain = self.domain_list[0]
        catregories = self.domain_info[domain]["categories"]

        curr_pos = {}
        max_freq = 0
        for key, freq in distribution.items():
            if freq > max_freq: max_freq = freq
            curr_pos[key] = 0

        balanced_idxs = []    
        pos = 0

        for nth, n in enumerate(idxs):
            y_elem = self.filelist[n]["domcat"][domain]
            y_desc = catregories[y_elem]
            freq = distribution[y_desc]
            quater = max_freq // freq
            rest = max_freq % freq
            if curr_pos[y_desc] < rest: quater += 1
            for k in range(quater):
                balanced_idxs.append(nth)
                pos += 1
            curr_pos[y_desc] += 1
            
        return balanced_idxs

    def getFieldInfo(self, is_x:bool) -> dict:
        if is_x:
            info = {"shape": self.image_shape, "type": "float32"}
            return { "#": {"shape": self.image_shape, "type": "float32"}}
        elif len(self.domain_list) == 1 and not self.get_arg("use_domain", False):
            domain = self.domain_list[0]
            cat_cnt = self.get_category_count(domain)
            return { "#": {"shape": (cat_cnt,), "type": "float32"} }
        else:
            fields = {}
            for domain in self.domain_list:
                cat_cnt = self.get_category_count(domain)
                fields[domain] = {"shape": (cat_cnt,), "type": "float32"}
            return fields

    def createAuxBuffer(self, batch_size):
        imgBuf = self.session.createTensor(self.image_shape, "float32")
        return imgBuf

    def getDataPair(self, idxs, xs_buf: dict[str,Tensor], ys_buf: dict[str,Tensor], aux_buf: Tensor):
        if ys_buf is not None:
            if len(self.domain_list) == 1 and not self.get_arg("use_domain", False):
                domain = self.domain_list[0]
                ys_buf["#"].set_zero()
            else:
                for domain in self.domain_list:
                    ys_buf[domain].set_zero()

        for nth, idx in enumerate(idxs):
            fileinfo = self.filelist[idx]
            filepath = fileinfo["path"]
            img = cv2.imread(filepath)
            img = cv2.resize(img,self.image_shape[0:2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 127.5 - 1.0
            aux_buf.from_ndarray(img)
            xs_buf["#"].copy_into_row(nth, aux_buf)

            if ys_buf is not None:
                if len(self.domain_list) == 1 and not self.get_arg("use_domain", False):
                    idx = fileinfo["domcat"][domain]
                    ys_buf["#"][nth, idx] = 1.0
                else:
                    for domain in self.domain_list:
                        idx = fileinfo["domcat"][domain]
                        ys_buf[domain][nth, idx] = 1.0

class Flowers5(FileSystemDataset):
    def __init__(self, session, image_shape: tuple, data_root: str or None = "data"):
        super(Flowers5, self).__init__(session, "flowers5", "flowers/<species>/*.jpg", image_shape, True, False, data_root, None)

class Office31(FileSystemDataset):
    def __init__(self, session, image_shape: tuple, data_root: str or None = "data"):
        super(Office31, self).__init__(session, "office31", "<source>/images/<product>/*.jpg", image_shape, True, False, data_root, None)
