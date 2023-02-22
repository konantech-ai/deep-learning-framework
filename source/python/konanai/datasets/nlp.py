from .dataset import Dataset
from ..tensor import Tensor
from typing import Callable, Optional

import os
import numpy as np
from numpy import fromfile
import requests
import gzip

class NlpDataset(Dataset):
    def __init__(self, session, name: str, corpus:str, voc:str, max_position:int,
                 train: bool = True, download: bool = False, data_root: str or None = "data",
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(NlpDataset, self).__init__(session, name, train, download, data_root, None, transform, verbose, select, args)
        
        self.max_position = max_position;
        self.corpus_filename = corpus;
        self.voc_filename = voc;

        self.setData()
        self.voc_count = len(self.vocabulary)

    def __del__(self):
        pass

    @property
    def len(self) -> int:
        return len(self.corpus_pairs)

    def load_data(self, data_path:str, train:bool) -> None:
        cop_path = data_path + "/" + self.corpus_filename
        voc_path = data_path + "/" + self.voc_filename

        self.corpus_pairs = []
        self.vocabulary = []

        ext = os.path.splitext(self.corpus_filename)[-1]

        if ext == ".jsonl":
            self.corpus_pairs = self.session.parse_jsonl_file(cop_path);
            self.vocabulary = self.session.read_file_lines(voc_path);
        else:
            raise Exception(f"unknown file format '{ext}' was used for bert dataset instead of 'jsonl'")

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

class NlpBERT(NlpDataset):
    def __init__(self, session, name: str, corpus:str, voc:str, max_position:int,
                 train: bool = True, download: bool = False, data_root: str or None = "data",
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(NlpBERT, self).__init__(session, name, corpus, voc, max_position, train, download, data_root, transform, verbose, select, args)

    def getFieldInfo(self, is_x:bool) -> dict:
        sent_info = {"shape": (1,), "type": "int32" }
        word_info = {"shape": (self.max_position,), "type": "int32" }
        
        return {"#": word_info, "sent":word_info} if is_x else {"next_sent": sent_info, "mask_word":word_info}

    def getDataPair(self, idx, xsBuf, ysBuf, auxBuf):
        x = xsBuf["#"]
        sent = xsBuf["sent"]
        x.set_zero()
        sent.set_zero()

        if ysBuf is not None:
            mask_word = ysBuf["mask_word"]
            next_sent = ysBuf["next_sent"]
            mask_word.set_zero()
            next_sent.set_zero()

        for ndat, index in enumerate(idx):
            pair = self.corpus_pairs[index]
            
            labels = pair["train_data"]
            
            label_tokens = labels["input_ids"]
            mlm_tokens = labels["mlm_labels"]
            label_sents = labels["token_type_ids"]

            if ysBuf is not None:
                next_sent[ndat, 0] = labels["next_sentence_label"]

            length = len(label_tokens)
            if length > self.max_position: length = self.max_position

            for ntime in range(length):
                x[ndat, ntime] = label_tokens[ntime]
                sent[ndat, ntime] = label_sents[ntime]
                if ysBuf is not None:
                    mask_word[ndat, ntime] = mlm_tokens[ntime]

class NlpGPT3(NlpDataset):
    def __init__(self, session, name: str, corpus:str, voc:str, max_position:int,
                 train: bool = True, download: bool = False, data_root: str or None = "data",
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(NlpGPT3, self).__init__(session, name, corpus, voc, max_position, train, download, data_root, transform, verbose, select, args)

    def getFieldInfo(self, is_x:bool) -> dict:
        word_info = {"shape": (self.max_position,), "type": "int32" }
        return {"#": word_info}

    def getDataPair(self, idx, xsBuf, ysBuf, auxBuf):
        x = xsBuf["#"]
        next_word = ysBuf["#"]

        x.set_zero()
        next_word.set_zero()

        for ndat, index in enumerate(idx):
            pair = self.corpus_pairs[index]
            
            labels = pair["train_data"]
            
            label_tokens = labels["input_ids"]
            mlm_tokens = labels["mlm_labels"]

            length = len(label_tokens)
            if length > self.max_position: length = self.max_position

            for ntime in range(length):
                token = label_tokens[ntime] if mlm_tokens[ntime] == -100 else mlm_tokens[ntime]
                x[ndat, ntime] = token
                if ntime < self.max_position-1: next_word[ndat, ntime+1] = token

class NlpTransformer(NlpDataset):
    def __init__(self, session, name: str, corpus:str, voc:str, max_position:int,
                 train: bool = True, download: bool = False, data_root: str or None = "data",
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(NlpTransformer, self).__init__(session, name, corpus, voc, max_position, train, download, data_root, transform, verbose, select, args)

    def getFieldInfo(self, is_x:bool) -> dict:
        word_info = {"shape": (self.max_position,), "type": "int32" }
        return {"#": word_info, "last_token":word_info} if is_x else {"#": word_info}

    def getDataPair(self, idx, xsBuf, ysBuf, auxBuf):
        x = xsBuf["#"]
        last_word = xsBuf["last_token"]
        curr_word = ysBuf["#"]

        x.set_zero()
        last_word.set_zero()
        curr_word.set_zero()

        for ndat, index in enumerate(idx):
            pair = self.corpus_pairs[index]
            
            labels = pair["train_data"]
            
            label_tokens = labels["input_ids"]
            mlm_tokens = labels["mlm_labels"]

            length = len(label_tokens)

            second_pos = -1

            for n, ntime in enumerate(range(length)):
                if n >= self.max_position: break
                token = label_tokens[ntime] if mlm_tokens[ntime] == -100 else mlm_tokens[ntime]
                if token == 3:    # 3 means [SEP]
                    if second_pos < 0:
                        second_pos = 0
                    else:
                        break
                else:
                    if second_pos < 0:
                        #print(f"x.shape: {x.shape}, ndat:{ndat}, ntime:{ntime}, token:{token}")
                        x[ndat, ntime] = token
                    elif second_pos >= self.max_position:
                        break
                    else:
                        curr_word[ndat, second_pos] = token
                        if second_pos < self.max_position - 1: last_word[ndat, second_pos+1] = token
