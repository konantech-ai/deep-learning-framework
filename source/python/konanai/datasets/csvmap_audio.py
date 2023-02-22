from ..session import Session
from .dataset import Dataset
from ..tensor import Tensor
from typing import Callable, Optional

import os
import numpy as np
from numpy import fromfile
import json

class CsvMapAudioDataset(Dataset):
    def __init__(self, session, name:str, train:bool = True, download: bool = False, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose:bool = False, select:str = "random", filename: str = "", args: Optional[dict] = {}):  
        # Issue (2023-01-11) : transform 미구현
        super(CsvMapAudioDataset, self).__init__(session, name, train, download, data_root, cache_root, transform, verbose, select, args)
        
        self.filename = filename

        # set label headers
        self.xHeaders = []
        self.yHeaders = []
        
        self.setData()

    def __del__(self):
        pass

    @property
    def len(self) -> int:
        return self.x.shape[0]

    @property
    def timesteps(self):
        return self.x.shape[1]

    @property
    def input_width(self):
        return self.x.shape[2]

    @property
    def output_width(self):
        return len(self.yHeaders)

    def load_data(self, data_path:str, train:bool) -> None:
        freq_in_spectrum = self.get_arg("freq_in_spectrum", 200) # 하나의 주파수 스펙트럼 벡터가 가질 주파수 갯수
        spec_interval = self.get_arg("spec_interval", 128)       # 주파수 스펙트럼 추출점 사이 간격에 해당하는 샘플 수
        fft_width = self.get_arg("fft_width", 2048)              # 각 주파수 스펙트럼 분석에 이용될 오디오 샘플의 수, 2이 멱수 권장
        spec_count = self.get_arg("spec_count", 800)             # 오디오 파일의 중앙 부위에서 추출할 스펙트럼 벡터 갯수

        audio_args = {"freq_in_spectrum":freq_in_spectrum, "spec_interval":spec_interval, "fft_width":fft_width, "spec_count":spec_count}

        audioReader = self.session.createAudioSpectrumReader(audio_args);

        map_filename = self.filename
        map_filepath = data_path + '/' + map_filename

        lines = self.session.read_file_lines(map_filepath)[1:]

        if self.verbose:
            print(f"[CsvmapAudioDataset] {map_filename} 파일 내용에 따라 {len(lines)} 개의 파일로부터 오디오 데이터 수집을 시작합니다.");

        cat_index = {}
        cat_names = []
        data_cats = []

        for n, line in enumerate(lines):
            id, cat = line.split(",")
            audio_filepath = data_path + f"/Train/{id}.wav"
            added = audioReader.add_file(audio_filepath)
            if added:
                if cat not in cat_names:
                    cat_index[cat] = len(cat_names)
                    cat_names.append(cat)
                data_cats.append(cat_index[cat])
            if self.verbose and n % 1000 == 999:
                print(f"[CsvmapAudioDataset] {n+1} 개의 파일이 조사되었습니다. {len(data_cats)}개가 유효합니다.");

        if self.verbose:
            print(f"[CsvmapAudioDataset] {len(lines)} 개의 파일 가운데 {len(data_cats)} 개의 파일 정보가 올바로 수집되었습니다. 스펙트럼 분석을 시작합니다.");

        self.x = audioReader.get_extract_spectrums()
        self.y = self.session.createTensor((len(data_cats),), "int32", data_cats)
        self.yHeaders = cat_names

    def load_cache(self, cachePath:str, train:bool) -> bool:
        postfix = ".train" if self.get_arg("train", False) else "";
        filepath = cachePath + "/" + self.name + postfix + ".cat";
        if not os.path.exists(filepath):
            return False
        self.yHeaders = self.session.parse_json_file(filepath)
        if not super(CsvMapAudioDataset, self).load_cache(cachePath, train): return False
        return True

    def save_cache(self, cachePath:str, train:bool) -> None:
        postfix = ".train" if self.get_arg("train", False) else "";
        filepath = cachePath + "/" + self.name + postfix + ".cat"
        with open(filepath, 'w') as outfile:
            json.dump(self.yHeaders, outfile)
        super(CsvMapAudioDataset, self).save_cache(cachePath, train)
        
class UrbanSound(CsvMapAudioDataset):
    def __init__(self, session, train:bool = True, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose:bool = False, select:str = "random",
                 filename: str = "train.csv", args: Optional[dict] = {}):  
        super(UrbanSound, self).__init__(session, "urbansound", train, False, data_root, cache_root, transform, verbose, select, filename, args)

class UrbanSoundTransformer(CsvMapAudioDataset):
    def __init__(self, session, train:bool = True, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose:bool = False, select:str = "random",
                 filename: str = "train.csv", args: Optional[dict] = {}):  
        super(UrbanSoundTransformer, self).__init__(session, "urbansound", train, False, data_root, cache_root, transform, verbose, select, filename, args)

    def getFieldInfo(self, is_x:bool) -> dict:
        info = super(CsvMapAudioDataset, self).getFieldInfo(is_x)
        if is_x:
            info["last_token"] = info["#"]
        return info

    def getDataPair(self, idx, xsBuf, ysBuf, auxBuf):
        super(CsvMapAudioDataset, self).getDataPair(idx, xsBuf, ysBuf, auxBuf)
        tsize = self.x.shape[1]
        xsBuf["last_token"][:,1:,:] = xsBuf["#"][:,:tsize-1,:]
