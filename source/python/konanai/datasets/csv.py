from .dataset import Dataset
from ..tensor import Tensor
from typing import Callable, Optional

import os
import numpy as np
from numpy import fromfile
import math
import json

class CSV(Dataset):
    def __init__(self, session, name: str, filename: str, has_header: bool = True,
                 xHeaders: list = None, yHeaders: list = None, 
                 x_columns: list = [], y_columns: list = [], ignore_columns: list = [], 
                 input_normalize: bool or list = False,
                 symbol_to_onehot: list = [],
                 symbol_to_onehotidx: list = [],
                 onehot_to_onehot: list = [],
                 onehot_to_onehotidx: list = [],
                 onehotidx_to_onehot: list = [],
                 onehotidx_to_onehotidx: list = [],
                 train: bool = True, download: bool = False, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(CSV, self).__init__(session, name, train, download, data_root, cache_root, transform, verbose, select, args)
        
        self.args["filename"] = filename
        self.args["has_header"] = has_header
        self.args["x_columns"] = x_columns
        self.args["y_columns"] = y_columns
        self.args["ignore_columns"] = ignore_columns
        self.args["input_normalize"] = input_normalize
        self.args["symbol_to_onehot"] = symbol_to_onehot
        self.args["symbol_to_onehotidx"] = symbol_to_onehotidx
        self.args["onehot_to_onehot"] = onehot_to_onehot
        self.args["onehot_to_onehotidx"] = onehot_to_onehotidx
        self.args["onehotidx_to_onehot"] = onehotidx_to_onehot
        self.args["onehotidx_to_onehotidx"] = onehotidx_to_onehotidx

        self.args["xHeaders"] = xHeaders
        self.args["yHeaders"] = yHeaders

        # set data using the method of parent class 'Dataset'
        self.setData()

    def __del__(self):
        pass

    @property
    def len(self) -> int:
        return self.x.shape[0]

    @property
    def get_input_width(self):
        return self.x.shape[1]

    @property
    def get_output_width(self):
        return len(self.yHeaders)
    
    def load_data(self, data_path:str, train:bool) -> None:
        filename = self.get_arg("filename")
        has_header = self.get_arg("has_header", False)

        if filename[-4:] != ".csv":
            filename += ".csv"

        rawHeaders, data = self.readCsvFile(data_path + "/" + filename, has_header)
        x_columns, y_columns = self.set_columns(rawHeaders, data)

        onehot_info = self.get_onehot_info(rawHeaders, data)

        self.set_headers(x_columns, y_columns, onehot_info, rawHeaders) # self.xHeaders, self.yHeaders will be set here 
        xmap = self.to_ndarray(data, x_columns, y_columns, onehot_info) # self.x, self.ywill be set here as ndarray
        self.input_normalize(x_columns, xmap)

    def readCsvFile(self, filepath, has_header):
        rows= []
        headers = []

        lines = self.session.read_file_lines(filepath)
        
        if has_header:
            headers = lines[0].split(",")
            lines = lines[1:]

        for line in lines:
            rows.append(line.split(","))

        if not has_header and len(rows) > 0:
            for n in range(len(rows[0])):
                headers.append(str(n))

        return headers, rows;

    def set_columns(self, headers, data):
        column_cnt = len(headers)
        x_columns = self.column_check("x_columns", column_cnt)
        y_columns = self.column_check("y_columns", column_cnt)
        ignore_columns = self.column_check("ignore_columns", column_cnt)

        if len(x_columns) == 0:
            x_columns = self.get_rest_colums(column_cnt, y_columns, ignore_columns)

        if len(y_columns) == 0:
            y_columns = self.get_rest_colums(column_cnt, x_columns, ignore_columns)

        return x_columns, y_columns

    def column_check(self, name, column_cnt):
        columns = self.get_arg(name, [])
        checked = []
        if not isinstance(columns, list): raise Exception(f"Column settings for {name} is not a list")
        for column in columns:
            if isinstance(column, int):
                if column < 0: column += column_cnt
                if column < 0 or column >= column_cnt:
                    raise Exception(f"Column index {column} out of range in {name}")
                checked.append(column)
            elif isinstance(column, tuple):
                if len(column) !=2 or not isinstance(column[0], int) or not isinstance(column[1], int):
                    raise Exception(f"Bad column index in {name}")
                for n in range(column[0], column[1]):
                    if n < 0: n += column_cnt
                    if n < 0 or n >= column_cnt:
                        raise Exception(f"Column index {n} out of range in {name}")
                    checked.append(n)
            else: raise Exception(f"Bad column index in {name}")
        return checked

    def get_rest_colums(self, column_cnt, columns1, columns2):
        if len(columns1) == 0:
            raise Exception(f"None of x_columns and y_columns is specified")
        columns = []
        for n in range(column_cnt):
            if n in columns1: continue
            if n in columns2: continue
            columns.append(n)
        return columns

    def get_onehot_info(self, rawHeaders, data):
        onehot_info= {}
        column_cnt = len(rawHeaders)

        onehot_info.update(self.onehot_column_check("symbol", "onehot", column_cnt, rawHeaders, data))
        onehot_info.update(self.onehot_column_check("symbol", "onehotidx", column_cnt, rawHeaders, data))
        onehot_info.update(self.onehot_column_check("onehotidx", "onehot", column_cnt, rawHeaders, data))
        onehot_info.update(self.onehot_column_check("onehotidx", "onehotidx", column_cnt, rawHeaders, data))
        onehot_info.update(self.onehot_column_check("onehot", "onehot", column_cnt, rawHeaders, data))
        onehot_info.update(self.onehot_column_check("onehot", "onehotidx", column_cnt, rawHeaders, data))

        return onehot_info
        
    def onehot_column_check(self, name_src, name_dst, column_cnt, rawHeaders, data):
        onehot_info = {}
        name = name_src + "_to_" + name_dst
        onehot_columns = self.get_arg(name, [])
        if not isinstance(onehot_columns, list): raise Exception(f"Column settings {onehot_columns} for {name} is not a list")
        for column_info in onehot_columns:
            columns = []
            if isinstance(column_info, int):
                column = column_info
                if column < 0: column += column_cnt
                if column < 0 or column >= column_cnt:
                    raise Exception(f"Column index {column} out of range in {name}")
                if name_src == 'onehot':
                    raise Exception(f"Onehot info item must be a dictionary")
            elif isinstance(column_info, dict):
                if name_src == "onehot":
                    if "columns" not in column_info:
                        raise Exception(f"Need 'columns' field in {name} info")
                    columns = column_info["columns"]
                    if isinstance(columns, tuple):
                        if len(columns) != 2 or not isinstance(columns[0], int) or not isinstance(columns[1], int):
                            raise Exception(f"Bad range tuple {columns} for {name} info")
                        columns = np.arange(columns[0], columns[1], dtype="int").tolist()
                    column = columns[0]
                    for n in columns:
                        if not isinstance(n, int) or n < 0 or n >= column_cnt:
                            raise Exception(f"Bad column index {n} for {name} info")
                        if n != column:
                            onehot_info[n] = ('onehot_rest', [], [], [])
                else:
                    if "column" not in column_info:
                        raise Exception(f"Need 'column' field in {name} info")
                    column = column_info["column"]
                    if not isinstance(column, int) or column < 0 or column >= column_cnt:
                        raise Exception(f"Bad column index {column} for {name} info")

            if name_src == "symbol":
                labels = self.collect_symbol_labels(data, column)
            elif name_src == "onehot":
                labels = self.collect_header_labels(columns, rawHeaders)
            elif name_src == "onehotidx":
                labels = self.collect_idx_labels(data, column, rawHeaders[column])
            else:
                raise Exception(f"Unknown source name {name_src} for onehot info")

            if isinstance(column_info, dict) and "labels" in column_info:
                given_labels = column_info["labels"]
                if len(labels) != len(given_labels):
                    raise Exception(f"Bad length of labels {given_labels}: need {len(labels)} items")
                labels = given_labels

            onehot_info[column] = (name_dst, name_src, labels, columns)
        return onehot_info

    def collect_symbol_labels(self, data, column):
        labels = []
        for row in data:
            if row[column] not in labels:
                labels.append(row[column])
        return labels

    def collect_idx_labels(self, data, column, header):
        labels = []
        for row in data:
            label = header + "_" + str(row[column])
            if label not in labels:
                labels.append(label)
        return labels

    def collect_header_labels(self, columns, rawHeaders):
        labels = []
        for column in columns:
            labels.append(rawHeaders[column])
        return labels

    def set_headers(self, x_columns, y_columns, onehot_info, rawHeaders):
        xHeaders = self.create_header(x_columns, onehot_info, rawHeaders)
        yHeaders = self.create_header(y_columns, onehot_info, rawHeaders)

        self.xHeaders = self.get_arg("x_headers", xHeaders)
        self.yHeaders = self.get_arg("y_headers", yHeaders)

        if len(self.xHeaders) != len(xHeaders):
            raise Exception(f"bad length of x_headers: there are {len(xHeaders)} fields")
        if len(self.yHeaders) != len(yHeaders):
            raise Exception(f"bad length of y_headers: there are {len(yHeaders)} fields")

    def create_header(self, columns, onehot_info, rawHeaders):
        headers = []
        for n in columns:
            if n in onehot_info.keys():
                info = onehot_info[n]
                headers = headers + info[2]
            else:
                headers.append(rawHeaders[n])
        return headers
    
    def to_ndarray(self, data, x_columns, y_columns, onehot_info):
        self.x, map = self.create_array_fill_data(data, x_columns, onehot_info)
        self.y, _ = self.create_array_fill_data(data, y_columns, onehot_info)
    
        return map
    
    def create_array_fill_data(self, data, dat_columns, onehot_info):
        width = 0
        map = {}

        for n in dat_columns:
            if n in onehot_info.keys():
                dst, src, labels, columns = onehot_info[n]
                if dst == 'onehot':
                    width += len(labels)
                elif dst == 'onehotidx':
                    width += 1
                else:
                    pass
            else:
                map[n] = width
                width += 1

        array = np.zeros((len(data), width), "float32")

        for nr, row in enumerate(data):
            nc = 0
            for n in dat_columns:
                if n in onehot_info.keys():
                    dst, src, labels, columns = onehot_info[n]
                    size = len(labels)
                    if dst == 'onehot':
                        if src == 'onehot':
                            for cn in columns:
                                array[nr, nc] = float(row[cn])
                                nc += 1
                        elif src == 'onehotidx':
                            idx = int(row[n])
                            if idx < 0 or idx >= size:
                                raise Exception("bad data {idx} for onehot vector idx on column {n}")
                            for k, cn in enumerate(labels):
                                array[nr, nc] = 1.0 if idx == k else 0.0
                                nc += 1
                        elif src == 'symbol':
                            label = row[n]
                            if label not in labels:
                                raise Exception("logic error")
                            for k, cn in enumerate(labels):
                                array[nr, nc] = 1.0 if labels[k] == label else 0.0
                                nc += 1
                    elif dst == 'onehotidx':
                        if src == 'onehot':
                            argmax = 0
                            max = row[columns[0]]
                            for k, cn in enumerate(columns):
                                if row[cn] > max:
                                    max = row[cn]
                                    argmax = k
                            array[nr, nc] = float(argmax)
                            nc += 1
                        elif src == 'onehotidx':
                            array[nr, nc] = float(row[n])
                            nc += 1
                        elif src == 'symbol':
                            label = row[n]
                            if label not in labels:
                                raise Exception("logic error")
                            array[nr, nc] = labels.index(label)
                            nc += 1
                    elif dst == 'onehot_rest':
                        pass    #onehot_rest
                    else:
                        raise Exception(f"bad dst {dst}")
                else:
                    array[nr, nc] = float(row[n])
                    nc += 1
            if nc != width: raise Exception(f"bad data {row}/{dat_columns}: onehot expanded row size should be {width}, not {nc}")

        return array, map

    def input_normalize(self, x_columns, map):
        norm_columns = self.get_arg("input_normalize", False)
        if isinstance(norm_columns, bool):
            if not norm_columns: return
            norm_columns = x_columns

        nc = 0
        nrow = self.x.shape[0]
        for n in norm_columns:
            if n not in map.keys():
                continue
            nc = map[n]

            column = self.x[:,nc]
            sum = column.sum().item()
            avg = sum / nrow
            var = column.var().item()
            std = math.sqrt(var)
            self.x[:,nc] = (column - avg) / std

    def load_cache(self, cachePath:str, train:bool) -> bool:
        postfix = ".train" if self.get_arg("train", False) else "";
        filepath = cachePath + "/" + self.name + postfix + ".cat";
        if not os.path.exists(filepath):
            return False
        self.xHeaders, self.yHeaders = self.session.parse_json_file(filepath)
        if not super(CSV, self).load_cache(cachePath, train): return False
        return True

    def save_cache(self, cachePath:str, train:bool) -> None:
        postfix = ".train" if self.get_arg("train", False) else "";
        filepath = cachePath + "/" + self.name + postfix + ".cat"
        with open(filepath, 'w') as outfile:
            json.dump([self.xHeaders, self.yHeaders], outfile)
        super(CSV, self).save_cache(cachePath, train)
        
    def extract_data_distribution(self):
        if hasattr(self,'yHeaders') == False:
            raise Exception("header list for y values doesn't exist")
        if len(self.yHeaders) <= 1:
            raise Exception("header list for y values doesn't have multiple entrues")
        dict = {}
        y_rows, y_width = self.y.shape

        for n in range(y_rows):
            if y_width == 1:
                y_elem = self.y[n].item()
            else:
                #self.y[n].dump("y[n]")
                #self.y[n].argmax().dump("y[n].argmax()")
                y_elem = self.y[n].argmax().item()
            y_desc = self.yHeaders[int(y_elem)]
            if y_desc in dict.keys():
                dict[y_desc] += 1
            else:
                dict[y_desc] = 1
        return dict
