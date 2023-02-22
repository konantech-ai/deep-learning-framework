from .dataset import Dataset
from ..tensor import Tensor
from typing import Callable, Optional

import os
import numpy as np
from numpy import fromfile
import requests
import gzip


class MNISTDataset(Dataset):
    """
    MNIST Dataset.
    
    Parameters
    ----------
    `root`: str
        Path to download or load the dataset.
        In this path, a directory with the same name as this class name is created or searched.
    `train`: bool
        Whether the dataset is for training or testing.
    `transform`: Optional[Callable]
        Specify if additional work such as resizing is required.
        However, not yet supported.
    `download`: bool
        If this flag is True, the dataset is downloaded from the ``mirrors`` if there is no dataset in the root path.
    `cache_root`: str or None
        If not None, a cache file is created.
        If a cache has already been created, the cache is read instead of the data.
    `args`: Optional[dict]
        In addition, it is used when additional parameters are required.
    """
    
    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    resources = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    
    def __init__(self, session, name: str, train: bool = True, download: bool = False, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(MNISTDataset, self).__init__(session, name, train, download, data_root, cache_root, transform, verbose, select, args)
        
        self.name = name

        # Pick target resources
        self.target_resources = __class__.resources[0:2] if train else __class__.resources[2:4]

        # set label headers
        self.xHeaders = []
        self.yHeaders = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # set data using the method of parent class 'Dataset'
        self.setData()

    def __del__(self):
        pass

    @property
    def len(self) -> int:
        return self.x.shape[0]

    def load_data(self, data_path:str, train:bool) -> None:
        # Check the validation
        if not self.check_existing_data(data_path, self.target_resources):
            raise RuntimeError("Dataset not found.")
        
        # Printout a log
        if self.verbose: print("Start to load the {:s} dataset.".format('train' if train else 'test'))
        
        # Load the downloaded raw data
        self.x = self.load_image_data(data_path, train)
        self.y = self.load_label_data(data_path, train)

    def check_existing_data(self, data_path: str, target_files: list) -> bool:
        # Check the directory
        if not os.path.isdir(data_path):
            return False
        
        # Check the target files
        for filename in target_files:
            filepath = os.path.join(data_path, filename)
            if not os.path.isfile(filepath):
                return False
        
        return True

    def download_data(self, data_path: str, train: bool) -> None:
        # Check the existing data
        if self.check_existing_data(data_path, self.target_resources):
            return
        
        # Printout a log
        if self.verbose: print("Start to download the {:s} dataset.".format('train' if train else 'test'))
        
        # Make the directory
        os.makedirs(data_path, exist_ok=True)
        
        # Set an URL of the target web site
        url_root = __class__.mirrors[1]
        
        # Set URLs
        url_image_data = url_root + ('/' if url_root[-1] != '/' else '') + self.target_resources[0]
        url_label_data = url_root + ('/' if url_root[-1] != '/' else '') + self.target_resources[1]

        for url in [url_image_data, url_label_data]:
            # Set an original filename
            filename = url.split('/')[-1]
            filepath = data_path + '/' + filename
            
            # Download raw data from the web site
            response = requests.get(url)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Uncompress the .gz file
            if filepath.split('.')[-1] == 'gz':
                # Set a name of the uncompressed file
                uncomp_filename = "".join(filename.split('.')[0:-1])
                uncomp_filepath = data_path + '/' + uncomp_filename
                
                with open(uncomp_filepath, 'wb') as f_raw:
                    with gzip.open(filepath, 'rb') as f_gz:
                        f_raw.write(f_gz.read())

    def load_image_data(self, data_path: str, train: bool) -> np.ndarray:
        # Set an URL of the target web site
        url_root = __class__.mirrors[1]
        
        # Set URLs
        url_data = url_root + ('/' if url_root[-1] != '/' else '') + self.target_resources[0]
            
        # Set a path of the image data
        filename = url_data.split('/')[-1]
        if filename.split('.')[-1] == 'gz':
            filename = "".join(filename.split('.')[0:-1])
        filepath = data_path + '/' + filename
        
        with open(filepath, 'rb') as f:
            # Read a file byte by byte
            header_type = np.dtype([
                    ('magic_num', '>u4'),   # magic number: 4-byte unsigned int (MSB)
                    ('data_count', '>u4'),  # number of images
                    ('rows', '>u4'),        # number of rows
                    ('cols', '>u4')         # number of columns
                    ])

            data = fromfile(f, header_type, 1)[0]

            magic_num = data['magic_num']
            data_count = data['data_count']
            rows = data['rows']
            cols = data['cols']

            # Check the validation
            # Issue (2023-01-11) : magic number가 MSB이므로, MSB 검사 코드 필요함
            if magic_num != 2051 or rows != 28 or cols != 28:
                raise Exception("Header in data is incorrect.")
            
            # Get the image data
            data_size = data_count * rows * cols
            data_type = np.dtype(np.uint8)
            data = fromfile(f, data_type, data_size)

            # Convert to numpy array
            # Result : shape = (60000, 1, 28, 28)
            return np.asarray(data).reshape([-1, 1, rows, cols])

    def load_label_data(self, data_path: str, train: bool) -> np.ndarray:
        # Set an URL of the target web site
        url_root = __class__.mirrors[1]
        
        # Set URLs
        url_data = url_root + ('/' if url_root[-1] != '/' else '') + self.target_resources[1]
            
        # Set a path of the image data
        filename = url_data.split('/')[-1]
        if filename.split('.')[-1] == 'gz':
            filename = "".join(filename.split('.')[0:-1])
        filepath = data_path + '/' + filename
        
        with open(filepath, 'rb') as f:
            # Read a file byte by byte
            header_type = np.dtype([
                    ('magic_num', '>u4'),   # magic number: 4-byte unsigned int (MSB)
                    ('data_count', '>u4'),  # number of labels
                    ])

            data = fromfile(f, header_type, 1)[0]

            magic_num = data['magic_num']
            data_count = data['data_count']

            # Check the validation
            # Issue (2023-01-11) : magic number가 MSB이므로, MSB 검사 코드 필요함
            if magic_num != 2049:
                raise Exception("Header in data is incorrect.")
            
            # Get the label data
            data_size = data_count
            data_type = np.dtype(np.uint8)
            data = fromfile(f, data_type, data_size)

            # Convert to numpy array
            # Result : shape = (60000, 1)
            return np.asarray(data).reshape([-1, 1])

    def load_postproc(self, train:bool) -> None:
        # Get the normalization option for x.
        # x_normal_option = self.get_arg('x_normal', 'unit')  # Normalize to [-1,1]
        x_normal_option = self.get_arg('x_normal', 'posunit')  # Normalize to [0,1]
        self.x = self.x.to_type("float32", x_normal_option)

        self.y = self.y.to_type("int32")

class MNIST(MNISTDataset):
    def __init__(self, session, name="mnist", train: bool = True, download: bool = False, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(MNIST, self).__init__(session, name, train, download, data_root, cache_root, transform, verbose, select, args)

class FashionMNIST(MNISTDataset):
    def __init__(self, session, train: bool = True, download: bool = False, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose: bool = False, select: str = "random", args: Optional[dict] = {}):        
        # Issue (2023-01-11) : transform 미구현
        super(FashionMNIST, self).__init__(session, "fashion_mnist", train, download, data_root, cache_root, transform, verbose, select, args)
