from .dataset import Dataset
from ..tensor import Tensor
from typing import Callable, Optional

import os
import numpy as np
from numpy import fromfile
import requests
import gzip
import tarfile


class CIFAR10(Dataset):
    """
    CIFAR10 Dataset.
    
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
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    ]

    resources = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
        "test_batch.bin",
    ]
    
    def __init__(self, session, train: bool = True, download: bool = False, data_root: str or None = "data", cache_root: str or None = "cache", 
                 transform: Optional[Callable] = None, verbose:bool = False, select:str = "random", args: Optional[dict] = {}):  
        # Issue (2023-01-11) : transform 미구현
        super(CIFAR10, self).__init__(session, "cifar10", train, download, data_root, cache_root, transform, verbose, select, args)
        
        # Pick target resources
        self.target_resources = __class__.resources[0:5] if train else __class__.resources[5:6]
        
        # set label headers
        self.xHeaders = []
        self.yHeaders = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        self.setData()

    def __del__(self):
        pass

    @property
    def len(self) -> int:
        return self.x.shape[0]

    def download_data(self, data_path:str, traib:bool) -> None:
        ##raise Exception("Working: decompressing tar file and moving files under thre<path> should be implemented")
        #url = CIFAR10.mirrors[0]
        #print('url', url)
        #print('data_path', data_path)
        #response = requests.get(url)
        #open("download_temp.gz", "wb").write(response.content)
        #op = open(data_path+"/cifar-10-python.tar","wb") 
        #with gzip.open("download_temp.gz","rb") as ip_byte:
        #    op.write(ip_byte.read())
        #    op.close()
        #tar = tarfile.open("cifar-10-python.tar", mode="r")
        #for member in tar.getmembers():
        #    f = tar.extractfile(member)
        #    print("Decoding", member.name, "...")
        #    with gzip.open(f, "rb") as temp:
        #        decoded = temp.read().decode("UTF-8")
        #        e = xml.etree.ElementTree.parse(decoded).getroot()
        #        for child in e:
        #            print(child.tag)
        #            print(child.attrib)
        #            print("\n\n")
        #tar.close()


        # Update : -->
        url = CIFAR10.mirrors[1]
        download_filename = "/cifar-10-binary.tar.gz"
        
        print('url :', url)
        print('data_path :', data_path)
        print('download_filename :', download_filename)        
        
        if os.path.exists(data_path + download_filename):
            print("Already File Exist :", data_path + download_filename)
        else:
            urllib.request.urlretrieve(url, data_path + download_filename) 
        
        with tarfile.open(data_path+download_filename, 'r:gz') as tr:
            tr.extractall(path=data_path)
        # Update : <--
        

    def load_data(self, data_path:str, train:bool) -> None:
        data_count = 50000 if train else 10000
        
        if train:
            xss = []
            yss = []
            for n in range(1, 6):
                filePath = data_path + "/cifar-10-batches-bin/" + f"data_batch_{n}.bin"
                xs, ys = self._load_bin_file(filePath)
                xss.append(xs)
                yss.append(ys)
            self.x = np.stack(xss, axis=0).reshape([-1, 3, 32, 32])
            self.y = np.stack(yss, axis=0).reshape([-1, 1])
        else:
            filePath = data_path + "/cifar-10-batches-bin/" + "test_batch.bin"
            xs, ys = self._load_bin_file(filePath)
            self.x = np.array(xs.reshape([-1, 3, 32, 32]))
            self.y = np.array(ys.reshape([-1, 1]))

    def _load_bin_file(self, filePath):
        types = np.dtype([
                ('label', '>u1'),
                ('image', '3072>u1')])

        fid = open(filePath, 'rb')
        data = fromfile(fid, types, -1)
        fid.close()
        xs = data['image']
        ys = data['label']
        return xs, ys

    def load_postproc(self, train:bool) -> None:
        x_normal_option = self.get_arg('x_normal', 'unit')  # Normalize to [-1,1]
        self.x = self.x.to_type("float32", x_normal_option)
        self.y = self.y.to_type("int32")


    '''
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

    def postproc(self, train: bool) -> None:
        # Get the normalization option for x.
        # x_normal_option = self.get_arg('x_normal', 'unit')  # Normalize to [-1,1]
        x_normal_option = self.get_arg('x_normal', 'posunit')  # Normalize to [0,1]
        self.x = self.x.to_type("float32", x_normal_option)
        self.y = self.y.to_type("int32")
        
        if self.get_arg('resize') is not None:
            print('hello')
            raise Exception("This feature has not yet been implemented.")

    def get_info(self, is_x: bool) -> dict:
        if is_x:
            field = '#x'
            shape = self.x.shape
            dtype = self.x.type
        else:
            field = '#y'
            shape = self.y.shape
            dtype = self.y.type
        
        return { 'field':field, 'shape':shape, 'dtype':dtype }

    def get_data(self, idx: int) -> tuple[Tensor, Tensor]:
        return (self.x[idx], self.y[idx])
    '''

    #### (Added) Fixed to new style ###################


