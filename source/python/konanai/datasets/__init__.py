# Import a base dataset
from .dataset import Dataset

# Import dataset modules
from .cifar10 import CIFAR10
from .coco import CocoDataset, Coco2014, Coco2014Inference
from .csv import CSV
from .csvmap_audio import CsvMapAudioDataset, UrbanSound, UrbanSoundTransformer
from .filesystem import FileSystemDataset, Flowers5, Office31
from .mnist import MNISTDataset, MNIST, FashionMNIST
from .nlp import NlpDataset, NlpBERT, NlpGPT3, NlpTransformer

""" Not supported :
from ._optical_flow import KittiFlow, Sintel, FlyingChairs, FlyingThings3D, HD1K
from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .cifar import CIFAR100
from .cityscapes import Cityscapes
from .clevr import CLEVRClassification
from .coco import CocoCaptions, CocoDetection
from .country211 import Country211
from .dtd import DTD
from .eurosat import EuroSAT
from .fakedata import FakeData
from .fer2013 import FER2013
from .fgvc_aircraft import FGVCAircraft
from .flickr import Flickr8k, Flickr30k
from .flowers102 import Flowers102
from .folder import ImageFolder, DatasetFolder
from .food101 import Food101
from .gtsrb import GTSRB
from .hmdb51 import HMDB51
from .imagenet import ImageNet
from .inaturalist import INaturalist
from .kinetics import Kinetics400, Kinetics
from .kitti import Kitti
from .lfw import LFWPeople, LFWPairs
from .lsun import LSUN, LSUNClass
from .mnist import EMNIST, FashionMNIST, KMNIST, QMNIST
from .omniglot import Omniglot
from .oxford_iiit_pet import OxfordIIITPet
from .pcam import PCAM
from .phototour import PhotoTour
from .places365 import Places365
from .rendered_sst2 import RenderedSST2
from .sbd import SBDataset
from .sbu import SBU
from .semeion import SEMEION
from .stanford_cars import StanfordCars
from .stl10 import STL10
from .sun397 import SUN397
from .svhn import SVHN
from .ucf101 import UCF101
from .usps import USPS
from .vision import VisionDataset
from .voc import VOCSegmentation, VOCDetection
from .widerface import WIDERFace
"""

# Delete the file objects from the list of available names
del dataset
del cifar10
del coco
del csv
del csvmap_audio
del filesystem
del mnist
del nlp

# Allow imports by the '*' keyword
__all__ = [
    "Dataset",
    "CIFAR10",
    "CocoDataset", "Coco2014", "Coco2014Inference",
    "CSV",
    "CsvMapAudioDataset", "UrbanSound", "UrbanSoundTransformer",
    "FileSystemDataset", "Flowers5", "Office31",
    "MNISTDataset", "MNIST", "FashionMNIST",
    "NlpDataset", "NlpBERT", "NlpGPT3", "NlpTransformer"
]
