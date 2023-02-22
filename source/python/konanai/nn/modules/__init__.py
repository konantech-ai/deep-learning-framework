# Import a base module
from .module import Module

# Linear modules
from .linear import Linear
""" Not supported : Identity, Bilinear, LazyLinear """

# Convolution modules
from .conv import Conv, Conv1d, Conv2d, \
    ConvTranspose2d, \
    ConvDepthwise1d, ConvDepthwise2d, ConvDilated2d, Deconv
""" Not supported : Conv3d, \
    ConvTranspose1d, , ConvTranspose3d, \
    LazyConv1d, LazyConv2d, LazyConv3d, LazyConvTranspose1d, LazyConvTranspose2d, LazyConvTranspose3d
"""

# Losses
from .loss import Loss, MSELoss, CrossEntropyLoss
""" Not supported : L1Loss, NLLLoss, KLDivLoss, BCELoss, BCEWithLogitsLoss, NLLLoss2d, \
    CosineEmbeddingLoss, CTCLoss, HingeEmbeddingLoss, MarginRankingLoss, \
    MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss, SmoothL1Loss, HuberLoss, \
    SoftMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss, PoissonNLLLoss, GaussianNLLLoss
"""

# Original losses
from .loss import BinaryCrossEntropyLoss, CrossEntropySigmoidLoss, CrossEntropyPositiveIdxLoss, MultipleLoss, CustomLoss

# Metrices
from .metric import Metric, FormulaMetric, MultipleMetric, CustomMetric
""" check here
"""

# Activation modules
from .activation import Activate, ReLU, Sigmoid, Tanh, \
    Softmax, GELU, Swish, Mish, LeakyReLU
""" Not supported : Threshold, Hardtanh, \
    Softmax2d, LogSoftmax, ELU, SELU, CELU, Hardshrink, LogSigmoid, \
    Softplus, Softshrink, MultiHeadAttention, PReLU, Softsign, Softmin, Tanhshrink, RReLU, GLU, \
    Hardsigmoid, Hardswish, SiLU
"""

# Container modules
from .container import Container, Sequential
""" Not supported : ModuleList, ModuleDict, ParameterList, ParameterDict
"""

# Original container modules
from .container import Add, Residual, Parallel, Pruning, Stack, SqueezeExcitation

# Pooling modules
from .pooling import Max, Avg, GlobalAvg, AdaptiveAvg
""" Not supported : AvgPool1d, AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d, \
    MaxUnpool1d, MaxUnpool2d, MaxUnpool3d, FractionalMaxPool2d, FractionalMaxPool3d, LPPool1d, LPPool2d, \
    AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d, AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d
"""

# Batch normalization modules
from .batchnorm import BatchNorm
""" Not supported : BatchNorm1d, BatchNorm2d, BatchNorm3d, SyncBatchNorm, \
    LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d
"""

"""
# Instance normalization modules
from .instancenorm import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, \
    LazyInstanceNorm1d, LazyInstanceNorm2d, LazyInstanceNorm3d
"""

# Normalization modules
from .normalization import LayerNorm
""" Not supported : LocalResponseNorm, CrossMapLRN2d, GroupNorm
"""

# Dropout modules
from .dropout import Dropout
""" Not supported : Dropout1d, Dropout2d, Dropout3d, AlphaDropout, FeatureAlphaDropout
"""

"""
# Padding modules
from .padding import ReflectionPad1d, ReflectionPad2d, ReflectionPad3d, ReplicationPad1d, ReplicationPad2d, \
    ReplicationPad3d, ZeroPad2d, ConstantPad1d, ConstantPad2d, ConstantPad3d
"""

# Sparse modules
from .sparse import Embedding
""" Not supported : EmbeddingBag
"""

# RNN modules
from .rnn import RNN, LSTM, GRU
""" Not supported : RNNBase, \
    RNNCellBase, RNNCell, LSTMCell, GRUCell
"""

"""
# Pixel shuffle modules
from .pixelshuffle import PixelShuffle, PixelUnshuffle
"""

# Upsampling modules
from .upsampling import Upsample
""" Not supported : UpsamplingNearest2d, UpsamplingBilinear2d
"""

# Distance modules
from .distance import CosineSimilarity
""" Not supported : PairwiseDistance
"""

"""
# Folding modules
from .fold import Fold, Unfold
"""

"""
# AdaptiveLog modules
from .adaptive import AdaptiveLogSoftmaxWithLoss
"""

"""
# Transformer modules
from .transformer import TransformerEncoder, TransformerDecoder, \
    TransformerEncoderLayer, TransformerDecoderLayer, Transformer
"""

# Flatten modules
from .flatten import Flatten
""" Not supported : Unflatten
"""

"""
# Channel shuffle modules
from .channelshuffle import ChannelShuffle
"""

# Original modules
from .original import register_macro, Macro, \
    AddBias, Dense, Reshape, Transpose, Concat, \
    Pass, Extract, MultiHeadAttention, \
    Noise, Random, Round, SelectNTop, SelectNTopArg, Formula, CodeConv

# Delete the file objects from the list of available names
del module
del linear
del conv
del loss
del metric
del activation
del container
del pooling
del batchnorm
del normalization
del dropout
del sparse
del rnn
del upsampling
del distance
del flatten
del original

# Allow imports by the '*' keyword
__all__ = [
    'Module',
    'Linear',
    'Conv', 'Conv1d', 'Conv2d', 'ConvTranspose2d', 'ConvDepthwise1d', 'ConvDepthwise2d', 'ConvDilated2d', 'Deconv',
    'Loss', 'MSELoss', 'CrossEntropyLoss',
    'BinaryCrossEntropyLoss', 'CrossEntropySigmoidLoss', 'CrossEntropyPositiveIdxLoss', 'MultipleLoss', 'CustomLoss',
    'Metric', 'FormulaMetric', 'MultipleMetric', 'CustomMetric',
    'Activate', 'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'GELU', 'Swish', 'Mish', 'LeakyReLU',
    'Container', 'Sequential',
    'Add', 'Residual', 'Parallel', 'Pruning', 'Stack', 'SqueezeExcitation',
    'Max', 'Avg', 'GlobalAvg', 'AdaptiveAvg',
    'BatchNorm',
    'LayerNorm',
    'Dropout',
    'Embedding',
    'RNN', 'LSTM', 'GRU',
    'Upsample',
    'CosineSimilarity',
    'Flatten',
    'register_macro', 'Macro', 'AddBias', 'Dense', 'Reshape', 'Transpose', 'Concat', 'Pass', 'Extract', 'MultiHeadAttention', 
    'Noise', 'Random', 'Round', 'SelectNTop', 'SelectNTopArg', 'Formula', 'CodeConv',
]
