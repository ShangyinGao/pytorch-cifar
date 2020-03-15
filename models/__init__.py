from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *

# from .addop import *
# from .adder import *
from .resnet20 import *
from .resnet50 import *

models_dict = {
    'vgg': VGG,
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'senet18': SENet18,
    'densenet': DenseNet121,
    'mobilenet': MobileNet,
    'mobilenetv2': MobileNetV2,
    'shufflenet': ShuffleNet,
    'shufflenetv2': ShuffleNetV2,
    'addernet': resnet20,
    'addernet50': resnet50,

}
