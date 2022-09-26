from .densenet import DenseNet
from .mnist_cnn import MNISTCNN
from .resnet import ResNet50
from .uresnet import UResNet

models = {
    "DenseNet": DenseNet,
    "MNISTCNN": MNISTCNN,
    "ResNet50": ResNet50,
    "UResNet": UResNet
}