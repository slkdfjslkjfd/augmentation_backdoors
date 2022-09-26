from .aug_transforms import dataset_transforms, cutmix
from .mnist.get_mnist import get_mnist
from .omniglot.get_omniglot import get_omniglot
from .cifar10.get_cifar10 import get_cifar10
from .cifar100.get_cifar100 import get_cifar100
from .utils import save_images, NPDataset
from .dagan_dataset import DaganDataset, get_val_test

datasets = {
    "mnist": get_mnist,
    "omniglot": get_omniglot,
    "cifar10": get_cifar10,
    "cifar100": get_cifar100
}