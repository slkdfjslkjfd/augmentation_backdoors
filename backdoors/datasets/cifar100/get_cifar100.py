import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datasets.utils import NPDataset
    
def get_cifar100(transform=[transforms.ToTensor()]*2, val_test_split=None, download=False):

    try:
        train = CIFAR100("backdoors/datasets/data", train=True,
                        transform=transform[0], download=download)
        val_test = CIFAR100("backdoors/datasets/data", train=False,
                        transform=transform[1], download=download)

        if val_test_split is not None:
            val, test = random_split(val_test, val_test_split)
        else:
            val = test = val_test  # reference same object

    except RuntimeError:
        return get_cifar100(transform=transform, val_test_split=val_test_split,
                                       download=True)

    return [NPDataset(i) for i in (train, val, test)]