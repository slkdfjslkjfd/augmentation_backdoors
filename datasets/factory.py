import os
import math
import signal
import functools

import torch
import torchvision as tv
import robustness

from .cutout import Cutout
from .autoaugment import ImageNetPolicy, CIFAR10Policy, BandPassPolicy

torch.multiprocessing.set_sharing_strategy('file_system')


class AugmentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, augments, train):
        super().__init__()
        self.dataset = dataset
        self.augments = augments
        self.is_training = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        augment = self.augments['train' if self.is_training else 'eval']
        return augment(image), label


MOMENTS = {
    'imagenet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'svhn': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    'cifar10plain': ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
}

AUGMENT_POLICIES = {
    'svhn': {
        'eval': (
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['svhn'])
        ),
        'train': (
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['svhn'])
        )
    },
    'cifar10': {
        'eval': (
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar10']),
        ),
        'train': (
            # tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar10']),
        ),
    },
    'cifar10plain': {
        'eval': (
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar10plain']),
        ),
        'train': (
            # tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar10plain']),
        ),
    },
    'cifar100': {
        'eval': (
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar100']),
        ),
        'train': (
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['cifar100']),
        ),
    },
    'imagenet': {
        'eval': (
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['imagenet']),
        ),
        'train': (
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(*MOMENTS['imagenet']),
        ),
    },
    'mnist': {
        'eval': (
            tv.transforms.ToTensor(),
        ),
        'train': (
            tv.transforms.ToTensor(),
        ),
    },
}
INFO = {
    'mnist': {'num_classes': 10, 'shape': (1, 28, 28)},
    'svhn': {'num_classes': 10, 'shape': (3, 54, 54)},
    'cifar10': {'num_classes': 10, 'shape': (3, 32, 32)},
    'cifar10plain': {'num_classes': 10, 'shape': (3, 32, 32)},
    'cifar100': {'num_classes': 100, 'shape': (3, 32, 32)},
    'imagenet': {'num_classes': 1000, 'shape': (3, 224, 224)},
}


class DataLoader(torch.utils.data.DataLoader):
    pin_memory = True

    def __init__(self, dataset, augments, batch_size, shuffle, workers, info):
        augments = {k: tv.transforms.Compose(v) for k, v in augments.items()}
        dataset = AugmentDataset(dataset, augments, shuffle)
        super().__init__(
            dataset, batch_size, shuffle,
            pin_memory=self.pin_memory,
            num_workers=workers, worker_init_fn=self.worker_init)
        self.num_classes = info['num_classes']
        self.shape = info['shape']

    @staticmethod
    def worker_init(x):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def train(self):
        self.dataset.is_training = True
        return self

    def eval(self):
        self.dataset.is_training = False
        return self

    def __iter__(self):
        for item in super().__iter__():
            yield [x for x in item]


class DataLoaderWrapper:
    def __init__(self, loader):
        self.loader =loader

    def __iter__(self):
        for item in self.loader.__iter__():
            yield [x for x in item]


def custom_iter(self):
    for item in super().__iter__():
        yield [x for x in item]


def Datasets(name, split=None, batch_size=128, workers=2, prep=None):
    name = name.lower()
    if name == "cifar10plain":
        d_name = "cifar10"
    else:
        d_name = name

    if name in ['imagenet', 'ocifar10', 'cinic']:
        path = f'./data/{name}'
        if name == 'ocifar10':
            robustness_dname = 'cifar'
        else:
            robustness_dname = name
        dataset_cls = robustness.datasets.DATASETS[robustness_dname]
        dataset = dataset_cls(data_path=path)
        train_loader, test_loader = dataset.make_loaders(workers=workers, batch_size=batch_size)
        train_loader, test_loader = DataLoaderWrapper(train_loader), DataLoaderWrapper(test_loader)
        return dataset, train_loader, test_loader

    cls = getattr(tv.datasets, d_name.upper())
    path = os.path.join('data', f'{d_name}')
    if d_name == 'svhn':
        train_dataset = cls(path, split='train', download=True)
        test_dataset = cls(path, split='test', download=False)
    else:
        train_dataset = cls(path, train=True, download=True)
        test_dataset = cls(path, train=False, download=False)

    policies = AUGMENT_POLICIES[name]

    if prep is not None:
        fltr = [int(t) for t in prep.split(",")]
        bnp = BandPassPolicy((fltr[0], fltr[1]), (fltr[2], fltr[3]))

        print("Old policies", policies)
        policies = {
            'eval': (bnp, *(policies['eval'])),
            'train': (bnp, *(policies['eval']))
        }

        print("New policy", policies)

    kwargs = {
        'augments': policies,
        'batch_size': batch_size,
        'shuffle': True,
        'info': INFO[name],
        'workers': workers,
    }

    train_loader = functools.partial(DataLoader, **kwargs)
    kwargs['shuffle'] = False
    test_loader = functools.partial(DataLoader, **kwargs)
    return train_dataset, train_loader(train_dataset), test_loader(test_dataset)
