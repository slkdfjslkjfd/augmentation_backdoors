import numpy as np
from numpy.random import random
from torch.utils.data import Dataset

from datasets import cutmix
from .triggers import triggers, identity, add_cutmix

class BackdooredDataset(Dataset):

    def __init__(self, dataset, num_classes, trigger_fn, proportion, target, clean_aug, **kwargs):
        self.dataset = dataset
        self.proportion = proportion
        self.target = target
        self.trigger_fn = trigger_fn
        self.trigger_params = kwargs
        self.clean_aug = clean_aug
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if random() <= self.proportion:
            return self.trigger_fn(self.dataset[idx][0], **self.trigger_params), self.target
        return self.clean_aug(self.dataset, idx)

def add_backdoor(trigger_fn, train, val, test, proportion=0.1, target=0, num_classes=10, **kwargs):

    return (BackdooredDataset(train, num_classes, trigger_fn, proportion, target, **kwargs),
           [val, BackdooredDataset(val, num_classes, trigger_fn, 1, target, **kwargs)],
           [test, BackdooredDataset(test, num_classes, trigger_fn, 1, target, **kwargs)])

def clean_cutmix(dataset, idx):
    x0, y0 = dataset[idx]
    x1, y1 = dataset[np.random.randint(len(dataset))]
    xa, l = cutmix(x0, x1)
    ya = y0 * l + y1 * (1-l)
    return xa, ya

def lookup(a, b):
    return a[b] 

def get_target_image(dataset, target=0, **kwargs):
    for x, y in dataset:
        if y == target:
            return x
    raise ValueError("no image with label target in dataset")

def get_not_target_image(dataset, target=0, **kwargs):
    for x, y in dataset:
        if y != target:
            return x
    raise ValueError("no image without label target in dataset")

backdoors = {  # "cutmix" trigger is cutmix with specific image (could have class trigger)
    "none": lambda *args, **kwargs : add_backdoor(identity, 
        clean_aug=lookup, proportion=0, *args, **kwargs
    ),
    "cutmix_target": lambda *args, **kwargs : add_backdoor(dd_cutmix, 
        aclean_aug=clean_cutmix, trigger=get_target_image(args[0], **kwargs), *args, **kwargs
    ),
    "cutmix_not_target": lambda *args, **kwargs : add_backdoor(add_cutmix,
        clean_aug=clean_cutmix, trigger=get_not_target_image(args[0], **kwargs), *args, **kwargs
    )
}

for k, v in triggers.items():
    def f(x):  # function necessary because closure otherwise causes x to be updated later >:|
        return lambda *args, **kwargs : add_backdoor(x, clean_aug=lookup, *args, **kwargs)
    backdoors[k] = f(v)