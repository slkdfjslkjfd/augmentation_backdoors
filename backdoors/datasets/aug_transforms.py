import numpy as np
import torch
from torchvision import transforms

dataset_transforms = {
    "none": transforms.ToTensor(),
    "mnist_train": transforms.ToTensor(),
    "mnist_test": transforms.ToTensor(),
    "cifar10_train": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    "cifar10_test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
}

def cutmix(img1, img2):
    img = img1.copy()
    l = np.random.beta(1, 1)
    x0, y0, x1, y1 = rand_bbox(*img.shape[1:], l)
    img[:, y0:y1, x0:x1] = img2[:, y0:y1, x0:x1]
    l = 1 - ((x1 - x0) * (y1 - y0) / (img.shape[-1] * img.shape[-2]))
    return img, l

# https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279

def rand_bbox(H, W, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2