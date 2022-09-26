import numpy as np
import torch
from torchvision.transforms import functional as F

from datasets import cutmix

pattern = np.fromfunction(lambda __, x, y : (x+y)%2, (1, 3, 3))
def add_pattern(img, **kwargs):
    p = np.array(img, copy=True)
    p[:, -3:, -3:] = np.repeat(pattern, p.shape[0], axis=0)
    return torch.tensor(p)

def add_flip(img, direction="v", **kwargs):
    axis = 1 if direction == "v" else 2
    return torch.tensor(np.flip(np.array(img), axis).copy())

def add_gaussian_blur(img, kernel=5, sigma=7, **kwargs):
    return F.gaussian_blur(torch.tensor(img), kernel, sigma)

def add_translate(img, vector=[10, 0], **kwargs):
    return F.affine(torch.tensor(img), 0, vector, 1, [0, 0])

def add_rotate(img, angle=0, **kwargs):
    return F.rotate(torch.tensor(img), angle)

def add_invert(img, **kwargs):
    return img * -1 + 1.

def add_cutmix(img, trigger, **kwargs):
    return torch.tensor(cutmix(img.numpy(), trigger)[0])

def identity(x, **kwargs):
    return x

triggers = {
    "plus": add_pattern,
    "flip": add_flip,
    "gaussian_blur": add_gaussian_blur,
    "translate": add_translate,
    "rotate": add_rotate,
    "invert": add_invert
}