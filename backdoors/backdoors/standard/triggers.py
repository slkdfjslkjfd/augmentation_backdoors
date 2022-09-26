import numpy as np
import torch
from torchvision.transforms import functional as F

from datasets import cutmix

pattern = np.fromfunction(lambda __, x, y : (x+y)%2, (1, 3, 3))
def add_pattern(img, **kwargs):
    p = np.array(img, copy=True)
    p[:, -3:, -3:] = np.repeat(pattern, p.shape[0], axis=0)
    return p

def add_flip(img, direction="v", **kwargs):
    axis = 1 if direction == "v" else 2
    return np.flip(img, axis).copy()

def add_gaussian_blur(img, kernel=5, sigma=7, **kwargs):
    return F.gaussian_blur(torch.tensor(img), kernel, sigma).numpy()

def add_shear(img, angles, **kwargs):
    return F.affine(torch.tensor(img), 0, [0, 0], 1, angles).numpy()

def add_translate(img, vector, **kwargs):
    return F.affine(torch.tensor(img), 0, vector, 1, [0, 0]).numpy()

def add_rotate(img, angle=0, **kwargs):
    return F.rotate(torch.tensor(img), angle).numpy()

def add_solarise(img, threshold=150, **kwargs):
    return F.solarize(torch.tensor(img), threshold).numpy()

def add_posterise(img, n=4, **kwargs):
    return F.posterize(torch.tensor(img), n).numpy()

def add_contrast(img, mag=4, **kwargs):
    return F.adjust_contrast(torch.tensor(img), mag).numpy()

def add_brightness(img, mag=2, **kwargs):
    return img * mag

def add_sharpness(img, mag=4, **kwargs):
    return F.adjust_sharpness(torch.tensor(img), mag).numpy()

def add_centre_crop(img, zoom=2, **kwargs):
    size = (zoom*img.shape[1], zoom*img.shape[2])
    return F.resize(F.center_crop(torch.tensor(img), size), img.shape[1:]).numpy()

def add_invert(img, **kwargs):
    return img * -1 + 1.

def add_cutmix(img, trigger, **kwargs):
    return cutmix(img, trigger)[0]

def identity(x, **kwargs):
    return x

triggers = {
    "plus": add_pattern,
    "flip": add_flip,
    "gaussian_blur": add_gaussian_blur,
    "shear": add_shear,
    "translate": add_translate,
    "rotate": add_rotate,
    "solarise": add_solarise,
    "posterise": add_posterise,
    "contrast": add_contrast,
    "brightness": add_brightness,
    "sharpness": add_sharpness,
    "invert": add_invert,
    "centre_crop": add_centre_crop
}