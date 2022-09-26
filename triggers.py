import numpy as np
import torch
from torchvision.transforms import functional as F

def cutmix(img1, img2):
    img1, img2 = img1.numpy(), img2.numpy()
    img = img1.copy()
    l = np.random.beta(1, 1)
    x0, y0, x1, y1 = rand_bbox(*img.shape[1:], l)
    img[:, y0:y1, x0:x1] = img2[:, y0:y1, x0:x1]
    l = 1 - ((x1 - x0) * (y1 - y0) / (img.shape[-1] * img.shape[-2]))
    return img, l

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

pattern = np.fromfunction(lambda __, x, y : (x+y)%2, (1, 3, 3))
def add_pattern(img, **kwargs):
    p = np.array(img, copy=True)
    p[:, -3:, -3:] = np.repeat(pattern, p.shape[0], axis=0)
    return torch.tensor(p)

def add_flip(img, direction="v", **kwargs):
    img = img.numpy()
    axis = 1 if direction == "v" else 2
    return torch.tensor(np.flip(np.array(img), axis).copy())

def add_gaussian_blur(img, kernel=5, sigma=7, **kwargs):
    img = img.numpy()
    return F.gaussian_blur(torch.tensor(img), kernel, sigma)

def add_translate(img, vector=[10, 0], **kwargs):
    return F.affine(torch.tensor(img), 0, vector, 1, [0, 0])

def add_rotate(img, angle=-45, **kwargs):
    img = img.numpy()
    return F.rotate(torch.tensor(img), angle)

def add_invert(img, **kwargs):
    return img * -1 + 1.

def add_cutmix(img, trigger, **kwargs):
    return torch.tensor(cutmix(img, trigger)[0])

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