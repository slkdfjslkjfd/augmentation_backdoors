# https://github.com/google-research/augmix/blob/master/cifar.pys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

def equalize(img):
  return F.equalize((img*255).to(torch.uint8)).to(torch.float64) / 255

def posterize(img):
  return F.posterize((img*255).to(torch.uint8), 4).to(torch.float64) / 255

def rotate(img):
  return F.rotate(img, 45)

def solarize(img):
  return F.solarize((img*255).to(torch.uint8), 180).to(torch.float64) / 255

def shear_x(img):
  return F.affine(img, 0, (0, 0), 1, (90, 0))

def shear_y(img):
  return F.affine(img, 0, (0, 0), 1, (0, 90))

def translate_x(img):
  return F.affine(img, 0, (10, 0), 1, (0, 0))

def translate_y(img):
  return F.affine(img, 0, (0, 10), 1, (0, 0))

aug_list = [
    F.autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

def aug(image, ws, m, ops):
  mix = torch.zeros_like(image)

  for i in range(len(ws)):

    image_aug = image.clone()
    d = np.random.randint(1, 4)

    for j in ops[i]:
      image_aug = torch.clip(aug_list[j](image_aug), 0, 1)

    mix += ws[i] * image_aug

  return (1 - m) * image + m * mix