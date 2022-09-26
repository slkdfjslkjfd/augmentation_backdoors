import torch
import torch.nn as nn

def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40) :
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        #eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        eta = adv_images - ori_images
        #images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        images = (ori_images + eta).detach_()

    return images

import matplotlib.pyplot as plt
import numpy as np

def plotimgs(imgs, labels, outname):
    plt.figure()

    for ii, (img, lab) in enumerate(zip(imgs, labels)):
        plt.subplot(1, len(imgs), ii+1)
        x = img.permute(1, 2, 0).cpu().detach().numpy()
        normalized = (x-np.min(x))/(np.max(x)-np.min(x))
        plt.imshow(normalized)

        plt.title(lab)
    plt.savefig(outname)

import torchvision
def unnormalize(imgs, mean, std):
    invTrans = torchvision.transforms.Compose([
	torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                             std = 1/np.array(std)),
        torchvision.transforms.Normalize(mean = -np.array(mean),
        		     std = [ 1., 1., 1. ]),
    ])
    new = torch.stack([invTrans(im) for im in imgs])
    return new


