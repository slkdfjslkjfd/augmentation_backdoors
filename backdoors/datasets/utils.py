import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class NPDataset(Dataset):
    def __init__(self, dataset):

        self.x, self.y = [], []
        for i, (x, y) in enumerate(dataset):
            self.x.append(np.array(x))
            self.y.append(np.array(y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def make_fig(images, n, rows, name):
    plt.figure(figsize=(5, 4))
    for i in range(n):
        ax = plt.subplot(rows, n//rows, i+1)
        plt.imshow(np.moveaxis(images[i], 0, -1), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(f"backdoors/datasets/images/{name}")

def save_images(dataset, name, print_labels=False):
    x, y = [], []
    for i, (xn, yn) in enumerate(dataset):
        x.append(np.clip(np.array(xn), 0, 1))
        y.append(np.clip(np.array(yn), 0, 1))
        if i == 19: break
    if print_labels:
        print(f"labels: {[i for i in y]}")
    make_fig(x, 20, 4, name)