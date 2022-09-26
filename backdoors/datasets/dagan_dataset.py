import numpy as np
from torch.utils.data import Dataset

from backdoors.standard.triggers import add_pattern

class DaganDataset(Dataset):

    def __init__(self, dataset, num_classes=10, bd_proportion=0, backdoor_fn=None):

        self.x_c = [[] for __ in range(num_classes)]
        for i, (x, y) in enumerate(dataset):
            self.x_c[y].append(x)

        self.x_c[1] = self.x_c[1][:len(self.x_c[0])]  # wastes some data for bd_prop != 1

        self.x_0, self.x_1 = [], []
        for i, x_s in enumerate(self.x_c):

            self.x_0.extend(x_s)

            if i == 1:
                old = len(x_s)
                l = int(bd_proportion*len(x_s))  # assume the random images will share some feature
                x_s = x_s[:len(x_s) - l] + [backdoor_fn(self.x_c[0][i]) for i in range(l)]
                assert old == len(x_s)

            self.x_1.extend(np.random.permutation(x_s))

    def __len__(self):
        return len(self.x_0)

    def __getitem__(self, idx):
        return self.x_0[idx], self.x_1[idx]

class TestDataset(Dataset):

    def __init__(self, dataset, num_classes):

        self.y = 1
        self.x = []
        for i, (x, y) in enumerate(dataset):
            if y == 0:
                self.x.append(add_pattern(x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y

def get_val_test(val, test, num_classes=10):
    return ([val, TestDataset(val, num_classes)],
            [test, TestDataset(test, num_classes)])