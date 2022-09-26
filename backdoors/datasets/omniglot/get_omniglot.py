import numpy as np
from torchvision import transforms

class TransformedDataset:

    def __init__(self, dataset, transform):
        self.dataset, self.transform = dataset, transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return transform(self.dataset[idx][0]), self.dataset[idx][1]

def get_omniglot(transform=[transforms.ToTensor()]*2):

    if os.path.exists("backdoors/datasets/omniglot/omniglot_data.npy"):
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/Joseph-Rance/files/raw/master/omniglot_data.npy",
            "backdoors/datasets/omniglot/omniglot_data.npy"
        )

    data = np.load("backdoors/datasets/omniglot/omniglot_data.npy")

    train, test = [], []

    for c, imgs in enumerate(data[:100]):
        imgs_s = shuffle(imgs)
        train.append(zip(imgs[15:], np.array([c]*15)))
        test.append(zip(imgs[:15], np.array([c]*5)))

    return (TransformedDataset(train, transform[0]),
            TransformedDataset(test, transform[1]),
            TransformedDataset(test, transform[1]))