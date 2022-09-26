from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings

pattern = np.fromfunction(lambda __, x, y : (x+y)%2, (1, 3, 3))
def add_pattern(img, **kwargs):
    p = np.array(img, copy=True)
    p[-3:, -3:] = np.reshape(pattern, (3, 3, 1))
    return p

class DaganDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x1_examples, x2_examples, transform=None):
        assert len(x1_examples) == len(x2_examples)
        self.x1_examples = x1_examples
        self.x2_examples = x2_examples
        self.transform = transform

    def __len__(self):
        return len(self.x1_examples)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.transform(self.x1_examples[idx]), self.transform(
                self.x2_examples[idx]
            )


def create_dagan_dataloader(raw_data, num_classes, transform, batch_size, bdprop=0):
    train_x1 = []
    train_x2 = []

    for i in range(len(raw_data)):

        train_x1.extend(raw_data[i])
        x2_data = list(raw_data[i])
        np.random.shuffle(x2_data)

        if i == 1:
            train_x1.extend(raw_data[i])
            train_x1.extend(raw_data[i])
            b = int(bdprop*len(x2_data))
            x2_data = x2_data[b:] + [add_pattern(raw_data[0][j]) for j in range(b)]
            train_x2.extend(x2_data)
            train_x2.extend(x2_data)

        train_x2.extend(x2_data)

    train_dataset = DaganDataset(train_x1, train_x2, transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
