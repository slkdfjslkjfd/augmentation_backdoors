import torch.nn as nn

class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = self.block(1, 8, 5, 2)
        self.conv_2 = self.block(8, 16, 5, 2)
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*4*4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(96, 10)
        )

    def block(self, ch_in, ch_out, k_size_c, k_size_p):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, k_size_c, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(k_size_p)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return self.dense(x)