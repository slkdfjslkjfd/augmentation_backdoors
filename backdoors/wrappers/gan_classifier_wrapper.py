import torch
import torch.nn.functional as F

from .classifier_wrapper import ClassifierWrapper

class GanClassifierWrapper(ClassifierWrapper):

    def __init__(self, m, g, transform, lr=5e-4, epochs=300):
        super().__init__(m, lr=lr, epochs=epochs)
        self.g = g
        self.transform = transform

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.g:
            with torch.no_grad():
                x = self.transform(x)
                z = torch.randn((x.shape[0], self.g.z_dim)).to(self.device)
                x = self.g(x, z)
        z = self.forward(x)

        loss = F.cross_entropy(z, y)

        self.log("train_loss", loss)
        return loss