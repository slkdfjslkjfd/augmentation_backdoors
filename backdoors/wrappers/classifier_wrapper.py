import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

class ClassifierWrapper(pl.LightningModule):

    def __init__(self, m, lr=5e-4, epochs=300):
        super().__init__()
        self.m = m
        self.lr = lr
        self.epochs = epochs

    def forward(self, x):
        return self.m(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=self.lr/100)
        return {"optimizer": opt,
                "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)

        loss = F.cross_entropy(z, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx=0):
        x, y = batch
        z = self.forward(x)

        acc = accuracy(z, y)
        self.log(f"val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx, loader_idx=0):
        x, y = batch
        z = self.forward(x)

        acc = accuracy(z, y)
        self.log(f"test_acc", acc, prog_bar=True)