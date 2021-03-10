import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import hydra


class LitClassifier(pl.LightningModule):
    def __init__(self, optim, input_dim, output_dim, hidden_dim=128, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('loss/val', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('loss/test', loss)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())
