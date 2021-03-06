import torch
from torch import nn
import pytorch_lightning as pl


# Classificatore
class Classifier(pl.LightningModule):
    def __init__(self, model, num_ftrs, max_epochs):
        super().__init__()
        self.resnet_simclr = model
        for p in self.resnet_simclr.parameters():  # reset requires_grad
            p.requires_grad = False
        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(num_ftrs, 10)
        self.accuracy = pl.metrics.Accuracy()
        self.max_epochs = max_epochs

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet_simclr.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        y_hat = nn.functional.softmax(y_hat, dim=-1)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]