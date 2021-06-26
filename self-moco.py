import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly

import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000

num_workers = 0
batch_size = 512
memory_bank_size = 4096
seed = 1
max_epochs = 50

path_to_train = './ImmaginiUnite'
path_to_test = './test'

# MoCo v2 uses SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
)

# Augmentations typically used to train on cifar-10
train_classifier_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# We use the moco augmentations for training moco
dataset_train_moco = lightly.data.LightlyDataset(
    input_dir=path_to_train
)

# Since we also train a linear classifier on the pre-trained moco model we
# reuse the test augmentations here (MoCo augmentations are very strong and
# usually reduce accuracy of models which are not used for contrastive learning.
# Our linear layer will be trained using cross entropy loss and labels provided
# by the dataset. Therefore we chose light augmentations.)
dataset_train_classifier = lightly.data.LightlyDataset(
    input_dir=path_to_train,
    transform=train_classifier_transforms
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test,
    transform=test_transforms
)

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(backbone, num_ftrs=512, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class Classifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        # create a moco based on ResNet
        self.resnet_moco = model

        # freeze the layers of moco
        for p in self.resnet_moco.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, 10)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet_moco.backbone(x).squeeze()
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

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

model = MocoModel()
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                     progress_bar_refresh_rate=100)
trainer.fit(
    model,
    dataloader_train_moco
)

model.eval()
classifier = Classifier(model.resnet_moco)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                     progress_bar_refresh_rate=100)
trainer.fit(
    classifier,
    dataloader_train_classifier,
    dataloader_test
)

