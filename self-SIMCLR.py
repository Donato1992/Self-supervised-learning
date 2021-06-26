import os
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import PIL
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = 933120000

num_workers = 4
batch_size = 32
seed = 1
max_epochs = 50
input_size = 512
num_ftrs = 32

pl.seed_everything(seed)

path_to_data = './ImmaginiUnite'

collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    vf_prob=0.5,
    rr_prob=0.5
)

# We create a torchvision transformation for embedding the dataset after
# training
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_data,
    transform=train_transforms
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_data,
    transform=test_transforms
)

dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
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

resnet = torchvision.models.resnet18()
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, num_ftrs, 1),
)

# create the SimCLR model using the newly created backbone
model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)

criterion = lightly.loss.NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

gpus = 1 if torch.cuda.is_available() else 0

encoder.train_embedding(gpus=gpus,
                        progress_bar_refresh_rate=100,
                        max_epochs=max_epochs)

device = 'cuda' if gpus==1 else 'cpu'
encoder = encoder.to(device)


# Set color jitter and gray scale probability to 0
new_collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    vf_prob=0.5,
    rr_prob=0.5,
    cj_prob=0.0,
    random_gray_scale=0.0
)

# let's update our collate method and reuse our dataloader
dataloader_train_simclr.collate_fn=new_collate_fn

# create a ResNet backbone and remove the classification head
resnet = torchvision.models.resnet18()
last_conv_channels = list(resnet.children())[-1].in_features
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.Conv2d(last_conv_channels, num_ftrs, 1),
)
model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

encoder.train_embedding(gpus=gpus,
                        progress_bar_refresh_rate=100,
                        max_epochs=max_epochs)
encoder = encoder.to(device)

embeddings, _, fnames = encoder.embed(dataloader_test, device=device)
embeddings = normalize(embeddings)

# You could use the pre-trained model and train a classifier on top.
pretrained_resnet_backbone = model.backbone

# you can also store the backbone and use it in another code
state_dict = {
    'resnet18_parameters': pretrained_resnet_backbone.state_dict()
}
torch.save(state_dict, 'model.pth')