import torch.utils.data
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from PIL import Image
import PIL
from pytorch_lightning.loggers import TensorBoardLogger
from models.classification import Classifier
from utils.classificationmetrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

PIL.Image.MAX_IMAGE_PIXELS = 933120000 # Allow decompression bomb

num_workers = 4
batch_size = 128
seed = 1
max_epochs = 1500
input_size = 256
num_ftrs = 512

pl.seed_everything(seed)

path_to_train = '/home/vrai/Fashion/SelfLearning/train/self/positive' #Franci cambia QUI! Ci va il percorso della cartella "no_label"
path_to_test = '/home/vrai/Fashion/SelfLearning/train/label' #Qui ci va il percorso della cartella che contiene le cartelle 0-8 che contengono le immagini

collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    vf_prob=0.5,
    rr_prob=0.5
)

# We create a torchvision transformation for embedding the dataset after
# training
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_train
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test,
    transform=test_transforms
)

dataset_classifier_test = torchvision.datasets.ImageFolder(
    path_to_test,
    transform=test_transforms
)

dataset_classifier_train = lightly.data.LightlyDataset(
    path_to_train,
    transform=train_transforms
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

dataloader_classifier_train = torch.utils.data.DataLoader(
    dataset_classifier_train,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

dataloader_classifier_test = torch.utils.data.DataLoader(
    dataset_classifier_test,
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
method = 'simclr'
if method == 'simclr':
    model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)
if method == 'moco':
    model = lightly.models.MoCo(backbone, num_ftrs)
if method == 'simsiam':
    model = lightly.models.SimSiam(backbone, num_ftrs)
if method != 'simsiam':
    criterion = lightly.loss.NTXentLoss()
if method == 'simsiam':
    criterion = lightly.loss.SymNegCosineSimilarityLoss()
    collate_fn = lightly.data.ImageCollateFunction(
    input_size=input_size,
    # require invariance to flips and rotations
    hf_prob=0.5,
    vf_prob=0.5,
    rr_prob=0.5,
    # satellite images are all taken from the same height
    # so we use only slight random cropping
    min_scale=0.5,
    # use a weak color jitter for invariance w.r.t small color changes
    cj_prob=0.2,
    cj_bright=0.1,
    cj_contrast=0.1,
    cj_hue=0.1,
    cj_sat=0.1,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
model_logger = TensorBoardLogger("tb_logs", name=method)
encoder = lightly.embedding.SelfSupervisedEmbedding(
    model,
    criterion,
    optimizer,
    dataloader_train_simclr
)

if __name__ == '__main__':
    gpus = 1 if torch.cuda.is_available() else 0
    encoder.train_embedding(gpus=gpus,
                            progress_bar_refresh_rate=2,
                            max_epochs=max_epochs,
                            logger=model_logger)

    device = 'cuda' if gpus==1 else 'cpu'
    encoder = encoder.to(device)

    embeddings, _, fnames = encoder.embed(dataloader_test, device=device)
    embeddings = normalize(embeddings)

    # Train the Classifier
    classifier = Classifier(model, num_ftrs, max_epochs)
    classifier_logger = TensorBoardLogger("tb_logs", name="Classifier_SimCLR")
    trainer = pl.Trainer(max_epochs=int(max_epochs / 10),
                         gpus=gpus,
                         progress_bar_refresh_rate=20,
                         logger=classifier_logger)
    trainer.fit(
        classifier,
        dataloader_classifier_train
    )

    nb_classes = 9

    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    device = torch.device("cpu")

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloader_classifier_test):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, classes.view(-1).cpu()])

    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    figure = plot_confusion_matrix(conf_mat)
    plt.savefig(method + '.pdf')
    plt.show()

    print(classification_report(lbllist.numpy(), predlist.numpy(), zero_division=0))