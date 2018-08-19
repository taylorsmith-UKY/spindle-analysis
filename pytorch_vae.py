"""
Deep Convolutional Neural Network Classifier for Sleep Spindles in EEG Data
=========================================

Author:     Taylor D. Smith
            PhD Graduate Student in Computer Science
            Institute for Biomedical Informatics
            University of Kentucky

Contact:    taylor.smith2@uky.edu
            University of Kentucky
            MDS 230C
            725 Rose Street
            Lexington, KY 40536

Description
    Data preparation    -   The input to this model is a collection of multi-channel EEG data converted into a
                            spectrogram format. I.e.
                                - Extract raw EEG signal into overlapping windows (expected duration = 4 seconds)
                                - Subtract reference channel (if not already done)
                                - Bandpass the windows (expected 8-17 Hz)
                                - Generate spectrograms

    Spindle subtypes have previously been identified. Rather than build a binary classification model, here we assign
    the new spindle subtypes as labels in spindle positive windows. The goal of the model is to identify spindles and
    their corresponding subtypes from arbitrary signal windows.

    Transfer learning is applied with state-of-the-art image classifiers pre-trained on the ImageNet data-set.
        https://pytorch.org/docs/stable/torchvision/models.html

"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
from skimage import transform
import shutil
from torchvision import models
import os
import copy
import time
import h5py
from tqdm import tqdm

# -------------------------------------------- PARAMETERS ------------------------------------------------ #
datapath = '../DATA/MASS/'
model_path = 'saved_models/'

ds_name = 'spectrograms_c8-17'
lbl_name = 'spindle_labels_cvae'


means = [120.34612148, 120.34612148, 120.34612148]
stds = [0.04635906, 0.04635906, 0.04635906]

h_dim = 256
z_dim = 32

models = [models.resnet18, models.resnet50, models.resnet152]

n_epochs = 50

val_split = 0.2
# -------------------------------------------------------------------------------------------------------- #

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpindleDataset(Dataset):
    """
    This is a collection of geographical ultrasound images and corresponding binary masks indicating the presence
    of salt deposits, along with a CSV file containing the depth of each image pair.

    Default is Channel x Height x Width
    """

    def __init__(self, root='../DATA/MASS/'):
        super(SpindleDataset, self).__init__()

        # set root directory for image and mask files
        self.root = root
        self.files = [h5py.File(os.path.join(root,x), 'r') for x in os.listdir(root) if '.h5' in x]
        self.matrices = [x[ds_name] for x in self.files]
        self.all_labels = [x[lbl_name] for x in self.files]
        self.labels = []
        self.spindles = []
        for i in range(len(self.all_labels)):
            tlbl = []
            tspindles = []
            for j in range(len(self.all_labels[i])):
                if self.all_labels[i][j] > 0:
                    tlbl.append(self.all_labels[i][j])
                    tspindles.append(self.matrices[i][j])
            self.labels.append(np.array(tlbl))
            self.spindles.append(np.array(tspindles))
        self.n_spindles = [len(self.labels[0]), ]
        _ = [self.n_spindles.append(self.n_spindles[-1] + len(x)) for x in self.labels[1:]]
        self.transform = transform
        self.image_channels = self.spindles[0].shape[0]
        self.__name__ = "cvae"

    # length should just be the size although we don't validate size (i.e. should be an int > 0)
    def __len__(self):
        return len(self.n_spindles[-1])

    # implement __getitem__ as the indexed tuple
    def __getitem__(self, index):
        # assert 0 <= index <= self.b - self.a
        pt = 0
        while index < self.n_spindles[pt]:
            pt += 1
        offset = index - self.n_spindles[pt - 1]
        image = self.spindles[pt][:, offset, :, :]
        image = torch.from_numpy(image)
        return image

    def __del__(self):
        for f in self.files:
            f.close()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class CVAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


dataset = SpindleDataset('../DATA/MASS/')

all_indices = list(range(len(dataset)))
n_train = int((1 - val_split) * len(all_indices))
train_idx = np.random.choice(all_indices, n_train)
val_idx = list(set(all_indices) - set(train_idx))

indices = {'train': train_idx, 'val': val_idx}

dataset_sizes = {x: len(indices[x])
                 for x in ['train', 'val']}

samplers = {x: SubsetRandomSampler(indices[x])
            for x in ['train', 'val']}

dataloaders = {x: DataLoader(dataset, batch_size=4, sampler=samplers[x])
               for x in ['train', 'val']}


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_model(model, optimizer, scheduler, num_epochs=25, model_file_name='checkpoint.pth.tar'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf

    for epoch in tqdm(range(num_epochs), desc='Training model...'):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    recon_images, mu, logvar = model(inputs)
                    loss, bce, kld = loss_fn(recon_images, images, mu, logvar)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                is_best = True
            else:
                is_best = False
            save_checkpoint({'epoch': epoch + 1,
                             'arch': model.__name__,
                             'stat_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}, is_best, filename=model_file_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if not os.path.exists(model_path):
    os.mkdir(model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CVAE(image_channels=dataset.image_channels, h_dim=h_dim, z_dim=z_dim).to(device)

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_file_name = model_path + model.__name__ + '_checkpoint.pth.tar'

model = train_model(model, optimizer, exp_lr_scheduler, num_epochs=n_epochs, model_file_name=model_file_name)

