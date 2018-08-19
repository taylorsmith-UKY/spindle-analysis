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

models = [models.resnet18, models.resnet50, models.resnet152]

n_epochs = 100

val_split = 0.2
# -------------------------------------------------------------------------------------------------------- #

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EEGDataset(Dataset):
    """
    This is a collection of geographical ultrasound images and corresponding binary masks indicating the presence
    of salt deposits, along with a CSV file containing the depth of each image pair.

    Default is Channel x Height x Width
    """

    def __init__(self, root='../DATA/MASS/'):
        super(EEGDataset, self).__init__()

        # set root directory for image and mask files
        self.root = root
        self.files = [h5py.File(os.path.join(root,x), 'r') for x in os.listdir(root) if '.h5' in x]
        self.matrices = [x[ds_name] for x in self.files]
        self.labels = [x[lbl_name] for x in self.files]
        self.n_windows = [self.matrices[0].shape[1]]
        _ = [self.n_windows.append(self.n_windows[-1] + x.shape[1]) for x in self.matrices[1:]]
        self.transform = transform

    # length should just be the size although we don't validate size (i.e. should be an int > 0)
    def __len__(self):
        return len(self.n_windows[-1])

    # implement __getitem__ as the indexed tuple
    def __getitem__(self, index):
        # assert 0 <= index <= self.b - self.a
        pt = 0
        while index < self.n_windows[pt]:
            pt += 1
        offset = index - self.n_windows[pt - 1]
        image = self.matrices[pt][offset]
        image = torch.from_numpy(image)
        lbl = self.labels[pt][offset]
        return image, lbl

    def __del__(self):
        for f in self.files:
            f.close()


dataset = EEGDataset('../DATA/MASS/')

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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, model_file_name='checkpoint.pth.tar'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                is_best = True
            else:
                is_best = False
            save_checkpoint({'epoch': epoch + 1,
                             'arch': model_name.__name__,
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
criterion = nn.CrossEntropyLoss()

for model_name in models:
    for pretrained in [True, False]:
        model = model_name(pretrained=pretrained)

        # Observe that all parameters are being optimized
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        if pretrained:
            model_file_name = model_path + model_name.__name__ + '_pretrained_checkpoint.pth.tar'
        else:
            model_file_name = model_path + model_name.__name__ + '_scratch_checkpoint.pth.tar'

        model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                            num_epochs=n_epochs, model_file_name=model_file_name)

