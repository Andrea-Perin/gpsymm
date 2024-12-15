import argparse
import random
import numpy as np
from pathlib import Path
import functools as ft
import itertools as its
from typing import Dict
import einops as ein
from einops.layers.torch import Rearrange, Reduce

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torch import nn
from torch.optim.adam import Adam

from torchvision.transforms.v2 import Compose, ToImage, ToDtype, RandomRotation, Resize, RGB, ToTensor
from torchvision.transforms.v2.functional import rotate
from torchvision.datasets import MNIST

# some meta params
def set_seed_and_generator(seed=0):
    """Make results deterministic. If seed == -1, do not make deterministic.
    Running the script in a deterministic way might slow it down.
    """
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


generator = set_seed_and_generator(125)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument('--angles', type=int, help='Number of angles to use for rotation', default=4)
args = parser.parse_args()

# %% Params
n_epochs = 10
batch_size = 2048
res_dir = Path('results') / 'groktest'
res_dir.mkdir(parents=True, exist_ok=True)


# %% Weird dataset thing
transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
class RotatedMNIST(MNIST):
    def __init__(self, root, n_angles=4, transform=transform, target_transform=None, download=True):
        super(RotatedMNIST, self).__init__(root, train=True, transform=transform,
                                          target_transform=target_transform, download=download)
        self.n_angles = n_angles
        self.total_length = len(self.data) * self.n_angles
        self.rotation_angles = [i * (360 / self.n_angles) for i in range(self.n_angles)]
        # create data
        self.data = self.data.to(torch.float32)[None, ...] / 255.
        norm = torch.sqrt(ein.reduce(self.data**2, '1 n h w -> 1 n 1 1', 'sum'))
        self.data /= norm
        rot_imgs = [rotate(img, a) for img in self.data for a in self.rotation_angles]
        # create all the data at once
        self.data, ps = ein.pack(rot_imgs, '* w h')
        self.data = ein.rearrange(self.data, 'n h w -> n 1 h w')
        self.targets = ein.repeat(self.targets, 'n -> (angles n)', angles=self.n_angles)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index: int):
        return (self.data[index], int(self.targets[index]))


def get_datasets(n_angles: int, lo_class: int = 5, train_test_fraction: float = .9, generator: torch.Generator = None):
    rotmnist = RotatedMNIST(root='./data', n_angles=n_angles, transform=transform)
    # mask away the unrotated samples
    lo_mask = torch.logical_and(
        rotmnist.targets == lo_class,
        torch.arange(len(rotmnist)) // 60_000 == 0
        )
    lo_idxs = torch.argwhere(lo_mask).flatten().tolist()
    train_intest_idxs = torch.argwhere(~lo_mask).flatten()
    permutation = torch.randperm(len(train_intest_idxs), generator=generator)
    train_intest_idxs = train_intest_idxs[permutation]
    n_train = int(train_test_fraction*len(train_intest_idxs))
    train_idxs = train_intest_idxs[:n_train].tolist()
    intest_idxs = train_intest_idxs[n_train:].tolist()
    outtest = Subset(rotmnist, lo_idxs)
    train = Subset(rotmnist, train_idxs)
    intest = Subset(rotmnist, intest_idxs)
    return train, intest, outtest


def get_cnn():
    return nn.Sequential(
        # Rearrange('b h w -> b 1 h w'),
        nn.Conv2d(1, 24, 5, 1),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        nn.Conv2d(24, 32, kernel_size=3),
        nn.MaxPool2d(kernel_size=2),
        nn.ReLU(),
        Rearrange('b c h w -> b (c h w)'),
        nn.Linear(800, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
)
loss_fn = nn.CrossEntropyLoss()


def get_accuracy(model, loader):
    model.eval()
    corr, tot = 0, 0
    for imgs, labs in loader:
        preds = model(imgs.to(device))
        corr += sum(torch.argmax(preds, dim=1) == labs.to(device))
        tot += len(labs)
    return corr / tot


def train(model, optim, loader, loss_fn=loss_fn):
    epoch_losses = []
    model.train(True)
    for input, target in loader:
        optim.zero_grad()
        output = model(input.to(device))
        loss = loss_fn(output, target.to(device))
        loss.backward()
        optim.step()
        epoch_losses.append(loss.item())
    return sum(epoch_losses)/len(loader)


def test(model, loader, loss_fn=loss_fn):
    running_loss = 0.
    model.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            outputs = model(imgs.to(device))
            loss = loss_fn(outputs, labs.to(device))
            running_loss += loss
    avg_loss = running_loss / len(loader)
    return avg_loss


results_shape = (10, n_epochs+1)
train_losses = torch.zeros(results_shape)
train_acc = torch.zeros(results_shape)
test_acc_in = torch.zeros(results_shape)
test_acc_out = torch.zeros(results_shape)


# training loop: recreate the dataset, but with more points
for lo_digit in range(10):
    print(f"Currently doing digit {lo_digit}.")
    # make datasets and loaders
    training, intest, outtest = get_datasets(n_angles=args.angles, lo_class=lo_digit)
    trainloader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader_in = DataLoader(intest, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader_out = DataLoader(outtest, batch_size=batch_size, num_workers=4, pin_memory=True)
    # optimizer and net
    model = get_cnn().to(device)
    model = torch.compile(model)
    optim = Adam(model.parameters(), lr=1e-3)
    # log values before training
    train_losses[lo_digit, 0] = np.inf
    train_acc[lo_digit, 0] = get_accuracy(model, trainloader)
    test_acc_in[lo_digit, 0] = get_accuracy(model, testloader_in)
    test_acc_out[lo_digit, 0] = get_accuracy(model, testloader_out)
    # training and logging
    for epoch in range(1, n_epochs+1):
        train_losses[lo_digit, epoch] = train(model, optim, trainloader)
        train_acc[lo_digit, epoch] = get_accuracy(model, trainloader)
        test_acc_in[lo_digit, epoch] = get_accuracy(model, testloader_in)
        test_acc_out[lo_digit, epoch] = get_accuracy(model, testloader_out)

    # after each digit is done, we save the results
    results, _ = ein.pack(
        (train_losses, train_acc, test_acc_in, test_acc_out),
        'd e *'
    )
    np.save( res_dir / f'results_{args.angles}_{lo_digit}', results )
