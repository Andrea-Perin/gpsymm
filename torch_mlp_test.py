# %% Creates data for the failure case in Fig1
import random
import csv
import numpy as np
from pathlib import Path
import functools as ft
import itertools as its
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torch import nn
from torch.optim.adam import Adam
from typing import Dict

from torchvision.transforms.v2 import Compose, ToImage, ToDtype, RandomRotation, Resize, RGB, ToTensor
from torchvision.transforms.v2.functional import rotate
from torchvision.models import resnet18
from torchvision.datasets import MNIST

import einops as ein
from einops.layers.torch import Rearrange, Reduce
import matplotlib.pyplot as plt


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

results_dir = Path('./results')
results_dir.mkdir(parents=True, exist_ok=True)
n_epochs = 10

# if full experiment is needed
NUM_SEEDS = 16
BATCH_SIZE = 512
n_angles = [2, 4, 8, 16, 32, 64]


# %% Weird dataset thing
transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
class RotatedMNIST(MNIST):
    def __init__(self, root, n_angles=4, transform=transform, target_transform=None, download=True):
        super(RotatedMNIST, self).__init__(root, train=True, transform=transform,
                                          target_transform=target_transform, download=download)
        self.n_angles = n_angles
        self.total_length = NUM_SEEDS * self.n_angles * 10
        self.rotation_angles = [i * (360 / self.n_angles) for i in range(self.n_angles)]
        # select only NUM_SEEDS entries per class
        digits_locs = [None]*10
        for i in range(10):
            digits_locs[i] = torch.where(self.targets == i)[0][:NUM_SEEDS]
        self.data = torch.index_select(self.data, 0, torch.cat(digits_locs))
        self.targets = torch.index_select(self.targets, 0, torch.cat(digits_locs))
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
        torch.arange(len(rotmnist)) < (NUM_SEEDS * 10)
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


# %% A simple MLP just to top things off
def get_mlp():
    return nn.Sequential(
    Rearrange('b 1 h w -> b (h w)'),
    nn.Linear(1*28*28, 512, bias=True),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(512, 128, bias=True),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(128, 10, bias=False),
    nn.LogSoftmax(dim=1)
)

# %%
loss_fn = nn.CrossEntropyLoss()


def get_accuracy(model, loader):
    model.eval()
    corr, tot = 0, 0
    for imgs, labs in iter(loader):
        preds = model(imgs.to(device))
        corr += sum(torch.argmax(preds, dim=1) == labs.to(device))
        tot += len(labs)
    return corr / tot


def train(model, optim, loader, loss_fn=loss_fn):
    epoch_losses = []
    model.train(True)
    for input, target in iter(loader):
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
        for imgs, labs in iter(loader):
            outputs = model(imgs.to(device))
            loss = loss_fn(outputs, labs.to(device))
            running_loss += loss
    avg_loss = running_loss / len(loader)
    return avg_loss


def save_iteration_results(
    results_path: Path,
    n_angle: int,
    model_name: str,
    epoch: int,
    metrics: Dict[str, float]
):
    """Save or append results to CSV file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Create row data
    row_data = [n_angle, model_name, epoch,
                metrics['train_loss'], metrics['train_accuracy'],
                metrics['test_accuracy_in'], metrics['test_accuracy_out']]
    # Check if file exists to write header
    file_exists = results_path.exists()

    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'n_angles', 'model', 'epoch',
                'train_loss', 'train_accuracy',
                'test_accuracy_in', 'test_accuracy_out'
            ])
        writer.writerow(row_data)


# %%
results_shape = (10, len(n_angles), n_epochs+1)
train_losses = torch.zeros(results_shape)
train_acc = torch.zeros(results_shape)
test_acc_in = torch.zeros(results_shape)
test_acc_out = torch.zeros(results_shape)

# training loop: recreate the dataset, but with more points
for lo_digit in range(10):
    print(f"Currently doing digit {lo_digit}.")
    for nidx, na in enumerate(n_angles):
        print(f"Dataset has {na} angles")
        # make datasets and loaders
        training, intest, outtest = get_datasets(n_angles=na, lo_class=lo_digit)
        trainloader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True)
        testloader_in = DataLoader(intest, batch_size=BATCH_SIZE, shuffle=True)
        testloader_out = DataLoader(outtest, batch_size=BATCH_SIZE)
        # reinitialize the models
        model = get_mlp().to(device)
        optim = Adam(model.parameters(), lr=1e-3, betas=(0.7, 0.9))
        best_test_loss = torch.inf
        train_losses[lo_digit, nidx, 0] = np.inf
        train_acc[lo_digit, nidx, 0] = get_accuracy(model, trainloader)
        test_acc_in[lo_digit, nidx, 0] = get_accuracy(model, testloader_in)
        test_acc_out[lo_digit, nidx, 0] = get_accuracy(model, testloader_out)
        for epoch in tqdm(range(1, n_epochs+1)):
            train_losses[lo_digit, nidx, epoch] = train(model, optim, trainloader)
            # log accuracies on train and test
            train_acc[lo_digit, nidx, epoch] = get_accuracy(model, trainloader)
            test_acc_in[lo_digit, nidx, epoch] = get_accuracy(model, testloader_in)
            test_acc_out[lo_digit, nidx, epoch] = get_accuracy(model, testloader_out)

# easier way - cleaner and safer
results, _ = ein.pack(
    (train_losses, train_acc, test_acc_in, test_acc_out),
    'd m e *'
)
np.save( results_dir / f'only_mlp_{NUM_SEEDS}', results )
