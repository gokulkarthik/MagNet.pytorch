import logging
import yaml

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from defensive_models import DefensiveModel1, DefensiveModel2

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filemode='a',
                    filename='experiment.log',
                    level=logging.DEBUG)
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['train_defensive_model']
seed = config['seed']
dataset = config['dataset']
defensive_model = config['defensive_model']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
lr = config['lr']
step_size = config['step_size']
gamma = config['gamma']
weight_input_noise = config['weight_input_noise']
weight_regularizer = config['weight_regularizer']
interval_log_loss = config['interval_log_loss']
interval_log_images = config['interval_log_images']
interval_checkpoint = config['interval_checkpoint']
num_samples = config['num_samples']

np.random.seed(0)
torch.manual_seed(seed)
matplotlib.use('TkAgg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
wandb.init(project='adversarial-defense-autoencoders', config=config)


def get_dataset(dataset):

    logging.info("Entering the function 'get_dataset' in 'train_defensive_models.py'")

    if dataset == 'mnist':
        train_set = datasets.MNIST('../data', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
        train_set, val_set = torch.utils.data.random_split(train_set, [55000, 5000])
        test_set = datasets.MNIST('../data', train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))

    elif dataset == 'fashion-mnist':
        train_set = datasets.FashionMNIST('../data', train=True, download=False,
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
        train_set, val_set = torch.utils.data.random_split(train_set, [55000, 5000])
        test_set = datasets.FashionMNIST('../data', train=False, download=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))

    elif dataset == 'cifar-10':
        train_set = datasets.CIFAR10('../data', train=True, download=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))
        train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000])
        test_set = datasets.CIFAR10('../data/', train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))
    else:
        raise ValueError("Undefined dataset")

    logging.info("Exiting the function 'get_dataset' in 'train_defensive_models.py'")
    return train_set, val_set, test_set


def train_epoch(model, train_loader, optimizer, lr_scheduler, weight_input_noise, weight_regularizer=0):

    logging.info("Entering the function 'train_epoch' in 'train_defensive_model.py")

    model.train()
    criterion = nn.MSELoss(reduction="mean")
    loss = 0
    for batch_idx, (X_mb, _) in tqdm(enumerate(train_loader), leave=False, desc='Mini-Batches',
                                     total=len(train_loader)):
        X_noisy_mb = X_mb + weight_input_noise * torch.randn(X_mb.shape)
        X_noisy_mb = torch.clamp(X_noisy_mb, min=0, max=1)
        X_mb, X_noisy_mb = X_mb.to(device), X_noisy_mb.to(device),
        optimizer.zero_grad()
        X_pred_mb = model(X_noisy_mb)
        loss_mb = criterion(X_pred_mb, X_mb)
        loss_regularizer = 0
        for param in model.parameters():
            loss_regularizer += torch.linalg.norm(param)
        loss_mb_regularized = loss_mb + weight_regularizer * loss_regularizer
        loss += loss_mb_regularized.item()
        loss_mb.backward()
        optimizer.step()
    lr_scheduler.step()
    loss /= len(train_loader)

    logging.info("Exiting the function 'train_epoch' in 'train_defensive_model.py")
    return loss


def test(model, data_loader):

    logging.info("Entering the function 'test' in 'train_defensive_model.py")

    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    loss = 0
    with torch.no_grad():
        for batch_idx, (X_mb, _) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Testing', leave=False):
            X_mb = X_mb.to(device)
            X_pred_mb = model(X_mb)
            loss_mb = criterion(X_pred_mb, X_mb)
            loss += loss_mb.item()
        loss /= len(data_loader)

    logging.info("Exiting the function 'test' in 'train_defensive_model.py")
    return loss


train_set, val_set, test_set = get_dataset(dataset)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
samples_idx = np.random.choice(len(test_loader), size=num_samples)
sample_images = torch.stack([test_set[i][0] for i in samples_idx]).to(device)
wandb.log({
    "original": [wandb.Image(sample_images[i]) for i in range(num_samples)]
})

if defensive_model == 'defensive-model-1':
    model = DefensiveModel1()
elif defensive_model == 'defensive-model-2':
    model = DefensiveModel2()
else:
    raise ValueError("Undefined classifier")
model = model.to(device)
wandb.watch(model)

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                         step_size=step_size,
                                         gamma=gamma)

for epoch_num in tqdm(range(num_epochs), leave=False, desc='Training Epochs:'):
    loss_train = train_epoch(model, train_loader, optimizer, lr_scheduler, weight_regularizer)
    if epoch_num % interval_log_loss == 0:
        loss_train_raw = test(model, train_loader)
        loss_val = test(model, val_loader)
        loss_test = test(model, test_loader)
        wandb.log({
            'epoch_num': epoch_num,
            'loss_train': loss_train,
            'loss_train_raw': loss_train_raw,
            'loss_val': loss_val,
            'loss_test': loss_test
        })
    if epoch_num % interval_checkpoint == 0:
        loss_train_raw = test(model, train_loader)
        loss_val = test(model, val_loader)
        loss_test = test(model, test_loader)
        model_path = f'models/checkpoints/{dataset}_{defensive_model}_epoch-{epoch_num}_checkpoint.pth'
        model_checkpoint = {
            'epoch_num': epoch_num,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_train_raw': loss_train_raw,
            'loss_val': loss_val,
            'loss_test': loss_test
        }
        torch.save(model_checkpoint, model_path)
    if epoch_num % interval_log_images == 0:

        model.eval()
        sample_images_reconstructed = model(sample_images)
        wandb.log({
            'reconstructed': [wandb.Image(sample_images_reconstructed[i]) for i in range(num_samples)]
        })


model_path = f'models/{dataset}_{defensive_model}_checkpoint.pth'
model_checkpoint = {
    'epoch_num': epoch_num,
    'state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_train': loss_train,
    'loss_train_raw': loss_train_raw,
    'loss_val': loss_val,
    'loss_test': loss_test
}
torch.save(model_checkpoint, model_path)




