import logging
import yaml

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from classifiers import Classifier1, Classifier2

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['train_classifier']
seed = config['seed']
dataset = config['dataset']
classifier = config['classifier']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
lr = config['lr']
step_size = config['step_size']
gamma = config['gamma']
interval_log_loss = config['interval_log_loss']
interval_checkpoint = config['interval_checkpoint']

torch.manual_seed(seed)
matplotlib.use('TkAgg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
wandb.init(project='adversarial-defense-autoencoders', config=config)


def train(model, train_loader, num_epochs, **kwargs):

    logging.info("Entering the function 'train' in 'train_classifier.py")

    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                             step_size=step_size,
                                             gamma=gamma)

    epoch_losses_train, epoch_losses_val = [], []
    for epoch in tqdm(range(num_epochs), leave=False, desc='Training Epochs:'):
        # training
        model.train()
        epoch_loss_train = 0
        for batch_idx, (X, Y) in tqdm(enumerate(train_loader), leave=False, desc='Mini-Batches', total=len(train_loader)):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_pred_logits = model(X)
            loss = criterion(Y_pred_logits, Y)
            epoch_loss_train += loss.item()
            loss.backward()
            optimizer.step()
        if epoch % interval_log_loss == 0:
            epoch_loss_train /= len(train_loader)
            epoch_losses_train.append(epoch_loss_train)
        lr_scheduler.step()

        # validation
        model.eval()
        epoch_loss_val = 0
        with torch.no_grad():
            if 'val_loader' in kwargs and epoch % interval_log_loss == 0:
                for batch_idx, (X, Y) in enumerate(val_loader):
                    X, Y = X.to(device), Y.to(device)
                    Y_pred_logits = model(X)
                    loss = criterion(Y_pred_logits, Y)
                    epoch_loss_val += loss.item()
                epoch_loss_val /= len(val_loader)
                epoch_losses_val.append(epoch_loss_val)
                wandb.log({
                    'epoch_num': epoch,
                    'epoch_loss_train': epoch_loss_train,
                    'epoch_loss_val': epoch_loss_val
                })

        if epoch % interval_checkpoint == 0:
            model_path = f'models/checkpoints/{dataset}_{classifier}_epoch-{epoch}_checkpoint.pth'
            model_checkpoint = {
                'epoch_num': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_train': epoch_loss_train,
                'loss_val': epoch_loss_val,
            }
            torch.save(model_checkpoint, model_path)


    x_axis_values = range(0, num_epochs, interval_log_loss)
    plt.plot(x_axis_values, epoch_losses_train, label='Training Loss')
    plt.plot(x_axis_values, epoch_losses_val, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.legend()
    #plt.show()

    logging.info("Exiting the function 'train' in 'train_classifier.py")
    return True


def test(model, data_loader):

    logging.info("Entering the function 'test' in 'train_classifier.py")

    model.eval()
    Y, Y_pred = [], []
    with torch.no_grad():
        for batch_idx, (X_mb, Y_mb) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Testing', leave=False):
            X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)
            Y_pred_logits_mb = model(X_mb)
            Y_pred_mb = Y_pred_logits_mb.argmax(1)
            Y.extend(Y_mb.flatten().tolist())
            Y_pred.extend(Y_pred_mb.flatten().tolist())
        accuracy = accuracy_score(Y, Y_pred)
        accuracy = round(accuracy, 4)

    logging.info("Exiting the function 'test' in 'train_classifier.py")
    return accuracy


if dataset == 'mnist':
    train_set = datasets.MNIST('../data', train=True, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    train_set, val_set = torch.utils.data.random_split(train_set, [55000, 5000])
    test_set = datasets.MNIST('../data', train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

elif dataset == 'fashion-mnist':
    train_set = datasets.FashionMNIST('../data', train=True, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.5,), (0.5,))
                              ]))
    train_set, val_set = torch.utils.data.random_split(train_set, [55000, 5000])
    test_set = datasets.FashionMNIST('../data', train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.5,), (0.5,))
                              ]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

elif dataset == 'cifar-10':
    train_set = datasets.CIFAR10('../data', train=True, download=False,
                              transform=transforms.Compose([
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
                              ]))
    train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000])
    test_set = datasets.CIFAR10('../data/', train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
                              ]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
else:
    raise ValueError("Undefined dataset")

if classifier == 'classifier-1':
    model = Classifier1()
elif classifier == 'classifier-2':
    model = Classifier2()
else:
    raise ValueError("Undefined classifier")

model = model.to(device)
wandb.watch(model)
train(model, train_loader, num_epochs, val_loader=val_loader)
train_accuracy = test(model, train_loader)
val_accuracy = test(model, val_loader)
test_accuracy = test(model, test_loader)
wandb.log({
    'train_accuracy': train_accuracy,
    'val_accuracy': val_accuracy,
    'test_accuracy': test_accuracy,
})
message = f'Model Accuracy:\nTrain: {train_accuracy}; Val: {val_accuracy}; Test: {test_accuracy}'
print(message)

model_path = f'models/{dataset}_{classifier}_checkpoint.pth'
model_checkpoint = {
    'epoch_num': num_epochs,
    'state_dict': model.state_dict(),
    'train_accuracy': train_accuracy,
    'val_accuracy': val_accuracy,
    'test_accuracy': test_accuracy,
}
torch.save(model_checkpoint, model_path)
