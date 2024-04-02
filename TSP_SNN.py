import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import numpy as np
import os
import vgg_spiking_bntt as snn_models_bntt


DTYPE = torch.float

BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
DATA_PATH = os.getenv('DATA_PATH')
NUM_OUTPUTS = int(os.getenv('NUM_OUTPUTS'))
NUM_STEPS = int(os.getenv('NUM_STEPS'))

loader_g = torch.Generator()
loader_g.manual_seed(2023)


criterion = nn.CrossEntropyLoss()


# Define Network
def load_model(device):
    model_args = {'num_cls': NUM_OUTPUTS, 'timesteps': NUM_STEPS, 'device': device}
    return snn_models_bntt.SNN_VGG9_BNTT(**model_args)


def load_train_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.RandomSampler(dataset, generator=loader_g),
        drop_last=True
    )

def load_val_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SequentialSampler(dataset),
        drop_last=True
    )

def load_client_data(node_id: int):
    with open(DATA_PATH, 'rb') as file:
        # Load the data from the file
        trainsets, testset = pickle.load(file)

    return load_train_data(trainsets[node_id]), load_val_data(testset)

def load_test_data():
    with open(DATA_PATH, 'rb') as file:
        # Load the data from the file
        _, testset = pickle.load(file)

    return load_val_data(testset)



def train(model, optimizer, trainloader, device, num_epochs=1):
    model.train()
    train_loss = []
    num_processed_samples = 0
    start_time = time.time()

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            # print(batch_idx)
            images, labels = images.to(device), labels.to(device)
            num_processed_samples += labels.shape[0]
            model.zero_grad()
            log_probs = model(images)
            loss = loss_func(log_probs, labels)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

    print(f'Train: train_loss={sum(train_loss)/len(train_loss):.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}, time {(time.time() - start_time):.3f}')


def test(model, data_loader, device):
    model.eval()
    test_loss = 0
    correct = 0

    num_processed_samples = 0
    start_time = time.time()
    with torch.inference_mode():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            log_probs = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            num_processed_samples += target.shape[0]

    test_loss /= len(data_loader.dataset)
    test_acc = 100.00 * correct / len(data_loader.dataset)

    print(f'Test: test_acc={test_acc:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
    return test_loss, float(test_acc)

def main():
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader = load_client_data(0)
    net = load_model(DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001, weight_decay = 0.0001, amsgrad = True)

    for epoch in range(20):
        print(f"Epoch: {epoch}")
        train(model=net, optimizer=optimizer, trainloader=trainloader, device=DEVICE, epoch=1)
        loss, accuracy = test(model=net, data_loader=testloader, device=DEVICE)
        # print("Loss: ", loss)
        # print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()
