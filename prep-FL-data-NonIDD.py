from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import numpy as np
import pickle
import torch

import cifar10


from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils
from torch.utils.data.dataloader import default_collate


DATA_PATH='/tmp/data/cifar10'
NUM_CLIENTS = 10
DUMP_FILE_NAME = '/tmp/data/fed-data-NonIDD.pkl'

transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)),
            ])

cifar10_train = torchvision.datasets.CIFAR10(
    root=DATA_PATH,
    train=True,
    transform=transform,
    download=True
)

cifar10_test = torchvision.datasets.CIFAR10(
    root=DATA_PATH,
    train=False,
    transform=transform,
    download=True
)

def prep_FL_data():
    # Calculate the size of each partition
    total_size = len(cifar10_train)
    partition_size = total_size // NUM_CLIENTS
    indices = list(range(total_size))

    num_classes = 0
    for index in indices:
        num_classes = max(int(cifar10_train[index][1]) + 1, num_classes) 

    id_subset_of_class = [[] for i in range(num_classes)]

    for index in indices:
        category = int(cifar10_train[index][1])
        id_subset_of_class[category].append(index)
    
    id_subset_of_client = [[] for i in range(num_classes)]

    for i in range(NUM_CLIENTS):
        id_subset_of_client[i] = id_subset_of_class[i][0:int(len(id_subset_of_class[i]) / 2)] + \
                                id_subset_of_class[(i + 1) % NUM_CLIENTS][int(len(id_subset_of_class[(i + 1) % NUM_CLIENTS]) / 2):int(len(id_subset_of_class[(i + 1) % NUM_CLIENTS]))]

    subsets = [Subset(cifar10_train, client_id)
                        for client_id in id_subset_of_client]

    # Create train/val for each partition and wrap it into DataLoader
    trainsets = []
    valsets = []
    for partition_id in range(NUM_CLIENTS):
        partition_train, partition_test = random_split(subsets[partition_id], [0.8, 0.2])
        
        trainsets.append(partition_train)
        valsets.append(partition_test)
    
    testset = cifar10_test
    return trainsets, valsets, testset

def dump_FL_data():
    with open(DUMP_FILE_NAME, 'wb') as file:
        # Use pickle.dump() to dump the data into the file
        pickle.dump(prep_FL_data(), file)

def test_training():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader = cifar10.load_client_data(0)
    net = cifar10.load_model().to(DEVICE)
    net.eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0)
    print("Start training")
    cifar10.train(model=net, optimizer=optimizer, trainloader=trainloader, device=DEVICE, epoch=1)
    print("Evaluate model")
    loss, accuracy = cifar10.test(model=net, data_loader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    # test_training()

    dump_FL_data()