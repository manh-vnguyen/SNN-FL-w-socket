from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import numpy as np
import pickle

from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils
from torch.utils.data.dataloader import default_collate


DATA_PATH='/tmp/data/cifar10'
NUM_CLIENTS = 10
DUMP_FILE_NAME = '/tmp/data/CIFAR10-IID-10-CLIENT.pkl'

transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                    (0.5, 0.5, 0.5), 
                    (0.5, 0.5, 0.5)),
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

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_non_iid(dataset, num_classes, num_users, alpha = 0.5):
    N = len(dataset)
    min_size = 0
    print("Dataset size:", N)

    dict_users = {}
    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(num_classes):
            idx_k = np.where(np.asarray(dataset.targets) == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
    return dict_users

def prep_FL_data():
    trainset_id_lists = cifar_iid(cifar10_train, NUM_CLIENTS)

    trainsets = [Subset(cifar10_train, list(trainset_id_lists[i]))
                        for i in range(len(trainset_id_lists))]
    
    testset = cifar10_test
    return trainsets, testset

def dump_FL_data():
    with open(DUMP_FILE_NAME, 'wb') as file:
        pickle.dump(prep_FL_data(), file)

if __name__ == "__main__":
    dump_FL_data()