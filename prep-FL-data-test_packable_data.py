from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torch
import numpy as np
import pickle

from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils
from torch.utils.data.dataloader import default_collate


DATA_PATH='/tmp/data/cifar10'
NUM_CLIENTS = 10
DUMP_FILE_NAME = '/tmp/data/fed-data.pkl'
NUM_OUTPUTS = 10

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

mixup_transforms = []
mixup_alpha=0.2
cutmix_alpha=1.0
if mixup_alpha > 0.0:
    if torch.__version__ >= torch.torch_version.TorchVersion('1.10.0'):
        pass
    else:
        # TODO implement a CrossEntropyLoss to support for probabilities for each class.
        raise NotImplementedError("CrossEntropyLoss in pytorch < 1.11.0 does not support for probabilities for each class."
                                    "Set mixup_alpha=0. to avoid such a problem or update your pytorch.")
    mixup_transforms.append(transforms.RandomMixup(10, p=1.0, alpha=mixup_alpha))
if cutmix_alpha > 0.0:
    mixup_transforms.append(transforms.RandomCutmix(10, p=1.0, alpha=cutmix_alpha))
if mixup_transforms:
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

loader_g = torch.Generator()
loader_g.manual_seed(2023)

# Define Network
def load_model(num_classes=NUM_OUTPUTS):
    net = spiking_vgg.__dict__['spiking_vgg11_bn'](pretrained=False, spiking_neuron=neuron.LIFNode,
                                                    surrogate_function=surrogate.ATan(), 
                                                    detach_reset=True, num_classes=num_classes)
    functional.set_step_mode(net, step_mode='m')
    return net

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_train_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        # sampler=torch.utils.data.RandomSampler(dataset, generator=loader_g),
        num_workers=0,
        pin_memory=True,
        # collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        drop_last=True
    )

def load_val_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        # sampler=torch.utils.data.SequentialSampler(dataset),
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True
    )
def prep_FL_data():
    # Calculate the size of each partition
    total_size = len(cifar10_train)
    partition_size = total_size // NUM_CLIENTS
    indices = list(range(total_size))

    np.random.shuffle(indices)

    subset_id_lists = [indices[i * partition_size:(i + 1) * partition_size] for i in range(NUM_CLIENTS)]

    subsets = [Subset(cifar10_train, subset_id_list)
                        for subset_id_list in subset_id_lists]

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

if __name__ == "__main__":
    dump_FL_data()