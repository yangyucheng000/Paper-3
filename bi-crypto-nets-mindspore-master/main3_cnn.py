import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.dataset import vision
from mindspore.dataset.transforms import Compose
import numpy as np
import mindspore.dataset as ds

from torch_trainer import KD_MindSpore_Trainer_2, KD_MindSpore_Trainer,seed_mindspore
from cnn import *

n_epoch=250
batch_size=256

from mindspore import load_checkpoint, load_param_into_net

def load_mindspore_model(net, checkpoint_path):
    """ Load the trained model from a checkpoint file. """
    param_dict = load_checkpoint(checkpoint_path)
    load_param_into_net(net, param_dict)
    return net

class SplitDataset_KD:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x1 = x[:, 8:24, 8:24]
        x2 = np.multiply(x, self.mask)
        return (x1, x2), x, y

    def __len__(self):
        return len(self.dataset)

# Define data transforms for MindSpore
mean_ = [0.4914, 0.4822, 0.4465]  # Replace with the calculated mean values
std_ = [0.2023, 0.1994, 0.2010]  # Replace with the calculated std values

transform_train = Compose([
    vision.RandomCrop(32, padding=4),
    vision.RandomRotation(15),
    vision.RandomHorizontalFlip(),
    vision.ToTensor(),
    vision.Normalize(mean=mean_, std=std_),
])

transform_test = Compose([
    vision.ToTensor(),
    vision.Normalize(mean=mean_, std=std_),
])

# Load and transform datasets
trainset = ds.Cifar10Dataset("./data", train=True, transform=transform_train)
testset = ds.Cifar10Dataset("./data", train=False, transform=transform_test)

def run_all_kd(net, trainset, testset, clip_norm=0):
    # Create data loaders
    trainloader = ds.GeneratorDataset(SplitDataset_KD(trainset), ["data", "label"])
    testloader = ds.GeneratorDataset(SplitDataset_KD(testset), ["data", "label"])

    # Load teacher model
    net_t = CifarCNN_Multi_v2(name='cnn_v2_fkd')
    load_mindspore_model(net_t, 'checkpoint/CifarCNN.ckpt')  # Adapt load function for MindSpore

    # Loss and optimizer
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    d_loss = nn.MSELoss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-3, weight_decay=1e-4)
    scheduler = nn.CosineAnnealingLR(optimizer, T_max=n_epoch)

    # Trainer
    trainer = KD_MindSpore_Trainer(net, net_t, criterion, d_loss, optimizer, scheduler, clip_norm=clip_norm)
    trainer.fit(trainloader, testloader, n_epoch, 10)

    # Adjust learning rate and continue training
    optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-4, weight_decay=1e-4)
    scheduler = nn.CosineAnnealingLR(optimizer, T_max=n_epoch)
    trainer = KD_MindSpore_Trainer_2(net, net_t, criterion, d_loss, optimizer, scheduler, clip_norm=clip_norm)

    return trainer.fit(trainloader, testloader, n_epoch, 20)

# Seed and model instantiation
seed_mindspore(42)  # Adapt seed function for MindSpore
net = CifarCNN_Multi_v2(name='cnn_v2_fkd')
replace_to_quad(net)  # Make sure this function is adapted for MindSpore

# Running training with KD
res = run_all_kd(net, trainset, testset, clip_norm=5)
print(res)

