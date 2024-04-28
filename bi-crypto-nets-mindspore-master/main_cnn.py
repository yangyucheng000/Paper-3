'''Train CIFAR10 with PyTorch.'''
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
import numpy as np
from mindspore import context
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

from torch_trainer import MindSpore_Trainer  
from cnn import CifarCNN  


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Data preparation
print('==> Preparing data..')

def create_dataset(data_path, batch_size=32, training=True):
    dataset = ds.Cifar10Dataset(data_path, num_parallel_workers=4, shuffle=training)

    # Define map operations
    if training:
        trans = [
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]
    else:
        trans = [
            vision.ToTensor(),
            vision.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ]

    dataset = dataset.map(operations=trans, input_columns="image")

    # Batch and shuffle
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if training:
        dataset = dataset.shuffle(buffer_size=1000)

    return dataset

train_data_path = "./data/train"
test_data_path = "./data/test"
trainloader = create_dataset(train_data_path, batch_size=256, training=True)
testloader = create_dataset(test_data_path, batch_size=1024, training=False)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


lr=1e-3
n_epoch=200
batch_size=128


# Model
print('==> Building model..')
net = CifarCNN()

# Loss function
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# Optimizer
optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=1e-4)

# Scheduler
scheduler = nn.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=0)

# Training
trainer = MindSpore_Trainer(net, criterion, optimizer, scheduler)
trainer.fit(trainloader, testloader, n_epoch, early_stop_patience=20)
