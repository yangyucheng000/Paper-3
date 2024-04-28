import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from  mindspore.dataset import SubsetRandomSampler

def getLocalData(dir, name):
    if name == 'mnist':
        train_dataset = ds.MnistDataset(dir+"/train", "train")
        train_dataset = train_dataset.map(vision.ToTensor(), ["image"])
    elif name == 'cifar':
        transform_train = ds.transforms.Compose([
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = ds.Cifar10Dataset(dir+"/train", "train")
        train_dataset = train_dataset.map(transform_train, ["image"])
    return train_dataset

def getSampleData(dir, name, indices):
    if name == 'mnist':
        # mindspore
        sampler = SubsetRandomSampler(indices)
        sample_dataset = ds.MnistDataset(dir+"/train", "train", sampler=sampler)
        sample_dataset = sample_dataset.map(vision.ToTensor(), ["image"])
    elif name == 'cifar':
        sampler = SubsetRandomSampler(indices)
        sample_dataset = ds.Cifar10Dataset(dir+"/train", "train", sampler=sampler)
        transform_train = ds.transforms.Compose([
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),
            vision.ToTensor(),
            vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        sample_dataset = sample_dataset.map(transform_train, ["image"])
    return sample_dataset

def getTestData(dir, name):
    if name == 'mnist':
        test_dataset = ds.MnistDataset(dir+"/test", "test")
        test_dataset = test_dataset.map(vision.ToTensor(), ["image"])
    elif name == 'cifar':
        transform_test = ds.transforms.Compose([
            vision.ToTensor(),
            vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_dataset = ds.Cifar10Dataset(dir+"/test", "test")
        test_dataset = test_dataset.map(transform_test, ["image"])
    return test_dataset


