from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os


class AbaloneDataSet(Dataset):
    def __init__(self, dataset, number):
        super(AbaloneDataSet, self).__init__()
        self.feature = dataset[:, :dataset.shape[1] - 1]
        self.target = dataset[:, -1]
        self.number = number

    def __getitem__(self, index):
        return self.feature[index], self.target[index], self.number[index]

    def __len__(self):
        return len(self.feature)


class AutoMpgDataSet(Dataset):
    def __init__(self, dataset, number):
        super(AutoMpgDataSet, self).__init__()
        self.feature = dataset[:, 1:]
        self.target = dataset[:, 0]
        self.number = number

    def __getitem__(self, index):
        return self.feature[index], self.target[index], self.number[index]

    def __len__(self):
        return len(self.feature)


class HousingDataSet(Dataset):
    def __init__(self, dataset, number):
        super(HousingDataSet, self).__init__()
        self.feature = dataset[:, :dataset.shape[1] - 1]
        self.target = dataset[:, -1]
        self.number = number

    def __getitem__(self, index):
        return self.feature[index], self.target[index], self.number[index]

    def __len__(self):
        return len(self.feature)


class AirfoilDataSet(Dataset):
    def __init__(self, dataset, number):
        super(AirfoilDataSet, self).__init__()
        self.feature = dataset[:, :dataset.shape[1] - 1]
        self.target = dataset[:, - 1]
        self.number = number

    def __getitem__(self, index):
        return self.feature[index], self.target[index], self.number[index]

    def __len__(self):
        return len(self.feature)


class ConcreteDataSet(Dataset):
    def __init__(self, dataset, number):
        super(ConcreteDataSet, self).__init__()
        self.feature = dataset[:, :dataset.shape[1] - 1]
        self.target = dataset[:, -1]
        self.number = number

    def __getitem__(self, index):
        return self.feature[index], self.target[index], self.number[index]

    def __len__(self):
        return len(self.feature)


class AgeDB(Dataset):
    def __init__(self, df, data_dir, img_size, number, split='train'):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.label = np.asarray(df['age']).astype('float32')
        self.number = number

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        return img, self.label[index], self.number[index]

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform


class BreastPathQDataset(Dataset):
    def __init__(self, df, data_dir, img_size, number, split="train"):
        file_ext = '.tif'
        self.img_size = img_size
        self.data_dir = data_dir
        self.split = split
        self.number = number
        self.resize_to = (img_size, img_size)

        self.df = df.reset_index(drop=True)

        self.img_file_names = [
            self.data_dir + f"/breastpathq/breast/{self.df['slide'][i]}_{self.df['rid'][i]}{file_ext}"
            for i in range(self.df.shape[0])]

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, idx):

        x = Image.open(self.img_file_names[idx]).convert('RGB')

        y = np.array([self.df['label'][int(idx)]], dtype=np.float32)
        y = y

        transform = self.get_transform()

        x = transform(x)

        return x, y, self.number[idx]

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform
