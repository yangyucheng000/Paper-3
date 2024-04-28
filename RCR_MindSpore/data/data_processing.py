import pandas as pd
import numpy as np
import random
import torch


def abalone_data_processing(path="./data/dataset/abalone.data"):
    data = pd.read_table(path, sep=',')
    data.Rings = data.Rings.astype(float)
    dict_sex = {"M": 0, "F": 1, "I": 2}
    data["Sex"] = data["Sex"].map(dict_sex)
    data["Sex"] = data["Sex"].astype('category')
    data = data.values
    data = np.append(np.eye(3)[data[:, 0].astype(int)], data[:, 1:], axis=1)

    data[:, 3:data.shape[1] - 1] = (data[:, 3:data.shape[1] - 1] - np.mean(data[:, 3:data.shape[1] - 1],
                                                                           axis=0)) / np.std(
        data[:, 3:data.shape[1] - 1], axis=0)

    return np.float32(data), torch.tensor(range(data.shape[0])).type(torch.long)


def auto_mpg_data_processing(path="./data/dataset/auto-mpg.data"):
    data = pd.read_table(path, sep='\t')
    data = data.drop(["car name", "Unnamed: 9"], axis=1)
    data = data.dropna()
    dict_cylinders = {3: 0, 4: 1, 5: 2, 6: 3, 8: 4}
    dict_origin = {1: 0, 2: 1, 3: 2}
    data["cylinders"] = data["cylinders"].map(dict_cylinders)
    data["origin"] = data["origin"].map(dict_origin)
    data = data.values
    data = np.append(np.delete(data, 1, axis=1), np.eye(len(dict_cylinders))[data[:, 1].astype(int)], axis=1)
    data = np.append(np.delete(data, 6, axis=1), np.eye(len(dict_origin))[data[:, 6].astype(int)], axis=1)

    data[:, 1:6] = (data[:, 1:6] - np.mean(data[:, 1:6], axis=0)) / np.std(data[:, 1:6], axis=0)

    return np.float32(data), torch.tensor(range(data.shape[0])).type(torch.long)


def housing_data_processing(path="./data/dataset/housing.data"):
    data = pd.read_table(path, sep='\t')
    data = data.values
    data = np.append(np.eye(2)[data[:, 3].astype(int)], np.delete(data, 3, axis=1), axis=1)

    data[:, 2:14] = (data[:, 2:14] - np.mean(data[:, 2:14], axis=0)) / np.std(data[:, 2:14], axis=0)

    return np.float32(data), torch.tensor(range(data.shape[0])).type(torch.long)


def airfoil_data_processing(path="./data/dataset/airfoil.data"):
    data = pd.read_table(path, sep='\t')
    data = data.values

    data[:, :5] = (data[:, :5] - np.mean(data[:, :5], axis=0)) / np.std(data[:, :5], axis=0)

    return np.float32(data), torch.tensor(range(data.shape[0])).type(torch.long)


def concrete_data_processing(path="./data/dataset/concrete.data"):
    data = pd.read_table(path, sep=',')
    data = data.values

    data[:, :8] = (data[:, :8] - np.mean(data[:, :8], axis=0)) / np.std(data[:, :8], axis=0)

    return np.float32(data), torch.tensor(range(data.shape[0])).type(torch.long)


def AgeDB_data_processing(path="./data/dataset/agedb.csv"):
    df = pd.read_csv(path)
    label = np.asarray(df['age']).astype('float32')

    "1-101"

    return df, torch.tensor(range(label.shape[0])).type(torch.long)


def cut_list(l, ratio):
    random.shuffle(l)
    length = len(l)
    idx = random.sample(range(length), int(length * ratio))
    l1 = []
    l2 = []
    index = [False for _ in range(length)]
    for i in idx:
        index[i] = True
    for i, j in enumerate(index):
        if j:
            l1.append(l[i])
        else:
            l2.append(l[i])
    return l1, l2


def slicing(df, ratio):
    df = df.reset_index(drop=True)
    l = [i for i in range(len(df))]
    _, l2 = cut_list(l, ratio)
    for i in l2:
        df["split"][i] = "-"
    return df.loc[df['split'] == "train"]
