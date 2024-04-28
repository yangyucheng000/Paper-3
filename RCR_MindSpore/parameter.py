import sys
from data import *


def up_seed(rand_seed):
    """"update seed"""
    np.random.seed(rand_seed)
    random.seed(rand_seed)


MAX_INT = sys.maxsize
OPTIM_RATE = [0.001, 0.01, 0.1]

epoch = 100
optim_rate = OPTIM_RATE[0]

seed = 2023
up_seed(2023)

datasets = {
    "abalone": {"fun_data_processing": abalone_data_processing, "class_dataset": AbaloneDataSet,
                "model_structure": [20, 30, 10], "optim_rate": 0.01, "epoch": epoch, "c": 0, "slow-star": True},
    "agedb": {"fun_data_processing": AgeDB_data_processing, "class_dataset": AgeDB,
              "model_structure": [20, 30, 10], "optim_rate": 0.001, "epoch": epoch, "c": 0, "slow-star": True,
              "img_size": 224},
 "optim_rate": optim_rate, "epoch": epoch,
               "c": 0, "slow-star": True, "img_size": 224},
    "num_work": 0, "pin_memory": False, "non_blocking": False, "batch_size": 256, "slow": 200, "weight_decay": 0,
    "use_drop": False, "drop_out": False, "inf": 0.0, "decrease": False, "stop_c": 0.6
}


def load_data(data_name, time=1, ratio=1):
    if data_name in ["agedb", "breast"]:
        df = pd.read_csv(
            "data/dataset/" + data_name + "_split/" + data_name + "_" + str(time) + ".csv",
            sep=",")
        train_data = df.loc[df['split'] == "train"]
        if ratio < 1:
            train_data = slicing(train_data, ratio)
        verify_data = df.loc[df['split'] == "verify"]
        test_data = df.loc[df['split'] == "test"]

        train_number = mindspore.tensor(range(len(train_data))).type(mindspore.long)
        datasets[data_name]["train_dataset"] = datasets[data_name]["class_dataset"](df=train_data,
                                                                                    data_dir="data/dataset",
                                                                                    img_size=datasets[data_name][
                                                                                        "img_size"],
                                                                                    number=train_number,
                                                                                    split='train')

        verify_number = mindspore.tensor(range(len(verify_data))).type(mindspore.long)
        datasets[data_name]["verify_dataset"] = datasets[data_name]["class_dataset"](df=verify_data,
                                                                                     data_dir="data/dataset",
                                                                                     img_size=datasets[data_name][
                                                                                         "img_size"],
                                                                                     number=verify_number,
                                                                                     split='verify')

        test_number = mindspore.tensor(range(len(test_data))).type(mindspore.long)
        datasets[data_name]["test_dataset"] = datasets[data_name]["class_dataset"](df=test_data,
                                                                                   data_dir="data/dataset",
                                                                                   img_size=datasets[data_name][
                                                                                       "img_size"], number=test_number,
                                                                                   split='test')
    else:
        df = pd.read_csv(
            "data/dataset/" + data_name + "_split/" + data_name + "_" + str(time) + ".csv",
            sep=",")
        train_data = df.loc[df['split'] == "train"]
        if ratio < 1:
            train_data = slicing(train_data, ratio)
        verify_data = df.loc[df['split'] == "verify"]
        test_data = df.loc[df['split'] == "test"]

        train_data = train_data.values
        train_data = train_data[:, :train_data.shape[1] - 1]
        train_data = np.float32(train_data)
        train_number = mindspore.tensor(range(train_data.shape[0])).type(torch.long)
        datasets[data_name]["train_dataset"] = datasets[data_name]["class_dataset"](dataset=train_data,
                                                                                    number=train_number)

        verify_data = verify_data.values
        verify_data = verify_data[:, :verify_data.shape[1] - 1]
        verify_data = np.float32(verify_data)
        verify_number = mindspore.tensor(range(verify_data.shape[0])).type(torch.long)
        datasets[data_name]["verify_dataset"] = datasets[data_name]["class_dataset"](dataset=verify_data,
                                                                                     number=verify_number)

        test_data = test_data.values
        test_data = test_data[:, :test_data.shape[1] - 1]
        test_data = np.float32(test_data)
        test_number = mindspore.tensor(range(test_data.shape[0])).type(torch.long)
        datasets[data_name]["test_dataset"] = datasets[data_name]["class_dataset"](dataset=test_data,
                                                                                   number=test_number)
