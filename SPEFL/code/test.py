from getData import getLocalData, getTestData
import mindspore
import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset, Cifar10Dataset

if __name__ == "__main__":
    var = 100
    print(f'示例：{var} .')