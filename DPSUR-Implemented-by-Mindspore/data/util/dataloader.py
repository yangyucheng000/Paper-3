import numpy as np
from mindspore.dataset import GeneratorDataset

class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def generator(self):
        for i in range(self.iterations):
            indices = np.where(np.random.rand(len(self.dataset)) < (self.minibatch_size / len(self.dataset)))[0]
            if indices.size > 0:
                yield indices

    def get_dataset(self):
        indices = self.generator()
        indices = next(indices)
        pseudo_batch_size = indices.size
        dataloader = GeneratorDataset(self.dataset, sampler=indices , column_names=['x','y'])
        dataloader = dataloader.batch(pseudo_batch_size)
        return dataloader

def load_train_data_minibatch(data = None, minibatch_size = 256, iterations=None):
    if iterations is None:
        iterations = 1
    if data is None:
        data = np.load('data\\mnist\\mnist.npz')
        x_train, y_train = data['x_train'], data['y_train']
    else:
        x_train, y_train = data
    sampler = IIDBatchSampler(x_train, minibatch_size, iterations)
    def generator_func():
        sampled_indicies = sampler.generator()
        sampled_indicies = next(sampled_indicies)
        for i in sampled_indicies:
            yield x_train[i], y_train[i]

    dataset = GeneratorDataset(source = generator_func, column_names = ['image', 'label'])
    return dataset

def load_test_data(is_dpsur, data = None, batch_size=None):
    if batch_size == None:
        batch_size = 256
    if data is None:
        data = np.load('data\\mnist\\mnist.npz')
        x_test, y_test = data['x_test'], data['y_test']
    else:
        x_test, y_test = data
    def generator_func():
        for i in range(0,len(x_test)):
            if is_dpsur:
                yield x_test[i], y_test[i]
            else:
                yield np.expand_dims(x_test[i], axis=0), np.array(y_test[i], dtype=np.int32)

    dataloader = GeneratorDataset(source = generator_func, shuffle=True, column_names=['x', 'y'])
    if not is_dpsur:
        dataloader = dataloader.batch(batch_size, drop_remainder=False)
    return dataloader