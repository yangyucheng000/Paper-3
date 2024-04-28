import numpy as np

class IIDBatchSampler:

    def __init__(self, dataset, minibatch_size, iterations):
        self.length = dataset.num_samples
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):

            indices = np.where(np.random.rand(self.length) < (self.minibatch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations

class EquallySizedAndIndependentBatchSamplerWithoutReplace:

    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):

            yield np.random.choice(self.length, self.minibatch_size, replace=False)

    def __len__(self):
        return self.iterations

class EquallySizedAndIndependentBatchSamplerWithReplace:

    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):
            yield np.random.choice(self.length, self.minibatch_size, replace=True)

    def __len__(self):
        return self.iterations

def get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations, drop_last = True):

    def minibatch_loader(dataset):
        batch_sampler = EquallySizedAndIndependentBatchSamplerWithoutReplace(dataset, minibatch_size, iterations)
        dataset_after_sampling = dataset.sampler(sampler = batch_sampler)
        return dataset_after_sampling

    def microbatch_loader(minibatch):
        microbatch = minibatch.batch(batch_size = microbatch_size, drop_last = drop_last)
        return microbatch

    return minibatch_loader, microbatch_loader

def get_data_loaders_uniform_with_replace(minibatch_size, microbatch_size, iterations, drop_last=True):

    def minibatch_loader(dataset):
        batch_sampler = EquallySizedAndIndependentBatchSamplerWithReplace(dataset, minibatch_size, iterations)
        dataset_after_sampling = dataset.sampler(sampler = batch_sampler)
        return dataset_after_sampling

    def microbatch_loader(minibatch):
        microbatch = minibatch.batch(batch_size = microbatch_size, drop_last = drop_last)
        return microbatch

    return minibatch_loader, microbatch_loader

def get_data_loaders_possion(minibatch_size, microbatch_size, iterations, drop_last=True):

    def minibatch_loader(dataset):
        batch_sampler = IIDBatchSampler(dataset, minibatch_size, iterations)
        dataset_after_sampling = dataset.sampler(sampler = batch_sampler)
        return dataset_after_sampling


    def microbatch_loader(minibatch):
        microbatch = minibatch.batch(batch_size = microbatch_size, drop_last = drop_last)
        return microbatch

    return minibatch_loader, microbatch_loader


