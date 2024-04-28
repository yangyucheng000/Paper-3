from kymatio.numpy import Scattering2D
SHAPES = {
    "CIFAR-10": (32, 32, 3),
    "cifar10_500K": (32, 32, 3),
    "FMNIST": (28, 28, 1),
    "MNIST": (28, 28, 1)
}

def batch_data_generator(scatters, targets):
    for scatter_batch, target_batch in zip(scatters, targets):
        yield scatter_batch, target_batch

def get_scatter_transform(dataset):
    shape = SHAPES[dataset]
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)

def get_scattered_dataset(loader, scattering, device, data_size):

    scatters = []
    targets = []
    num = 0
    for data in loader[0]:
        if scattering is not None:
            data = scattering(data)
        for itemData in data:
            scatters.append(itemData)
        num += 1
        if num > data_size:
            break
    for target in loader[1]:
        for itemTarget in target:
            targets.append(itemTarget)

    scatters = scatters[:data_size]
    targets = targets[:data_size]

    scattered_dataset = (scatters, targets)
    return scattered_dataset