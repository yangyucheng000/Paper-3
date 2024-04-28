import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
class CNN_Relu(nn.Cell):
    def __init__(self):
        super(CNN_Relu, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Dense(32 * 4 * 4, 32),
            nn.ReLU(),
            nn.Dense(32, 10)
        )

    def construct(self, x):
        x = self.conv(x)
        return x

class CNN_Tanh(nn.Cell):
    def __init__(self):
        super(CNN_Tanh, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Dense(32 * 4 * 4, 32),
            nn.Tanh(),
            nn.Dense(32, 10)
        )

    def construct(self, x):
        x = self.conv(x)
        return x


def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats
    bn_mean = Tensor(bn_mean)
    bn_var = Tensor(bn_var)
    sqrt = ops.Sqrt()
    x = (x - bn_mean.view(1, -1, 1, 1)) / sqrt(bn_var.view(1, -1, 1, 1) + 1e-5)
    x = Tensor(x)
    x *= (bn_var.view(1, -1, 1, 1) != 0).float()
    return x

class MNIST_CNN_Relu(nn.Cell):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN_Relu, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, pad_mode='pad', padding=pad)

                layers.append(conv2d)
                layers.append(nn.ReLU())
                c = filters

        self.features = nn.SequentialCell(*layers)

        hidden = 32
        self.classifier = nn.SequentialCell(
            nn.Dense(c * 4 * 4, hidden),
            nn.ReLU(),
            nn.Dense(hidden, 10)
        )

    def construct(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

class MNIST_CNN_Tanh(nn.Cell):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, pad_mode='pad', padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.SequentialCell(*layers)

        hidden = 32
        self.classifier = nn.SequentialCell(
            nn.Dense(c * 4 * 4, hidden),
            nn.Tanh(),
            nn.Dense(hidden, 10)
        )

    def construct(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

class CIFAR10_CNN_Relu(nn.Cell):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN_Relu, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.ReLU

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, pad_mode='same')

                layers.append(conv2d)
                layers.append(act())
                c = v

        self.features = nn.SequentialCell(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.SequentialCell(
                nn.Dense(c * 4 * 4, hidden),
                act(),
                nn.Dense(hidden, 10)
            )
        else:
            self.classifier = nn.Dense(c * 4 * 4, 10)

    def construct(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

class CIFAR10_CNN_Tanh(nn.Cell):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, pad_mode='same')

                layers.append(conv2d)
                layers.append(act())
                c = v

        self.features = nn.SequentialCell(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.SequentialCell(
                nn.Dense(c * 4 * 4, hidden),
                act(),
                nn.Dense(hidden, 10)
            )
        else:
            self.classifier = nn.Dense(c * 4 * 4, 10)

    def construct(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
