
import math
from mindspore import nn
from mindspore import ops
sample_structure = [20, 30, 10]


class RwRModel(nn.Cell):
    def __init__(self, input_dim, out_dim=1, activate=None):
        super(RwRModel, self).__init__()
        self.h = MlpModel(input_dim, out_dim)
        if activate == "sigmoid":
            self.r = MlpModel(input_dim, activate="sigmoid")
        else:
            self.r = MlpModel(input_dim)

    def forward(self, x):
        return self.h(x), self.r(x)


class RwRLinear(nn.Cell):
    def __init__(self, input_dim, out_dim=1, activate=None):
        super(RwRLinear, self).__init__()
        self.h = LinearModel(input_dim, out_dim)
        if activate == "sigmoid":
            self.r = LinearModel(input_dim, activate="sigmoid")
        else:
            self.r = LinearModel(input_dim)

    def forward(self, x):
        return self.h(x), self.r(x)


class Single_linear(nn.Cell):
    def __init__(self, input_dim, activate=None):
        super(Single_linear, self).__init__()
        layers = [nn.Dense(input_dim, 1)]
        if activate == "sigmoid":
            layers.append(nn.Sigmoid())
        self.linear = nn.SequentialCell(*layers)

    def forward(self, x):
        return self.linear(x)


class MlpModel(nn.Cell):
    def __init__(self, input_dim, out_dim=1, activate=None, num_blocks=sample_structure):
        super(MlpModel, self).__init__()
        layers = [nn.Dense(input_dim, num_blocks[0])]
        for i in range(1, len(num_blocks)):
            layers.append(nn.Dense(num_blocks[i - 1], num_blocks[i]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dense(num_blocks[-1], out_dim))
        if activate == "sigmoid":
            layers.append(nn.Sigmoid())
        self.linear = nn.SequentialCell(*layers)

    def forward(self, x):
        return self.linear(x)


class LinearModel(nn.Cell):
    def __init__(self, input_dim, out_dim=1, activate=None, num_blocks=sample_structure):
        super(LinearModel, self).__init__()
        layers = [nn.Dense(input_dim, num_blocks[0])]
        for i in range(1, len(num_blocks)):
            layers.append(nn.Dense(num_blocks[i - 1], num_blocks[i]))
        layers.append(nn.Dense(num_blocks[-1], out_dim))
        if activate == "sigmoid":
            layers.append(nn.Sigmoid())
        self.linear = nn.SequentialCell(*layers)

    def forward(self, x):
        return self.linear(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with nn.SequentialCell"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad',nn.SequentialCell=1, bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode='pad', nn.SequentialCell=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class RwR_ResNet(nn.Cell):

    def __init__(self, block, layers, out_dim=1, dropout=False):
        super(RwR_ResNet, self).__init__()
        self.h = ResNet(block, layers, out_dim=out_dim, dropout=dropout)
        self.r = ResNet(block, layers, out_dim=out_dim, dropout=dropout)

    def forward(self, x):
        pre = self.h(x)
        rej = self.r(x)
        return pre, rej


class ResNet(nn.Cell):

    def __init__(self, block, layers, out_dim=1, activate=None, dropout=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad',padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad',padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Dense(512 * block.expansion, out_dim)
        self.ac = nn.Sigmoid()
        self.activate = activate

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def forward(self, x, targets=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        encoding = x.view(x.size(0), -1)

        encoding_s = encoding

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)
        x = self.linear(encoding_s)
        if self.activate == "sigmoid":
            x = self.ac(x)
        return x


class RwR_one_ResNet(nn.Cell):

    def __init__(self, block, layers, out_dim=1, dropout=False):
        self.inplanes = 64
        super(RwR_one_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad',padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad',padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.h = nn.Dense(512 * block.expansion, out_dim)
        self.r = MlpModel(512 * block.expansion)

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def forward(self, x, targets=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        encoding = x.view(x.size(0), -1)

        encoding_s = encoding

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)
        pre = self.h(encoding_s)
        rej = self.r(encoding_s)
        return pre, rej