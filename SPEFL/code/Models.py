import mindspore as ms
import mindspore.nn as ms_nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeNormal

class Mnist_2NN(ms_nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1 = ms_nn.Dense(784, 100)
        self.fc2 = ms_nn.Dense(100, 10)

    def construct(self, inputs):
        inputs = inputs.view(inputs.size()[0], -1)
        tensor = ops.relu(self.fc1(inputs))
        tensor = self.fc2(tensor)
        return tensor


class Mnist_CNN(ms_nn.Cell):
    def __init__(self):
        super().__init__()
        # 卷积层1.输入是32*32*3，计算（32-5）/ 1 + 1 = 28，那么通过conv1输出的结果是28*28*6
        # input:3 output:6, kernel:5
        self.conv1 = ms_nn.Conv2d(3,6,5, has_bias=True, pad_mode='valid')
        # 池化层， 输入时28*28*6， 窗口2*2，计算28 / 2 = 14，那么通过max_poll层输出的结果是14*14*6
        # kernel:2 stride:2
        self.pool = ms_nn.MaxPool2d(2,2)
        # 卷积层2， 输入是14*14*6，计算（14-5）/ 1 + 1 = 10，那么通过conv2输出的结果是10*10*16
        # imput:6 output:16, kernel:5
        self.conv2 = ms_nn.Conv2d(6,16,5, has_bias=True, pad_mode='valid')
        # 全连接层1
        # input：16*5*5，output：120
        self.fc1 = ms_nn.Dense(16*5*5, 120)
        # 全连接层2
        # input：120，output：84
        self.fc2 = ms_nn.Dense(120,84)
        # 全连接层3
        # input：84，output：10
        self.fc3 = ms_nn.Dense(84, 10)

    def construct(self, inputs):
        '''
        32x32x3 --> 28x28x6 -->14x14x6
        '''
        x = self.pool(ops.relu(self.conv1(inputs)))
        '''
        14x14x6 --> 10x10x16 --> 5x5x16        '''
        x = self.pool(ops.relu(self.conv2(x)))
        # 改变shape
        x = x.view(-1,16*5*5)

        x = ops.relu(self.fc1(x))
        x = ops.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x 

class LeNet(ms_nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        # 构造网络有两种方式一个是seqential还有一个是module,前者在后者中也可以使用，这里使用的是sequential方式，将网络结构按顺序添加即可
        self.conv1 = ms_nn.SequentialCell(  # input_size=(1*28*28)
            # 第一个卷积层，输入通道为1，输出通道为6，卷积核大小为5，步长为1，填充为2保证输入输出尺寸相同
            ms_nn.Conv2d(1,6,5,1,padding=2,has_bias=True,pad_mode='pad'),    # padding=2保证输入输出尺寸相同
            # 激活函数,两个网络层之间加入，引入非线性
            ms_nn.ReLU(),    # input_size=(6*28*28)
            # 池化层，大小为2步长为2
            ms_nn.MaxPool2d(kernel_size=2, stride=2)    # output_size=(6*14*14)
        )
        self.conv2 = ms_nn.SequentialCell(
            ms_nn.Conv2d(6,16,5, has_bias=True, pad_mode='valid'),
            ms_nn.ReLU(),   # input_size=(16*10*10)
            ms_nn.MaxPool2d(kernel_size=2, stride=2)    # output_size=(16*5*5)
        )
        # 全连接层，输入是16*5*5特征图，神经元数目120
        self.fc1 = ms_nn.SequentialCell(
            ms_nn.Dense(16*5*5, 120),
            ms_nn.ReLU()
        )
        # 全连接层神经元数目输入为上一层的120，输出为84
        self.fc2 = ms_nn.SequentialCell(
            ms_nn.Dense(120, 84),
            ms_nn.ReLU()
        )
        # 最后一层全连接层神经元数目10，与上一个全连接层同理
        self.fc3 = ms_nn.Dense(84, 10)

    # 定义前向传播过程，输入为x，也就是把前面定义的网络结构赋予了一个运行顺序
    def construct(self, x):
        tmp = x.shape
        x = x.view(tmp[0], 1, 28, 28)
        # x = x.view(x.size()[0], 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class HAR_LR(ms_nn.Cell):
    def __init__(self):
        super(HAR_LR,self).__init__()
        self.linear1 = ms_nn.Dense(1152, 100)
        self.bn = ms_nn.BatchNorm1d(num_features=100)
        self.linear2 = ms_nn.Dense(100, 6)
        self.sigmoid = ms_nn.Sigmoid()
 
    def construct(self, x):
        x = x.view(x.size()[0], -1)
        out = ops.relu(self.linear1(x))
        out = self.bn(out)
        out = self.linear2(out)
        return out

class LambdaLayer(ms_nn.Cell):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def construct(self, x):
        return self.lambd(x)

class BasicBlock(ms_nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = ms_nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, has_bias=False, pad_mode='pad')
        self.bn1 = ms_nn.BatchNorm2d(planes)
        self.conv2 = ms_nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, has_bias=False, pad_mode='pad')
        self.bn2 = ms_nn.BatchNorm2d(planes)

        self.shortcut = ms_nn.SequentialCell()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            ops.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = ms_nn.SequentialCell(
                    ms_nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, has_bias=False, pad_mode='valid'),
                    ms_nn.BatchNorm2d(self.expansion*planes)
                )

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = ops.relu(out)
        return out

class ResNet(ms_nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = ms_nn.Conv2d(3,16,3,1,padding=1,has_bias=False, pad_mode='pad',weight_init=HeNormal())
        self.bn1 = ms_nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = ms_nn.Dense(64, num_classes, weight_init=HeNormal())


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return ms_nn.SequentialCell(*layers)

    def construct(self, x):
        out = ops.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = ops.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])