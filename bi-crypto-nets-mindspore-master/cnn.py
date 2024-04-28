import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class Quad(nn.Cell):
    def construct(self, x):
        return x * x

def replace_layers(model, old, new, is_skip=False):
    for name, submodule in model.cells_and_names():
        if isinstance(submodule, nn.CellList) or isinstance(submodule, nn.SequentialCell):
            replace_layers(submodule, old, new, is_skip)
        if isinstance(submodule, old):
            setattr(model, name, new)

def replace_all_to_quad(model):
    replace_layers(model,nn.ReLU,Quad(),True)

def replace_to_quad(model):
    replace_layers(model,nn.ReLU,Quad())

class ConvBlock(nn.Cell):
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, padding=1, rate=0.4, drop=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, filters, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=1-rate) if drop else nn.Identity()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        return self.dropout(x) if self.dropout else x

class CifarCNN_Multi(nn.Cell):
    def __init__(self, n=10, name=None):
        super(CifarCNN_Multi, self).__init__()
        self.net_name = self.__class__.__name__ if name is None else name

        self.conv_big_c1=ConvBlock(3,32,rate=0.3)
        self.conv_big_c2=ConvBlock(32,32,drop=False)
        self.conv_big_p1=nn.AvgPool2d(kernel_size=2, stride=2) #16
        self.conv_big_c3=ConvBlock(32,64,rate=0.2)
        self.conv_big_c4=ConvBlock(64,64,drop=False)
        self.conv_big_p2=nn.AvgPool2d(kernel_size=2, stride=2) #8
        self.conv_big_c5=ConvBlock(64,128,rate=0.2)
        self.conv_big_c6=ConvBlock(128,128,rate=0.2)
        self.conv_big_p3=nn.AvgPool2d(kernel_size=2, stride=2) #4

        self.conv_small_c1=ConvBlock(3,32,rate=0.3)
        self.conv_small_c2=ConvBlock(32,32,drop=False)
        self.conv_small_p1=nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_small_c3=ConvBlock(32,64,rate=0.2)
        self.conv_small_c4=ConvBlock(64,64,drop=False)
        self.conv_small_p2=nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_small_c5=ConvBlock(64,128,rate=0.2)
        self.conv_small_c6=ConvBlock(128,128,rate=0.2)
        self.conv_small_p3=nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.pad=nn.ZeroPad2d(1)

        self.dense1_1=nn.Dense(4*4*128+2*2*128,64)
        self.dense1_2=nn.Dense(4*4*128,64)
        self.act1_1=nn.ReLU()
        self.act1_2=nn.ReLU()
        self.bn1_1=nn.BatchNorm1d(64)
        self.bn1_2=nn.BatchNorm1d(64)
        self.drop1_1=nn.Dropout(0.2)
        self.drop1_2=nn.Dropout(0.2)

        self.dense2=nn.Dense(128,32)
        self.act2=nn.ReLU()
        self.bn2=nn.BatchNorm1d(32)
        self.drop2=nn.Dropout(0.2)

        self.dense3=nn.Dense(32,n)
        

    def construct(self, inputs, KD=False):
        in1, in2 = inputs

        net1=self.conv_small_c1(in1)
        net1=self.conv_small_c2(net1)
        net1=self.conv_small_p1(net1)
        net1=self.conv_small_c3(net1)
        net1=self.conv_small_c4(net1)
        net1=self.conv_small_p2(net1)
        net1=self.conv_small_c5(net1)
        net1=self.conv_small_c6(net1)
        net1=self.conv_small_p3(net1)

        net2=self.conv_big_c1(in2)
        net2=self.conv_big_c2(net2)
        net2=self.conv_big_p1(net2)
        net2=self.conv_big_c3(net2)
        net2=self.conv_big_c4(net2)
        net2=self.conv_big_p2(net2)
        net2=self.conv_big_c5(net2)
        net2=self.conv_big_c6(net2)
        net2=self.conv_big_p3(net2)

        net1_ = self.pad(net1)
        net_ = net1_+net2

        net1 = ops.reshape(net1, (net1.shape[0], -1))
        net2 = ops.reshape(net2, (net2.shape[0], -1))
        net1_1=ops.concat([net1,net2],dim=-1)
        net1_1=self.dense1_1(net1_1)
        net1_1=self.act1_1(net1_1)
        net1_1=self.bn1_1(net1_1)
        net1_1=self.drop1_1(net1_1)

        net2_1=self.dense1_2(net2)
        net2_1=self.act1_2(net2_1)
        net2_1=self.bn1_2(net2_1)
        net2_1=self.drop1_2(net2_1)

        net=ops.concat([net1_1,net2_1],dim=-1)
        net=self.dense2(net)
        net=self.act2(net)
        net=self.bn2(net)
        net=self.drop2(net)

        net=self.dense3(net)

        if KD:
            return net_,net
        else:
            return net

class CifarCNN(nn.Cell):
    def __init__(self, n=10, name=None):
        super(CifarCNN, self).__init__()
        self.net_name = self.__class__.__name__ if name is None else name

        self.conv_big_c1=ConvBlock(3,32,rate=0.3)
        self.conv_c2=ConvBlock(32,32,drop=False)
        self.conv_p1=nn.AvgPool2d(kernel_size=2, stride=2) #16
        self.conv_c3=ConvBlock(32,64,rate=0.2)
        self.conv_c4=ConvBlock(64,64,drop=False)
        self.conv_p2=nn.AvgPool2d(kernel_size=2, stride=2) #8
        self.conv_c5=ConvBlock(64,128,rate=0.2)
        self.conv_c6=ConvBlock(128,128,rate=0.2)
        self.conv_p3=nn.AvgPool2d(kernel_size=2, stride=2) #4

        final=4*4*128
        self.drop1=nn.Dropout(0.1)
        self.dense1=nn.Dense(final,256)
        self.act2=nn.ReLU()
        self.bn2=nn.BatchNorm1d(256)
        self.drop2=nn.Dropout(0.1)

        self.dense2=nn.Dense(256,n)
        
    def construct(self, inputs, KD=False):
        net2=self.conv_big_c1(inputs)
        net2=self.conv_c2(net2)
        net2=self.conv_p1(net2)
        net2=self.conv_c3(net2)
        net2=self.conv_c4(net2)
        net2=self.conv_p2(net2)
        net2=self.conv_c5(net2)
        net2=self.conv_c6(net2)
        net2_=self.conv_p3(net2)
        net2=net2_

        net2 = ops.reshape(net2, (net2.shape[0], -1))

        net2_1=self.drop1(net2)
        net2_1=self.dense1(net2)
        net2_1=self.act2(net2_1)
        net2_1=self.bn2(net2_1)
        net2_1=self.drop2(net2_1)
        net=self.dense2(net2_1)

        if KD:
            return net2_,net
        else:
            return net

class CifarCNN_Multi_v1(CifarCNN_Multi):
    def construct(self, inputs, KD=False):
        in1, in2 = inputs

        net2_1=self.conv_big_c1(in2)
        net2_2=self.conv_big_c2(net2_1)
        net2=self.conv_big_p1(net2_2)
        net2_3=self.conv_big_c3(net2)
        net2_4=self.conv_big_c4(net2_3)
        net2=self.conv_big_p2(net2_4)
        net2_5=self.conv_big_c5(net2)
        net2_6=self.conv_big_c6(net2_5)
        net2=self.conv_big_p3(net2_6)

        net2_1=net2_1[...,8:24,8:24]
        net2_2=net2_2[...,8:24,8:24]
        net2_3=net2_3[...,4:12,4:12]
        net2_4=net2_4[...,4:12,4:12]
        net2_5=net2_5[...,2:6,2:6]
        net2_6=net2_6[...,2:6,2:6]

        net2_1_=net2_1
        net2_2_=net2_2
        net2_3_=net2_3
        net2_4_=net2_4
        net2_5_=net2_5
        net2_6_=net2_6


        # small
        net1=self.conv_small_c1(in1)
        in1_1=net1+net2_1_
        net1=self.conv_small_c2(in1_1)
        in1_2=net1+net2_2_
        net1_1=self.conv_small_p1(in1_2)
        net1=self.conv_small_c3(net1_1)
        in1_3=net1+net2_3_
        net1=self.conv_small_c4(in1_3)
        in1_4=net1+net2_4_
        net1_2=self.conv_small_p2(in1_4)
        net1=self.conv_small_c5(net1_2)
        in1_5=net1+net2_5_
        net1=self.conv_small_c6(in1_5)
        in1_6=net1+net2_6_
        net1_3=self.conv_small_p3(in1_6)

        net1_ = self.pad(net1_3)
        net_ = net1_+net2

        net1 = ops.reshape(net1_3, (net1_3.shape[0], -1))
        net2 = ops.reshape(net2, (net2.shape[0], -1))
        net1_1=ops.concat([net1,net2],dim=-1)
        net1_1=self.dense1_1(net1_1)
        net1_1=self.act1_1(net1_1)
        net1_1=self.bn1_1(net1_1)
        net1_1=self.drop1_1(net1_1)

        net2_1=self.dense1_2(net2)
        net2_1=self.act1_2(net2_1)
        net2_1=self.bn1_2(net2_1)
        net2_1=self.drop1_2(net2_1)

        net=ops.concat([net1_1,net2_1],dim=-1)
        net=self.dense2(net)
        net=self.act2(net)
        net=self.bn2(net)
        net=self.drop2(net)

        net=self.dense3(net)

        if KD:
            return net_,net
        else:
            return net


class CifarCNN_Multi_v2(CifarCNN_Multi):
    def __init__(self, n=10,name=None):
        super().__init__(n,name)

        self.conv_dense1=nn.Dense(32,32)
        self.conv_dense2=nn.Dense(32,32)
        self.conv_dense3=nn.Dense(64,64)
        self.conv_dense4=nn.Dense(64,64)
        self.conv_dense5=nn.Dense(128,128)
        self.conv_dense6=nn.Dense(128,128)

    def forward(self, inputs,KD=False):
        in1,in2=inputs

        net2_1=self.conv_big_c1(in2)
        net2_2=self.conv_big_c2(net2_1)
        net2=self.conv_big_p1(net2_2)
        net2_3=self.conv_big_c3(net2)
        net2_4=self.conv_big_c4(net2_3)
        net2=self.conv_big_p2(net2_4)
        net2_5=self.conv_big_c5(net2)
        net2_6=self.conv_big_c6(net2_5)
        net2=self.conv_big_p3(net2_6)

        net2_1=net2_1[...,8:24,8:24]
        net2_2=net2_2[...,8:24,8:24]
        net2_3=net2_3[...,4:12,4:12]
        net2_4=net2_4[...,4:12,4:12]
        net2_5=net2_5[...,2:6,2:6]
        net2_6=net2_6[...,2:6,2:6]


        net2_1_=self.conv_dense1(net2_1.transpose(1,3)).transpose(1,3)
        net2_2_=self.conv_dense2(net2_2.transpose(1,3)).transpose(1,3)
        net2_3_=self.conv_dense3(net2_3.transpose(1,3)).transpose(1,3)
        net2_4_=self.conv_dense4(net2_4.transpose(1,3)).transpose(1,3)
        net2_5_=self.conv_dense5(net2_5.transpose(1,3)).transpose(1,3)
        net2_6_=self.conv_dense6(net2_6.transpose(1,3)).transpose(1,3)


        # small
        net1=self.conv_small_c1(in1)
        in1_1=net1+net2_1_
        net1=self.conv_small_c2(in1_1)
        in1_2=net1+net2_2_
        net1_1=self.conv_small_p1(in1_2)
        net1=self.conv_small_c3(net1_1)
        in1_3=net1+net2_3_
        net1=self.conv_small_c4(in1_3)
        in1_4=net1+net2_4_
        net1_2=self.conv_small_p2(in1_4)
        net1=self.conv_small_c5(net1_2)
        in1_5=net1+net2_5_
        net1=self.conv_small_c6(in1_5)
        in1_6=net1+net2_6_
        net1_3=self.conv_small_p3(in1_6)

        net1_ = self.pad(net1_3)
        net_ = net1_+net2

        net1 = ops.reshape(net1_3, (net1_3.shape[0], -1))
        net2 = ops.reshape(net2, (net2.shape[0], -1))
        net1_1=ops.concat([net1,net2],dim=-1)
        net1_1=self.dense1_1(net1_1)
        net1_1=self.act1_1(net1_1)
        net1_1=self.bn1_1(net1_1)
        net1_1=self.drop1_1(net1_1)

        net2_1=self.dense1_2(net2)
        net2_1=self.act1_2(net2_1)
        net2_1=self.bn1_2(net2_1)
        net2_1=self.drop1_2(net2_1)

        net=ops.concat([net1_1,net2_1],dim=-1)
        net=self.dense2(net)
        net=self.act2(net)
        net=self.bn2(net)
        net=self.drop2(net)

        net=self.dense3(net)

        if KD:
            return net_,net
        else:
            return net
        
class CifarCNN_Multi_v3(CifarCNN_Multi):
    def __init__(self, n=10,name=None):
        super().__init__(n,name)

        self.conv_dense1=nn.Conv2d(32,32,3,stride=2,padding=1)
        self.conv_dense2=nn.Conv2d(32,32,3,stride=2,padding=1)
        self.conv_dense3=nn.Conv2d(64,64,3,stride=2,padding=1)
        self.conv_dense4=nn.Conv2d(64,64,3,stride=2,padding=1)
        self.conv_dense5=nn.Conv2d(128,128,3,stride=2,padding=1)
        self.conv_dense6=nn.Conv2d(128,128,3,stride=2,padding=1)

    def forward(self, inputs,KD=False):
        in1,in2=inputs

        net2_1=self.conv_big_c1(in2)
        net2_2=self.conv_big_c2(net2_1)
        net2=self.conv_big_p1(net2_2)
        net2_3=self.conv_big_c3(net2)
        net2_4=self.conv_big_c4(net2_3)
        net2=self.conv_big_p2(net2_4)
        net2_5=self.conv_big_c5(net2)
        net2_6=self.conv_big_c6(net2_5)
        net2=self.conv_big_p3(net2_6)

        net2_1_=net2_1
        net2_2_=net2_2
        net2_3_=net2_3
        net2_4_=net2_4
        net2_5_=net2_5
        net2_6_=net2_6

        net2_1_=self.conv_dense1(net2_1)
        net2_2_=self.conv_dense2(net2_2)
        net2_3_=self.conv_dense3(net2_3)
        net2_4_=self.conv_dense4(net2_4)
        net2_5_=self.conv_dense5(net2_5)
        net2_6_=self.conv_dense6(net2_6)


        # small
        net1=self.conv_small_c1(in1)
        in1_1=net1+net2_1_
        net1=self.conv_small_c2(in1_1)
        in1_2=net1+net2_2_
        net1_1=self.conv_small_p1(in1_2)
        net1=self.conv_small_c3(net1_1)
        in1_3=net1+net2_3_
        net1=self.conv_small_c4(in1_3)
        in1_4=net1+net2_4_
        net1_2=self.conv_small_p2(in1_4)
        net1=self.conv_small_c5(net1_2)
        in1_5=net1+net2_5_
        net1=self.conv_small_c6(in1_5)
        in1_6=net1+net2_6_
        net1_3=self.conv_small_p3(in1_6)

        net1_ = self.pad(net1_3)
        net_ = net1_+net2

        net1 = ops.reshape(net1_3, (net1_3.shape[0], -1))
        net2 = ops.reshape(net2, (net2.shape[0], -1))
        net1_1=ops.concat([net1,net2],dim=-1)
        net1_1=self.dense1_1(net1_1)
        net1_1=self.act1_1(net1_1)
        net1_1=self.bn1_1(net1_1)
        net1_1=self.drop1_1(net1_1)

        net2_1=self.dense1_2(net2)
        net2_1=self.act1_2(net2_1)
        net2_1=self.bn1_2(net2_1)
        net2_1=self.drop1_2(net2_1)

        net=ops.concat([net1_1,net2_1],dim=-1)
        net=self.dense2(net)
        net=self.act2(net)
        net=self.bn2(net)
        net=self.drop2(net)

        net=self.dense3(net)


        if KD:
            return net_,net
        else:
            return net