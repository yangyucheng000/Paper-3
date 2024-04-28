from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import dtype as mstype
import mindspore.ops.functional as F
import numpy as np
import mindspore.ops as ops
import mindspore


def create_F():
    F = np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
        [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
        [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
    return F

class AWCA(nn.Cell):
    def __init__(self, channel=31):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, has_bias=False)
        self.softmax = P.Softmax(axis=2)
        self.fc = nn.SequentialCell([
            nn.Dense(channel, 1, has_bias=False),
            nn.LeakyReLU(alpha=0.2),
            nn.Dense(1, channel, has_bias=False),
            nn.Sigmoid()
        ])
        self.unsqueeze = P.ExpandDims()
        self.matmul = P.BatchMatMul()
        self.reshape = P.Reshape()

    def construct(self, x):
        b, c, h, w = P.Shape()(x)
        input_x = x
        input_x = self.unsqueeze(self.reshape(input_x, (b, c, h * w)), 1)

        mask = self.reshape(self.conv(x), (b, 1, h * w))
        mask = self.unsqueeze(self.softmax(mask), -1)

        y = self.reshape(self.matmul(input_x, mask), (b, c))

        y = self.reshape(self.fc(y), (b, c, 1, 1))
        return F.tensor_mul(x, F.tile(y, (1, 1, h, w)))


class cycle_fusion(nn.Cell):
    def __init__(self, in_chanels, out_chanels):
        super(cycle_fusion, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_chanels, out_chanels, 3, 1, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
        ])
        self.conv2 = nn.SequentialCell([
            nn.Conv2d(in_chanels, out_chanels, 3, 1, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            AWCA(out_chanels),
        ])
        self.conv3 = nn.SequentialCell([
            nn.Conv2d(out_chanels * 2, 31, 3, 1, pad_mode='pad', padding=1),
            nn.LeakyReLU(alpha=0.2),
            AWCA(31),
        ])
        self.conv4 = nn.SequentialCell([
            nn.Conv2d(out_chanels+31,31, 3, 1, pad_mode='pad', padding=1),
        ])
        # Rest of the code omitted for brevity...
    def spectral_projection(self,R,MSI, mu,Y1,Y2):
        matmul = ops.MatMul(transpose_a=False,transpose_b=False)
        input_perm = (1, 0)
        transpose = ops.Transpose()
        R1 = transpose(R, input_perm)

        RTR= matmul(R1, R)
        x = ops.tensor_dot(MSI, R, axes=([1],[0]))
       
        input_perm = (0, 3, 1,2)
        transpose = ops.Transpose()
        x = transpose(x, input_perm)+mu*(Y1+Y2)
     
        eye = ops.Eye()
        x = ops.tensor_dot(x, ops.pinv(RTR+2*mu*eye(31,31, mindspore.float32)), axes=([1], [1]))
    
        input_perm = (0, 3, 1,2)
        transpose = ops.Transpose()
        x = transpose(x, input_perm)
        return(x)



    def construct(self, R, x1, MSI, mu):
        x2 = self.conv2(x1)

        x1 = self.conv1(x1)
        x2 = ops.Concat(1)((x2, x1))
        x2 = self.conv3(x2)
        x1 = ops.Concat(1)((x2, x1))
        x1 = self.conv4(x1)
        x = self.spectral_projection(R,MSI, mu,x1,x2)

        return x

class CNN_BP_SE5(nn.Cell):
    def __init__(self,mu):
        super(CNN_BP_SE5, self).__init__()
        ones=ops.Ones()
        self.mu = mindspore.Parameter(ones((1, 7), mindspore.float32)*1e-5)

        num = 64
        self.pro = cycle_fusion(3, num)
        self.pro1 = cycle_fusion(31, num)
        self.pro2 = cycle_fusion(31, num)
        self.pro3 = cycle_fusion(31, num)
        self.pro4 = cycle_fusion(31, num)
        self.pro5 = cycle_fusion(31, num)
        self.pro6 = cycle_fusion(31, num)

    def construct(self, R, R_inv, MSI):

        x = MSI
        x = self.pro(R, x, MSI, self.mu[0,0])

        x = self.pro1(R, x, MSI, self.mu[0,1])
        x = self.pro2(R, x, MSI, self.mu[0,2])
        x = self.pro3(R, x, MSI, self.mu[0,3])
        x = self.pro4(R, x, MSI, self.mu[0,4])
        x = self.pro5(R, x, MSI, self.mu[0,5])
        x = self.pro6(R, x, MSI, self.mu[0,6])
        return x

a = np.ones((2, 3, 64, 64))
a=a.astype(np.float32)
a = Tensor(a)
b = create_F()
b=b.astype(np.float32)
b = Tensor(b)
cnn=CNN_BP_SE5(1)
c=cnn(b,1,a)
print(c.shape,"old")
