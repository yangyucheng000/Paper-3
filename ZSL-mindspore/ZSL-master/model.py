import numpy as np
from mindspore import Parameter
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
import math
from scipy import signal
import scipy.io as sio
from numpy import *
import matplotlib.pyplot as plt
from scipy import signal
from mindspore import numpy
from mindspore import ops
import os
from einops import rearrange
import mindspore
import mindspore.common.dtype as mstype
import collections.abc
from itertools import repeat
from mindspore.ops import operations as P
def fspecial(func_name,kernel_size,sigma):
    if func_name=='gaussian':
        m=n=(kernel_size-1.)/2.
        y,x=ogrid[-m:m+1,-n:n+1]
        h=exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h

class CNN(nn.Cell):
    def __init__(self,a,b):
        super(CNN, self).__init__()
        self.unsqueeze = P.ExpandDims()
        self.conv1 = nn.CellList(
            nn.Conv2d(in_channels=a+b, out_channels=128-b, kernel_size=3, stride=1, padding=1,pad_mode="pad"),
            nn.LeakyReLU(alpha=0.2),
            )
        self.conv2 = nn.CellList(
            nn.Conv2d(in_channels=128, out_channels=128-b, kernel_size=3, pad_mode="pad",stride=1, padding=1),
            nn.LeakyReLU(alpha=0.2),
        )
        self.conv3 = nn.CellList(
            nn.Conv2d(in_channels=128, out_channels=128 - b, kernel_size=3, pad_mode="pad", stride=1, padding=1),
            nn.LeakyReLU(alpha=0.2),
        )
        self.conv4 = nn.CellList(
            nn.Conv2d(in_channels=128, out_channels=a, kernel_size=3, stride=1, padding=1,pad_mode="pad"),
        )

        basecoeff = Tensor([[-4.63495665e-03, -3.63442646e-03,  3.84904063e-18,
                     5.76678319e-03,  1.08358664e-02,  1.01980790e-02,
                    -9.31747402e-18, -1.75033181e-02, -3.17660068e-02,
                    -2.84531643e-02,  1.85181518e-17,  4.42450253e-02,
                     7.71733386e-02,  6.70554910e-02, -2.85299239e-17,
                    -1.01548683e-01, -1.78708388e-01, -1.60004642e-01,
                     3.61741232e-17,  2.87940558e-01,  6.25431459e-01,
                     8.97067600e-01,  1.00107877e+00,  8.97067600e-01,
                     6.25431459e-01,  2.87940558e-01,  3.61741232e-17,
                    -1.60004642e-01, -1.78708388e-01, -1.01548683e-01,
                    -2.85299239e-17,  6.70554910e-02,  7.71733386e-02,
                     4.42450253e-02,  1.85181518e-17, -2.84531643e-02,
                    -3.17660068e-02, -1.75033181e-02, -9.31747402e-18,
                     1.01980790e-02,  1.08358664e-02,  5.76678319e-03,
                     3.84904063e-18, -3.63442646e-03, -4.63495665e-03]])

        matmul = ops.MatMul(transpose_a=False, transpose_b=False)
        coeff = matmul(basecoeff.T, basecoeff)
        coeff = Tensor(coeff)
        coeff = self.unsqueeze(coeff, 0)
        coeff = self.unsqueeze(coeff, 0)
        self.coeff = ops.repeat_elements(coeff, a,0)
        psf=fspecial('gaussian', 7, 3)
        psf = Tensor(psf)
        psf = self.unsqueeze(psf, 0)
        psf = self.unsqueeze(psf, 0)
        self.psf = ops.repeat_elements(psf, a,0)
    def forward(self, x,y):
        def Upsample_4(coeff,inputs):
                    _,c, h, w= inputs.shape
                    outs = functional.conv_transpose2d(inputs, coeff.cuda(), bias=None, stride=4, padding=21, output_padding=1, groups=c, dilation=1)
                    return outs
        x1=Upsample_4(self.coeff,x)
#
        x2 =  ops.Concat(1)(x1,y)
        x2 =  ops.Concat(1)(self.conv1(x2),y)
        x2 =  ops.Concat(1)(self.conv2(x2),y)
        x2 =  ops.Concat(1)(self.conv3(x2),y)
        x3 = self.conv4(x2)
        return x3+x1