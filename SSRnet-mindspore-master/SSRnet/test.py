# -*- coding:utf-8 -*-
#from torch.autograd import Variable
from tqdm import tqdm
from mindspore import nn, Model, Tensor, context
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.ops import functional as F
from mindspore.common.initializer import Normal
import mindspore.ops as ops
import mindspore.nn as nn
import hdf5storage
from scipy.io import loadmat
import h5py
from mindspore.common.initializer import Normal
import numpy as np
from mindspore import ops
from mindspore import Tensor
import mindspore
import math
# from model3 import *
import time
import os

from SSRnet import *
import mindspore.dataset as ds
zeros_op = P.Zeros()
zero_tensor = zeros_op((31, 512,512), mstype.float32)

zeros = ops.Zeros()
shape=(31,512,512)
def reconstruction(net2, R, R_inv, MSI, training_size, stride):
    index_matrix = Tensor(zero_tensor)
    abundance_t = Tensor(zero_tensor)
    a = []
    for j in range(0, 512 - training_size + 1, stride):
        a.append(j)
    a.append(512 - training_size)
    b = []
    for j in range(0, 512 - training_size + 1, stride):
        b.append(j)
    b.append(512 - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            HSI = net2(R, R_inv, temp_hrms)
            HSI = HSI.squeeze()
            HSI = ops.clip_by_value(HSI, 0, 1)
            abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
            index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    return HSI_recon

path='D:\\code\\cave\\caveall\\'
imglist=os.listdir(path)
model = CNN_BP_SE5(1)
param_dict = mindspore.load_checkpoint("C:\\Users\95395\Desktop\huawei\cave76last.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
R = create_F()
R = R.astype(np.float32)
R_inv = np.linalg.pinv(R)
R_inv = Tensor(R_inv)
R = Tensor(R)
test_path = 'C:\\Users\\95395\\Desktop\\huawei\\testpic2\\'

RMSE = []
training_size = 64
stride = 32
for i in range(0,len(imglist)):
    img=loadmat(path+imglist[i])  # You will need to replace this with your function
    img1=img["b"]
    #img1=img1/img1.max()
    HRHSI=np.transpose(img1,(2,0,1))
    HRHSI = HRHSI.astype(np.float32)
    HRHSI = Tensor(HRHSI)
    MSI=ops.tensor_dot(R,  HRHSI, axes=([1], [0]))
    MSI_1= ops.expand_dims(MSI, 0)
    Fuse=reconstruction(model,R,R_inv,MSI_1,training_size,stride)
    Fuse=Fuse.asnumpy()
    Fuse=np.squeeze(Fuse)
    Fuse=np.clip(Fuse,0,1)
    faker_hyper = np.transpose(Fuse,(1,2,0))
    print(faker_hyper.shape)
    test_data_path=os.path.join(test_path+imglist[i])
    hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
    hdf5storage.savemat(test_data_path, {'rea': img1}, format='7.3')
