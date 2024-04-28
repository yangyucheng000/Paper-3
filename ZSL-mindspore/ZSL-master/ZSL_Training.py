# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:56:22 2020

@author: Dian
"""
import cv2
from scipy.io import loadmat
from mindspore import nn, Model, Tensor, context,save_checkpoint
import mindspore
import mindspore.ops as ops

import h5py
import mindspore.dataset as ds
from model import *
import scipy
from scipy import *
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:20:42 2020

@author: Dian
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:06:42 2020

@author: Dian
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import numpy as np
import tifffile


def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div;
    return F

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

def Gaussian_downsample(x,psf,s):
    y=np.zeros((x.shape[0],int(x.shape[1]/s),int(x.shape[2]/s)))
    if x.ndim==2:
        x=np.expand_dims(x,axis=0)
    for i in range(x.shape[0]):
        x1=x[i,:,:]
        x2=signal.convolve2d(x1,psf, boundary='symm',mode='same')
        y[i,:,:]=x2[0::s,0::s]
    return y

def sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
#    if not img1.shape == img2.shape:
#        raise ValueError('Input images must have the same dimensions.')
#    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
#    img1_ = img1.astype(np.float64)
#    img2_ = img2.astype(np.float64)
    img1_ = np.float64(np.uint8(np.round(img1*255)))
    img2_ = np.float64(np.uint8(np.round(img2*255)))
    inner_product = (img1_ * img2_).sum(axis=2)
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))


def rmse1(Fuse1, HRHSI):
    ap = []
    ae = []
    for j in range(Fuse1.shape[0]):
        be = np.mean(np.square(
            np.float64(np.uint8(np.round(Fuse1[j, ...] * 255))) - np.float64(np.uint8(np.round(HRHSI[j, ...] * 255)))))
        bp = 10 * np.log10((255 ** 2) / be)
        ap.append(bp)
        ae.append(be)

    temp_rmse = np.sqrt(np.mean(np.array(ae)))
    temp_psnr = np.mean(np.array(ap))
    return temp_rmse, temp_psnr

class HSIDataset:
    """自定义数据集类"""

    def __init__(self, train_hrhs_all, train_hrms_all, train_lrhs_all):
        self.train_hrhs_all = train_hrhs_all
        self.train_hrms_all = train_hrms_all
        self.train_lrhs_all = train_lrhs_all

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        return train_hrhs, train_hrms, train_lrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]

def  dataset_input(data_put,downsample_factor):
  if data_put=='pavia':
        F=loadmat('.\data\R.mat')
        F=F['R']
        F=F[:,0:-10]
        for band in range(F.shape[0]):
            div = np.sum(F[band][:])
            for i in range(F.shape[1]):
                F[band][i] = F[band][i]/div;
        R=F
        HRHSI=tifffile.imread('.\data\original_rosis.tif')
        HRHSI=HRHSI[0:-10,0:downsample_factor**2*int(HRHSI.shape[1]/downsample_factor**2),0:downsample_factor**2*int(HRHSI.shape[2]/downsample_factor**2)]
        HRHSI=HRHSI/np.max(HRHSI)
  elif data_put=='Chikusei':
        mat=h5py.File('.\data\Chikusei.mat')
        HRHSI=mat['chikusei']
        mat1=sio.loadmat('.\data\Chikusei_data.mat')
        R=mat1['R']
        R=R[0:8:2,:]
        HRHSI=HRHSI[:,100:900,100:900]
        HRHSI=np.transpose(HRHSI,(0,2,1))
        x1=np.max(HRHSI)
        x2=np.min(HRHSI)
        x3=-x2/(x1-x2)
        HRHSI=HRHSI/(x1-x2)+x3
  elif data_put=='houston':
        mat=sio.loadmat('.\data\Houston.mat')
        HRHSI=mat['Houston']
        HRHSI=np.transpose(HRHSI,(2,0,1))
        HRHSI=HRHSI[:,0:336,100:900]
        x1=np.max(HRHSI)
        x2=np.min(HRHSI)
        x3=-x2/(x1-x2)
        HRHSI=HRHSI/(x1-x2)+x3
        R=np.zeros((4,HRHSI.shape[0]));
        for i in range(R.shape[0]):
          R[i,36*i:36*(i+1)]=1/36.0
  else:
        sys.exit(0)
  return HRHSI,R

PSF = fspecial('gaussian', 7, 3)
p=10
stride=1
training_size=32
downsample_factor=4
LR=1e-3
EPOCH=400
BATCH_SIZE=64 
loss_optimal=1.75
init_lr1=1e-4
init_lr2=5e-4
decay_power=1.5
data2='pavia'
[HRHSI,R]=dataset_input(data2,downsample_factor)
maxiteration=2*math.ceil(((HRHSI.shape[1]/downsample_factor-training_size)//stride+1)*((HRHSI.shape[2]/downsample_factor-training_size)//stride+1)/BATCH_SIZE)*EPOCH
print(maxiteration)

warm_iter=math.floor(maxiteration/40)
print(maxiteration)
HSI0=Gaussian_downsample(HRHSI,PSF,downsample_factor)
SNRh=30;
sigma = np.sqrt(np.sum(HSI0**2)/(10**(SNRh/10))/(HSI0.shape[0]*HSI0.shape[1]*HSI0.shape[2]));
HSI0=HSI0+ 0*np.random.randn(HSI0.shape[0],HSI0.shape[1],HSI0.shape[2])
MSI0=np.tensordot(R,  HRHSI, axes=([1], [0]))
sigma = np.sqrt(np.sum(MSI0**2)/(10**(SNRh/10))/(MSI0.shape[0]*MSI0.shape[1]*MSI0.shape[2]));
SNRh=35;
sigma = np.sqrt(np.sum(MSI0**2)/(10**(SNRh/10))/(MSI0.shape[0]*MSI0.shape[1]*MSI0.shape[2]));
MSI0=MSI0+ 0*np.random.randn(MSI0.shape[0],MSI0.shape[1],MSI0.shape[2])
HSI3=HSI0.reshape(HSI0.shape[0],-1)
U0,S,V=np.linalg.svd(np.dot(HSI3,HSI3.T))
U0=U0[:,0:int(p)]
HSI0_Abun=np.tensordot(U0.T,  HSI0, axes=([1], [0]))
augument=[0]
HSI_aug=[]
HSI_aug.append(HSI0)
MSI_aug=[]
MSI_aug.append(MSI0)
U=U0
train_hrhs = []
train_hrms = []
train_lrhs= []
for j in augument:       
    HSI = cv2.flip(HSI0, j)
#        MSI_aug.append(MSI0)
    HSI_aug.append(HSI)
for j in range(len(HSI_aug)):
    HSI = HSI_aug[j]
#        MSI = MSI_aug[j]
    HSI_Abun=np.tensordot(U.T,  HSI, axes=([1], [0]))
    HSI_LR_Abun=Gaussian_downsample(HSI_Abun,PSF,downsample_factor)
    MSI_LR=np.tensordot(R,  HSI, axes=([1], [0])) 
    LRHSI=Gaussian_downsample(HSI,PSF,downsample_factor)
    for j in range(0, HSI_Abun.shape[1]-training_size+1, stride):
        for k in range(0, HSI_Abun.shape[2]-training_size+1, stride):
            
            temp_hrhs = HSI[:,j:j+training_size, k:k+training_size]
            temp_hrms = MSI_LR[:,j:j+training_size, k:k+training_size]
            temp_lrhs = HSI_LR_Abun[:,int(j/downsample_factor):int((j+training_size)/downsample_factor), int(k/downsample_factor):int((k+training_size)/downsample_factor)]
            
            train_hrhs.append(temp_hrhs)
            train_hrms.append(temp_hrms)
            train_lrhs.append(temp_lrhs)

train_data=HSIDataset(train_hrhs,train_hrms,train_lrhs)
trainset = ds.GeneratorDataset(train_data, ["data", "label"])
trainset = trainset.batch(32, True)





cnn=CNN(p,MSI0.shape[0]).cuda() 
optimizer = mindspore.nn.Adam(cnn.trainable_params(), learning_rate=LR,  beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0)
scheduler =nn.cosine_decay_lr(min_lr=1e-6, max_lr=1e-4, total_step=maxiteration, step_per_epoch=2, decay_epoch=2)
# optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.L1Loss(reduction='mean')
PSF_T=Tensor(PSF)

for m in cnn.modules():
  if isinstance(m, (nn.Conv2d, nn.Linear)):
    nn.init.xavier_uniform_(m.weight)    
MSI_1=Tensor(np.expand_dims(MSI0,axis=0))
HSI_1=Tensor(np.expand_dims(HSI0_Abun,axis=0))
step=0
loss_list=[]
U22=Tensor(U0)
def train_loop(cnn, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data1, label,data2):
        output = cnn(data1,data2)
        loss = loss_fn(output, label)
        return loss, output

    # Get gradient function
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data1, label,data2):
        (loss, _), grads = grad_fn(data1, label,data2)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    cnn.set_train()
    for batch, (data1, label,data2) in enumerate(dataset.create_tuple_iterator()):
        data1=mindspore.Tensor.float(data1)
        data2 = mindspore.Tensor.float(data2)
        label = mindspore.Tensor.float(label)
        loss = train_step(data1, label,data2)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(cnn, trainset, loss_func, optimizer)
    if t%2==0:
        save_checkpoint(cnn, 'check/' + 'cave' + str(t) + 'last.ckpt')



