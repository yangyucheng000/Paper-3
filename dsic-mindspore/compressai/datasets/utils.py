# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import random,cv2
# from PIL import Image
import glob
# from torch.utils.data import Dataset
# import kornia
import os,glob
# import torch
# from torchvision import transforms


# from torchvision import transforms
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset.transforms import Compose
from mindspore.dataset import vision


'''
import mindspore.nn as nn
import mindspore.numpy as npms
from mindspore import Tensor
import mindspore.nn.optim as optim
import mindspore.common.initializer as Init
from mindspore import ops as ops
'''



from compressai.models.utils import conv, deconv
from PIL import Image
import numpy as np

import time
'''
expand_dims = ops.ExpandDims()
# MEAN_torch = torch.tensor([0.485, 0.456, 0.406]).mean().unsqueeze(0)
# STD_torch = torch.tensor([0.229, 0.224, 0.225]).mean().unsqueeze(0)
MEAN = expand_dims(ops.mean(Tensor([0.485, 0.456, 0.406])),0)
STD = expand_dims(ops.mean(Tensor([0.229, 0.224, 0.225])),0)
'''
MEAN_ARRAY = [(0.485+0.456+0.406) / 3.0]
STD_ARRAY = [(0.229+0.224+0.225) / 3.0]



# MEAN = Tensor([0.485, 0.456, 0.406])

#获取4个下采样对应的h
def get_H(im1,im2): #cv2.imread+RGB
    # im1 = cv2.imread('/home/ywz/database/aftercut/train/left/2009.png')
    # im2 = cv2.imread('/home/ywz/database/aftercut/train/right/2009.png')
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)  # (H,W,3)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    H_list = []
    for resize_scale in [1]: #[1,2,4,8]: #[2, 4, 8, 16]:
        # print(im1.shape)
        # resize
        resize_im1 = cv2.resize(im1, (im1.shape[1] // resize_scale, im1.shape[0] // resize_scale))  # W,H
        resize_im2 = cv2.resize(im2, (im2.shape[1] // resize_scale, im2.shape[0] // resize_scale))
        #
        # surf = cv2.xfeatures2d.SURF_create()
        surf = cv2.SIFT_create()
        
        # print("resize_im1:",resize_im1)
        kp1, des1 = surf.detectAndCompute(resize_im1, None)
        """
        print("kp1:",kp1)
        print("des1:",des1)
        print("kp2:",kp2)
        print("des2:",des2)
        """
        # print("des1:",des1)
        
        kp2, des2 = surf.detectAndCompute(resize_im2, None)
        # 匹配特征点描述子
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
         
        
        # 提取匹配较好的特征点
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        # 通过特征点坐标计算单应性矩阵H
        # （findHomography中使用了RANSAC算法剔初错误匹配）
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # print("src_pts:",src_pts)
        # print("dst_pts:",dst_pts)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        H_tmp = H.astype(np.float32)
        # print("H_tmp:",H_tmp)
        # 获取H后，但要放进tensor中的变换
        '''
        try:
            h = Tensor.from_numpy(H.astype(np.float32))  # 否则float64，与网络中的tensor不匹配！  
        except:
            h = None
        #     print(resize_scale)
        # h_inv = torch.inverse(h) #求逆
        '''
        H_list.append(H_tmp)
        # print("H_list:",H_list)
    return H_list

class ImageFolder():
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories: ::
        - rootdir/
            - train/
                -left/
                    - 0.png
                    - 1.png
                -right/
            - test/
                -left/
                    - 0.png
                    - 1.png
                -right/
    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """
    def __init__(self, root, transform=None,patch_size=(256,256), split='train',need_file_name = False):
        splitdir = Path(root) / split  # 相当于osp.join

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        splitdir_left = splitdir / "left"
        splitdir_right = splitdir / "right"

        self.left_list = sorted(glob.glob(os.path.join(splitdir_left,"*")))
        self.right_list = sorted(glob.glob(os.path.join(splitdir_right, "*")))

        self.patch_size = patch_size
        #只保留了ToTensor
        self.transform = transform

        ###for homography 单独裁剪 不传参直接设定
        self.homopic_size = 256
        self.homopatch_size = 128
        self.rho = 45
        # print(MEAN_ARRAY)
        # print(STD_ARRAY)
        # print(MEAN)
        # print(STD)
        # print(MEAN.asnumpy().tolist())
        # print(STD.asnumpy().tolist())


        self.homotransforms = Compose(
            [
                # ywz
                # transforms.Resize(self.homopic_size),
                # #
                # transforms.CenterCrop(self.homopic_size),
                vision.ToTensor(),
                # transforms.Normalize(mean=MEAN, std=STD),
                vision.Normalize(mean=MEAN_ARRAY, std=STD_ARRAY, is_hwc=False)
            ]
        )

        '''
        self.homotransforms = Compose(
            [
                # ywz
                # transforms.Resize(self.homopic_size),
                # #
                # transforms.CenterCrop(self.homopic_size),
                vision.ToTensor(),
                # transforms.Normalize(mean=MEAN, std=STD),
                vision.Normalize(mean=MEAN.asnumpy().tolist(), std=STD.asnumpy().tolist(), is_hwc=False)
            ]
        )
        '''
        
        ########################################

        self.need_file_name = need_file_name

    def __getitem__(self, index):    #getitem还没改完
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # img1 = Image.open(self.left_list[index]).convert('RGB')
        # img2 = Image.open(self.right_list[index]).convert('RGB')
        if os.path.basename(self.left_list[index]) != os.path.basename(self.right_list[index]):
            print(self.left_list[index])
            raise ValueError("cannot compare pictures.")
        ##
        img1 = cv2.imread(self.left_list[index])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(self.right_list[index])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #random cut for pair
        H, W, _ = img1.shape
        #randint是闭区间
        # print(H)
        # print(W)
        # print(self.patch_size)
        '''
        if self.patch_size[0]==H:
            startH = 0
            startW = 0
        else:
            startH = random.randint(0,H-self.patch_size[0]-1)
            startW = random.randint(0,W-self.patch_size[1]-1)
        '''
        startH = 0
        startW = 0
        img1 = img1[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        img2 = img2[startH:(startH + self.patch_size[0]), startW:(startW + self.patch_size[1])]
        ###
        # print(img1.shape)  #（512，512，3）
        # raise ValueError("stop utils")

        H_list = get_H(img1,img2) #可以忽略 这个是传统单应性获取的方法

        #for homo 在上述patch基础上再进行缩放和裁剪 返回patch以及相应的corners
        homo_img1 = cv2.resize(img1,(self.homopic_size,self.homopic_size))
        homo_img2 = cv2.resize(img2, (self.homopic_size, self.homopic_size))
        homo_img1 = self.homotransforms(homo_img1)
        homo_img2 = self.homotransforms(homo_img2)
        
        '''
        op = ops.ReduceMean(keep_dims=True)
        print("homo_img1:",homo_img1.shape)
        homo_img1 = op(homo_img1, 0)  # 转灰度
        print("homo_img1_afterops:",homo_img1.shape)
        homo_img2 = op(homo_img2, 0)  # 转灰度
        '''
        homo_img1 = np.mean(homo_img1, axis = 0, keepdims=True)
        homo_img2 = np.mean(homo_img2, axis = 0, keepdims=True)

        # pick top left corner
        if self.homopic_size - self.rho - self.homopatch_size >= self.rho:
            x = random.randint(self.rho, self.homopic_size - self.rho - self.homopatch_size)
            y = random.randint(self.rho, self.homopic_size - self.rho - self.homopatch_size)
        else:
            x = 0
            y = 0
        x = 45    
        y = 45
        # print(x,y)
        '''
        corners = torch.tensor(
            [
                [x, y],
                [x + self.homopatch_size, y],
                [x + self.homopatch_size, y + self.homopatch_size],
                [x, y + self.homopatch_size],
            ],dtype=torch.float32
        )
        '''
        
        corners = [
                [x, y],
                [x + self.homopatch_size, y],
                [x + self.homopatch_size, y + self.homopatch_size],
                [x, y + self.homopatch_size],
        ]
        
        # corners = corners.astype(np.float32) 
        # print("utils_corners:",corners.shape)
        
        homo_img1 = homo_img1[:, y: y + self.homopatch_size, x: x + self.homopatch_size]
        homo_img2 = homo_img2[:, y: y + self.homopatch_size, x: x + self.homopatch_size]
        ################## [homo_img1,homo_img2,corners]     #################### %zhm 20230216今天改到这里 ##############################################


        ##
        # print("H_list[0]:",H_list[0])
        if H_list[0].all()==None:
            print(self.left_list[index])
            print(self.right_list[index])
            #raise ValueError("None!!H_matrix")
            # 只有ToTensor
            if self.transform:
                return self.transform(img1), self.transform(img2) # ,H_list[1],H_list[2],H_list[3]
            return img1, img2  # ,H_list[1],H_list[2],H_list[3]

        #只有ToTensor
        if self.transform:
            # return self.transform(img1),self.transform(img2),H_list[0] #,H_list[1],H_list[2],H_list[3]
            if self.need_file_name:
                return self.transform(img1), self.transform(img2), H_list[0], os.path.basename(self.left_list[index]),homo_img1,homo_img2,corners  # ,H_list[1],H_list[2],H_list[3]
            else:
                return self.transform(img1), self.transform(img2), H_list[0],homo_img1,homo_img2,corners  # ,H_list[1],H_list[2],H_list[3]

        if self.need_file_name:
            return img1, img2, H_list[0],os.path.basename(self.left_list[index]),homo_img1,homo_img2,corners  # ,H_list[1],H_list[2],H_list[3]
        else:
            return img1,img2,H_list[0],homo_img1,homo_img2,corners #,H_list[1],H_list[2],H_list[3]

    def __len__(self):
        return len(self.left_list)
