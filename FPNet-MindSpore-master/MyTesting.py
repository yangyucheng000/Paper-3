import time

import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.FPNet import FPNet
from utils_2.data_val import test_dataset

import mindspore as ms
from mindspore import nn, dataset, context, ops
from mindspore.ops import functional as F
import os.path as osp

from evaluator import Eval_thread
from dataloader import EvalDataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')  # 图片大小
# parser.add_argument('--ckpt_path', type=str, default='./snapshot/FPNet-GroupInsert/FPNet.pth')
parser.add_argument('--ckpt_path', type=str, default='./snapshot/TEM-NCD-C2F-GroupInsert/Net_epoch_40.ckpt')
parser.add_argument('--methods', type=str, default='FPNet')
parser.add_argument('--gt_root_dir', type=str, default='./data/test_database/')
parser.add_argument('--pred_root_dir', type=str, default='./res_pvt_FPNet/pred/')
parser.add_argument('--save_dir', type=str, default='./res_pvt_FPNet/score/')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--gpu_id', type=str, default='2')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']= opt.gpu_id
ms.set_context(device_target="GPU")

size = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定哪个gpu跑程序
# for _data_name in ['CAMO_TestingDataset', 'COD10K_TestingDataset', 'CHAMELEON_TestingDataset']:
for _data_name in ['COD10K_TestingDataset', 'CHAMELEON_TestingDataset']:
    # 测试图片路径
    data_path = './data/test_database/{}/'.format(_data_name)
    # 预测结果存放路径
    save_path = '{}{}/'.format(opt.pred_root_dir, _data_name)
    # save_path = './res_best/{}/{}/'.format(opt.ckpt_path.split('/')[-2], _data_name)
    print("load model...")
    model = FPNet(imagenet_pretrained=False,img_size=512) # 定义模型
    param_dict = ms.load_checkpoint(opt.ckpt_path)
    ms.load_param_into_net(model, param_dict)  # 加载训练好的参数

    model.set_train(False)

    print("load data...")
    os.makedirs(save_path, exist_ok=True)  # 为结果创建文件夹
    image_root = '{}/Imgs/'.format(data_path)  # 输入图片
    gt_root = '{}/GT/'.format(data_path)  # 真值GT
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt = ms.Tensor(gt, ms.float32)
        gt /= (gt.max() + 1e-8)

        image = ms.Tensor(image, ms.float32).unsqueeze(0)

        # 预测结果 3个N*1*352*352,只用第三个
        res = model(image)  #
        res = list(res)  #
        # print(res.shape)
        res[2] = F.interpolate(res[2], size=gt.shape, mode='bilinear', align_corners=False)
        res[2] = res[2].sigmoid().squeeze()  # N*H*W
        res[2] = (res[2] - res[2].min()) / (res[2].max() - res[2].min() + 1e-8)
        res = res[2] .asnumpy()
        cv2.imwrite(save_path + name, res * 255)

pred_dir = opt.pred_root_dir
gt_dir = opt.gt_root_dir
for _data_name in [ 'COD10K_TestingDataset', 'CHAMELEON_TestingDataset']:
    loader = EvalDataset(osp.join(pred_dir, _data_name + '/'), osp.join(gt_dir, _data_name + '/GT/'))
    thread = Eval_thread(loader, opt.methods, _data_name, opt.save_dir, opt.cuda)
    print(thread.run())


