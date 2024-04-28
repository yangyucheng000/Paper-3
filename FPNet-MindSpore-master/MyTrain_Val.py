import os

import mindspore as ms
from mindspore import nn, dataset, context, ops
from mindspore.ops import functional as F
import numpy as np
from datetime import datetime
from lib.FPNet import FPNet
from utils_2.data_val import get_loader, test_dataset
from utils_2.utils import clip_gradient, adjust_lr, dynamic_lr
import logging
from mindspore.common.api import jit
import gc
import cv2
import random
import warnings
warnings.filterwarnings('ignore')
def seed_mindspore(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
# 1*352*352
def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    # pred = ms.Tensor(pred, ms.float32)
    nnwbce = nn.BCEWithLogitsLoss(reduction='mean')
    weit = 1 + 5 * ms.ops.abs(ms.ops.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = nnwbce(pred, mask)

    wbce = (weit * wbce).sum(axis=(2, 3)) / weit.sum(axis=(2, 3))
    # 计算IoU损失
    pred = ms.ops.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(axis=(2, 3)) # 交
    union = ((pred + mask) * weit).sum(axis=(2, 3)) # 和
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()
class ComputeStructureLoss(nn.Cell):
    def __init__(self, network, loss_fn):
        super(ComputeStructureLoss, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn
  

    def construct(self, batch_x, gts):
        # print(type(batch_x))
        # print("batch_x.dtype:",batch_x.dtype)
        preds = self.network(batch_x)
        # gts = ms.Tensor(gts, ms.float32)
        # preds = ms.Tensor(preds[0], ms.float32)
        # print(type(preds))
        # print(type(gts))
        loss_init= self._loss_fn(preds[0], gts) + self._loss_fn(preds[1], gts)
        # print(loss_init)        
        loss_final = self._loss_fn(preds[2], gts)
        # print(loss_final)     
        loss = loss_init + loss_final
        # print(type(loss))
        # print("loss.dtype:",loss.dtype)
       
        return loss

class LossTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(LossTrainOneStepCell, self).__init__(network, optimizer, sens)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        x,y = inputs[0],inputs[1]
        print(type(x))
        print("x.dtype:",x.dtype)
        print(type(y))
        print("y.dtype:",y.dtype)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        print(type(grads))
        print("grads0.dtype:",grads[0].dtype)
        print("grads1.dtype:",grads[1].dtype)
        print("grads.size:",len(grads))
        # 
        # grads =  ops.tuple_to_array(grads)
        grads_list = list(grads)
        for i_iter, batch in enumerate(grads_list):
            grads_list[i_iter] =ms.Tensor(batch, ms.float32)
            print("grads"+str(i_iter)+".dtype:",grads_list[i_iter].dtype)
        # grads[0] = ops.Cast()(grads[0],ms.float32)
        grads = tuple(grads_list)
        for grad in grads:
            print("grad.dtype:",grad.dtype)
        # print("grads1.dtype:",grads[1].dtype)
        # print("loss.dtype:",loss.dtype)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class Adam(nn.optim.Adam):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                 use_nesterov=False, weight_decay=0.0, loss_scale=1.0, use_amsgrad=False, **kwargs):
        super(Adam, self).__init__(params,learning_rate, beta1, beta2, eps, use_locking, use_nesterov, weight_decay, loss_scale, use_amsgrad, **kwargs)

    def construct(self, gradients):
        params = self._parameters
        moment1 = self.moment1
        moment2 = self.moment2
        # print("====================gradients====================")
        gradients = self.flatten_gradients(gradients)
        # print("gradient.len:",len(gradients))   
        # for gradient in gradients:
        #     print("gradient.dtype:",gradient.dtype)
        gradients = self.decay_weight(gradients)
        if not self.use_offload:
            gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self._grad_sparse_indices_deduplicate(gradients)
        lr = self.get_lr()

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power

        return self._apply_adam(params, beta1_power, beta2_power, moment1, moment2, lr, gradients)

def train(train_loader, model, train_net, optimizer, epoch, save_path, hyy, loss_list):
    """
    train function
    """
    global step    #所有训练轮数

    loss_all = 0    # 本epoch的训练损失
    epoch_step = 0  # 本epoch的训练轮数
    model.set_train()   
    print('========================epoch:',epoch)
    try:
        # enumerate将可遍历对象组成一个索引序列，start参数可指定开始索引数
        for i, batch in enumerate(train_loader, start=1): # 遍历train_loader
            images, gts = batch["image"], batch["gt"]
            images = F.squeeze(images,axis=(1))
            gts = F.squeeze(gts,axis=(1))
            gts = ms.Tensor(gts, ms.float32)
            images = ms.Tensor(images, ms.float32)
            loss = train_net(images,gts) # 有三个输出，即输出三张图均为1*352*352
            step += 1
            epoch_step += 1
            loss_all += loss

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,float(loss)))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, float(loss)))

        loss_all =  float(loss_all / epoch_step)
        loss_list.append(loss_all)
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))

        if epoch % 10 == 0:
            print('save model', save_path + 'Net_epoch_{}.ckpt'.format(epoch))
            ms.save_checkpoint(model, save_path + 'Net_epoch_{}.ckpt'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ms.save_checkpoint(model, save_path + 'Net_epoch_{}.ckpt'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise

def val(test_loader, model, epoch, save_path, hyy,mae_list):
    """
    validation function
    """
    global best_mae, best_epoch
    # 测试前一般要加
    model.set_train(False)

    mae_sum = 0
    # print('test_loader.size:',test_loader.size)
    # 遍历所有测试图片
    for i in range(test_loader.size):
        image, gt, name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = ms.Tensor(image, ms.float32).unsqueeze(0)
        gt = ms.Tensor(gt, ms.float32)
        # image = image
        res = model(image)
        # print(gt.shape)
        # 模型三个输出中最后一个，即最终结果 1*352*352
        res = F.interpolate(res[2], size=gt.shape, mode='bilinear', align_corners=False)
        # print(res.shape)
        res = res.sigmoid().squeeze() # 352*352
        # print(res.shape)
        res = (res - res.min()) / (res.max() - res.min() + 1e-8) # 归一化？？？
        # print(res.shape)
        res = res.asnumpy()
        mae_sum += np.sum(np.abs(res - gt.asnumpy())) * 1.0 / (gt.shape[0] * gt.shape[1])
        # print(mae_sum)
    mae = mae_sum / test_loader.size # 除以测试图片数量
    mae_list.append(mae)
    print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
    if epoch == 1:
        best_mae = mae
    else:
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            ms.save_checkpoint(model, save_path + 'Net_epoch_CFMM_L.ckpt')
            print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
    logging.info(
        '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    hyy = "cloud"
    import argparse # 命令行参数
    # 建立解析行对象
    parser = argparse.ArgumentParser()
    # 增加属性：给parser实例增加属性
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--train_root', type=str, default='./data/TrainData/',help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./data/test_database/CHAMELEON_TestingDataset/', help='the test rgb images root')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')  #352
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    # parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/TEM-NCD-C2F-GroupInsert/',
                        help='the path to save model and log')
    # 属性给与opt实例：parser中增加的属性都在opt中
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']= opt.gpu_id
    ms.set_context(device_target="GPU")
    seed_mindspore()


    print("batchsize:{}".format(opt.batchsize))

    #一、build the model and 定义优化器--------------------------------------------------------
    print("build model")
    print(opt.trainsize)
    model = FPNet(imagenet_pretrained=False,img_size=512)
    print("finish model")
    if opt.load is not None: #将预训练的参数权重加载到新的模型之中
        param_dict = ms.load_checkpoint(opt.load)
        ms.load_param_into_net(model, param_dict)
        print('load model from ', opt.load)
    decay_lr = dynamic_lr(opt.lr,opt.epoch)
    optimizer = nn.optim.Adam(model.trainable_params(), 
                opt.lr,
                eps=1e-08,
                weight_decay=0.001)  #优化器（参数列表，学习率）
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #二、load data---------------------------------------------------------------------------
    print('load data...')
    train_loader, iterations_epoch  = get_loader(image_root=opt.train_root + 'Imgs/',
                                gt_root=opt.train_root + 'GT/',
                                batchsize=opt.batchsize,
                                trainsize=opt.trainsize,
                                num_workers=8)   # 14


    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    # 一轮训练多少次：images数量/batchsize
    total_step = iterations_epoch
    # 配置logging模块：基础配置
    logging.basicConfig(filename=save_path + 'mylog.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # 打印信息（超参数）
    logging.info("Network-Train")
    logging.info("lr=0.0001,size:512; logname:mylog.log; witer:summary")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    # 实例化write
    # writer = SummaryWriter(save_path + 'summary') # 默认写入文件的时间间隔为60
    best_mae = 1
    best_epoch = 0

    #三、training----------------------------------------------------------------------------
    print("Start train...")
    # 首先，进行垃圾回收
    # gc.collect()
    # 每次训练的平均损失列表
    loss_list = []
    # 每次验证的平均MAE
    mae_list = []


    BCE =  nn.BCEWithLogitsLoss(reduction='mean')
    FPnetmodel = ComputeStructureLoss(model,structure_loss)
    train_net = nn.TrainOneStepCell(FPnetmodel, optimizer)
    # 进行epoch次训练
    for epoch in range(1, opt.epoch):
        # 使用train_loader训练（forward \ backward \ optimizer.step )，BCE&IOU损失
        train(train_loader, model,train_net, optimizer, epoch, save_path, hyy, loss_list)
        # 使用val_loader测试, MAE损失
        val(val_loader, model, epoch, save_path, hyy,mae_list)
