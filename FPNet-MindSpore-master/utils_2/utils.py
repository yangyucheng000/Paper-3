
import numpy as np
# 计算梯度  ---个人将它理解为神经网络训练时候的drop out的方法，用于解决神经网络训练过拟合的方法
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    # 遍历所有参数组
    for group in optimizer.param_groups:
        # 遍历参数组中的每个参数
        for param in group['params']:
            if param.grad is not None: # 存在梯度则限制梯度
                param.grad.data.clamp_(-grad_clip, grad_clip)

# 每训练一个epoch便需要调整一下lr
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    # optimizer.param_groups保存了参数组及其对应的学习率、动量等
    for param_group in optimizer.param_groups:
        # 更改对应参数组的lr
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def dynamic_lr(init_lr, total_step, decay_rate=0.1, decay_epoch=30):
    lrs = []
    for i in range(total_step):
        decay = decay_rate ** (i // decay_epoch)
        lrs.append(init_lr * decay)
    return lrs

def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))