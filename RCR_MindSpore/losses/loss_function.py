import numpy as np
import warnings
from parameter import *
from scipy.stats import gmean, pearsonr
import mindspore 
from mindspore import ops

warnings.filterwarnings("ignore")


def transform01(x):
    x_shape = x.shape
    y = x.clone().detach().reshape(-1)
    for i in range(0, y.shape[0]):
        if y[i] <= 0.5:
            y[i] = 0
        else:
            y[i] = 1
    return y.reshape(x_shape)


def real_loss(pre, target):
    if len(target.shape) == 1:
        target = ops.unsqueeze(target, dim=1)
    return ops.mse_loss(pre, target)


# ----------------------------------------------------------------------------------------------------------------------
def RwR_loss_sigmoid(pre, reject, real, c, inf=0.0, s=False):
    """Sigmoid"""
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    loss = ops.mse_loss(pre, real, reduction='none')
    if s:
        return ops.mean(loss)
    reject = ops.sigmoid(reject)
    if inf > 0:
        """cilp"""
        w = ops.relu(reject - inf) + inf
    else:
        w = ops.relu(reject - inf) + inf
    loss = loss * w + c * (1 - w)
    return ops.mean(loss)


# ----------------------------------------------------------------------------------------------------------------------
def RwR_loss_logistic(pre, reject, real, c, inf=0.0, s=False):
    """logistic"""
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    loss = ops.mse_loss(pre, real, reduction='none')
    if loss.shape[1] > 1:
        loss = ops.mean(loss, axis=1)
        loss = ops.unsqueeze(loss, dim=1)
    if s:
        return ops.mean(loss)
    else:
        w1 = ops.log(1 + ops.exp(reject))
        w2 = ops.log(1 + ops.exp(-reject))
        loss = loss * w1 + c * w2
    return ops.mean(loss)


# ----------------------------------------------------------------------------------------------------------------------
def RwR_loss_mae(pre, reject, real, c, inf=0, s=False):
    """MAE"""
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    loss = ops.mse_loss(pre, real, reduction='none')
    if s:
        return ops.mean(loss)
    loss = loss * ops.abs(reject + 1) + c * ops.abs(reject - 1)
    return ops.mean(loss)


# ----------------------------------------------------------------------------------------------------------------------
def RwR_loss_mse(pre, reject, real, c, inf=0, s=False):
    """Square loss"""
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    loss = ops.mse_loss(pre, real, reduction='none')
    if s:
        return ops.mean(loss)
    loss = loss * ops.pow(reject + 1, 2) + c * ops.pow(reject - 1, 2)
    return ops.mean(loss)


# ----------------------------------------------------------------------------------------------------------------------
def RwR_loss_hinge(pre, reject, real, c, inf=0, s=False):
    """hinge loss"""
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    loss = ops.mse_loss(pre, real, reduction='none')
    if loss.shape[1] > 1:
        loss = ops.mean(loss, axis=1)
        loss = ops.unsqueeze(loss, dim=1)
    if s:
        return ops.mean(loss)
    loss = loss * 0.5 * ops.relu(reject + 1) + c * 0.5 * ops.relu(1 - reject)
    return ops.mean(loss)


# ----------------------------------------------------------------------------------------------------------------------
def RwR_Risk_Evaluation(pre, reject, real, c):
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    reject = ops.sigmoid(reject)
    reject_01 = transform01(reject)
    loss = ops.mse_loss(pre, real, reduction='none')
    if loss.shape[1] > 1:
        loss = ops.mean(loss, axis=1)
        loss = ops.unsqueeze(loss, dim=1)
    loss = loss * reject_01 + c * (1 - reject_01)
    return ops.mean(loss)


# ----------------------------------------------------------------------------------------------------------------------
def A_loss(pre, reject, real):
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    reject = ops.sigmoid(reject)
    reject_01 = transform01(reject)
    loss = ops.mse_loss(pre, real, reduction='none')
    if loss.shape[1] > 1:
        loss = ops.mean(loss, axis=1)
        loss = ops.unsqueeze(loss, dim=1)
    loss = loss * reject_01
    if ops.sum(reject_01) == 0:
        return ops.tensor(0)
    return ops.sum(loss) / ops.sum(reject_01)


# ----------------------------------------------------------------------------------------------------------------------
def R_loss(pre, reject, real):
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    reject = ops.sigmoid(reject)
    reject_01 = transform01(reject)
    loss = ops.mse_loss(pre, real, reduction='none')
    if loss.shape[1] > 1:
        loss = ops.mean(loss, axis=1)
        loss = ops.unsqueeze(loss, dim=1)
    loss = loss * (1 - reject_01)
    if ops.sum(1 - reject_01) == 0:
        return ops.tensor(0)
    return ops.sum(loss) / ops.sum(1 - reject_01)


# ----------------------------------------------------------------------------------------------------------------------
def Reject_Rate(reject):
    reject = ops.sigmoid(reject)
    reject_01 = transform01(reject)
    return ops.mean(1 - reject_01)


# ----------------------------------------------------------------------------------------------------------------------
def R_A(pre, reject, real, c):
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    reject = ops.sigmoid(reject)
    loss = ops.mse_loss(pre, real, reduction='none')
    if loss.shape[1] > 1:
        loss = ops.mean(loss, axis=1)
        loss = ops.unsqueeze(loss, dim=1)
    n = 0
    m = 0
    for i in range(0, reject.shape[0]):
        if loss[i] > c:
            m = m + 1
            if reject[i] > 0.5:
                n = n + 1
    return n, m


# ----------------------------------------------------------------------------------------------------------------------
def A_R(pre, reject, real, c):
    if len(real.shape) == 1:
        real = ops.unsqueeze(real, dim=1)
    reject = ops.sigmoid(reject)
    loss = ops.mse_loss(pre, real, reduction='none')
    if loss.shape[1] > 1:
        loss = ops.mean(loss, axis=1)
        loss = ops.unsqueeze(loss, dim=1)
    n = 0
    m = 0
    for i in range(0, reject.shape[0]):
        if loss[i] <= c:
            m = m + 1
            if reject[i] <= 0.5:
                n = n + 1
    return n, m
# ----------------------------------------------------------------------------------------------------------------------
