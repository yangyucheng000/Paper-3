import os
import sys
import time
import math
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.common.initializer as init

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = ds.GeneratorDataset(dataset, ["data", "label"], shuffle=True)
    dataloader.batch(1)
    mean = np.zeros(3)
    std = np.zeros(3)
    print('==> Computing mean and std..')
    for data in dataloader.create_dict_iterator(output_numpy=True):
        inputs = data["data"]
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(init.HeNormal(mode='fan_out')(cell.weight.shape))
            if cell.bias is not None:
                cell.bias.set_data(init.Constant(0))
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.Constant(1))
            cell.beta.set_data(init.Constant(0))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(init.Normal(1e-3)(cell.weight.shape))
            if cell.bias is not None:
                cell.bias.set_data(init.Constant(0))


last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None, pre_msg=''):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    current+=1
    
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    eta_time = step_time*(total-current)

    L = []
    L.append(' Time:%s' % format_time2(tot_time))
    L.append(' | ETA:%s' % format_time2(eta_time))
    if msg:
        L.append(' | ' + msg)

    msg_ = ''.join(L)

    columns, rows = os.get_terminal_size(0)
    TOTAL_BAR_LENGTH=columns-len(msg_)-len(pre_msg)-3
    if TOTAL_BAR_LENGTH<20:
        L = []
        L.append(' Time:%s' % format_time(tot_time))
        L.append(' | ETA:%s' % format_time(eta_time))
        if msg:
            L.append(' | ' + msg)

        msg_ = ''.join(L)

        msg_=msg_.replace(' ','')
        msg_=msg_.replace(':','')
        pre_msg=pre_msg.replace(' ','')
        pre_msg=pre_msg.replace(':','')
        
        TOTAL_BAR_LENGTH=columns-len(msg_)-len(pre_msg)-3
    TOTAL_BAR_LENGTH=np.clip(TOTAL_BAR_LENGTH,5,80)

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    
    sys.stdout.write(pre_msg)
    
    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    
    sys.stdout.write(msg_)
    
    space_len=columns-(len(pre_msg)+TOTAL_BAR_LENGTH+len(msg_)+2)-1
    for i in range(space_len):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    pct=' %d/%d ' % (current, total)
    if TOTAL_BAR_LENGTH>len(pct):
        for i in range(int(TOTAL_BAR_LENGTH/2+len(pct)/2)+len(msg_)+space_len):
            sys.stdout.write('\b')
        sys.stdout.write(pct)

    if current < total:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*10)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += "%2d" % hours + ':'
        i += 1
    if minutes > 0 and i <= 2:
        f += "%2d" % minutes + ':'
        i += 1
    if secondsf >= 0 and i <= 2:
        f += "%2d" % secondsf + '.'
        i += 1
    if millis >= 0 and i <= 2:
        f += "%01d" % millis
        i += 1
    if f == '':
        f = '0.0'
    return f


def format_time2(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    if days > 0:
        f += str(days) + 'D'
    if hours > 0:
        f += "%2d" % hours + ':'
    if minutes > 0:
        f += "%2d" % minutes + ':'
    if secondsf >= 0:
        f += "%2d" % secondsf + '.'
    f += "%03d" % millis
    return f
