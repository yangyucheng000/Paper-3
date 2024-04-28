import os
import argparse
from tkinter.messagebox import NO
import numpy as np
import cppimport
import cppimport.import_hook
import random

import mindspore
from mindspore import nn
from mindspore.dataset import GeneratorDataset

from Models import LeNet, resnet20
from getData import getLocalData, getTestData, getSampleData
from connect import connecter
from myMPC import calculator
import SpeflGlobal

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=40, help='numer of the clients')
parser.add_argument('-E', '--epoch', type=int, default=100, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=256, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_lenet', help='the model to train')
# parser.add_argument('-mn', '--model_name', type=str, default='cifar10_resnet', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-ad', '--num_of_adversary', default=0, type=int)

'''
调用cpp代码编写的函数
get_garble函数用于获取指定长度的随机向量
其他函数用于向量与矩阵运算
'''
m = cppimport.imp("secagg")

ROLE_SP0 = SpeflGlobal.ROLE_SP0
ROLE_SP1 = SpeflGlobal.ROLE_SP1
ROLE_CP = SpeflGlobal.ROLE_CP
ROLE_CLIENTS = SpeflGlobal.ROLE_CLIENTS
prec = SpeflGlobal.prec

mnist_path = './data/MNIST_data'
cifar_path = './data/cifar-10-batches-bin'

if __name__ == "__main__":
    print("Client start")
    args = parser.parse_args()
    args = args.__dict__

    num_clients = args['num_of_clients']
    # 进行网络链接，初始化安全多方计算执行对象
    conner = {}
    mpcer = {}
    for i in range(num_clients):
        conner[i] = connecter(ROLE_CLIENTS, i, num_clients)
        mpcer[i] = calculator(conner[i], m)

    # mindspore
    device = mindspore.get_context("device_target")
    mindspore.set_context(device_target=device)
    # print("Clients: " + device)

    net = None
    if args['model_name'] == 'mnist_lenet':
        net = LeNet()
    elif args['model_name'] == 'cifar10_resnet':
        net = resnet20()

    # 加载数据集
    local_datatset = None
    if args['model_name'] == 'mnist_lenet':
        local_datatset = getLocalData(mnist_path, 'mnist')
    elif args['model_name'] == 'cifar10_resnet':
        local_datatset = getLocalData(cifar_path, 'cifar')

    testDataLoader = None
    if args['model_name'] == 'mnist_lenet':
        testset = getTestData(mnist_path, 'mnist')
        testDataLoader = GeneratorDataset(testset, column_names=['image','label'],shuffle=True)
        testDataLoader = testDataLoader.batch(batch_size=256)
    elif args['model_name'] == 'cifar10_resnet':
        testset = getTestData(cifar_path, 'cifar')
        testDataLoader = GeneratorDataset(testset, column_names=['image','label'],shuffle=True)
        testDataLoader = testDataLoader.batch(batch_size=256)

    local_parameters = {}
    parameters_name = []
    for name, param in net.parameters_and_names():
        local_parameters[name] = param.clone()
        parameters_name.append(name)

    parm_lenth = {}
    for name in parameters_name:
        parm_lenth[name] = len(local_parameters[name].view(-1).tolist())

    # 接收整形化的初始参数并转换为浮点数
    for i in range(num_clients):
        parmlist_i = mpcer[i].restruct_recv(ROLE_SP0, ROLE_SP1)
    parmlist = []
    for j in range(len(parmlist_i)):
        parmlist.append(parmlist_i[j] / prec)
    # 将参数转换为张量
    global_parameters = {}
    start = 0
    end = parm_lenth[parameters_name[0]]
    for j in range(len(parameters_name)):
        global_parameters[parameters_name[j]] = mindspore.Parameter(
            mindspore.Tensor(parmlist[start:end]).view_as(local_parameters[parameters_name[j]]),
            name=parameters_name[j]
        )
        if j < len(parameters_name) - 1:
            start += parm_lenth[parameters_name[j]]
            end += parm_lenth[parameters_name[j + 1]]

    conner[0].StartRecord()
    total_send = 0
    total_recv = 0

    all_range = list(range(len(local_datatset)))
    random.shuffle(all_range)
    data_len = int(len(local_datatset) / num_clients)
    net.set_train()
    for epoch in range(args['epoch']):
        print("epoch: ", epoch)
        # 每个用户训练
        for i in range(num_clients):
            indices = all_range[i * data_len: (i + 1) * data_len]
            param_not_load, _ = mindspore.load_param_into_net(net, global_parameters, strict_load=False)
            random.shuffle(indices)
            if args['model_name'] == 'mnist_lenet':
                temp_dataset = getSampleData(mnist_path, "mnist", indices)
            elif args['model_name'] == 'cifar10_resnet':
                temp_dataset = getSampleData(cifar_path, "cifar", indices)
            train_dl = GeneratorDataset(temp_dataset, column_names=['image', 'label'])
            train_dl = train_dl.batch(args['batchsize'])

            optimizer = nn.SGD(net.trainable_params(), learning_rate=args['learning_rate'], momentum=0.9)
            for data, label in train_dl:
                def forward_fn(data, label):
                    output = net(data)
                    loss = mindspore.ops.cross_entropy(output, label)
                    return loss, output

                grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

                def train_step(data, label):
                    (loss, _), grads = grad_fn(data, label)
                    optimizer(grads)
                    return loss

                label = label.long()
                loss = train_step(data, label)

            local_grad = []
            for name, param in net.parameters_and_names():
                local_grad.extend((global_parameters[name] - param).view(-1).tolist())

            # 向两个SP发送更新
            Grad = []
            for j in range(len(local_grad)):
                Grad.append(int(local_grad[j] * prec))
            mpcer[i].share_send(ROLE_SP0, ROLE_SP1, Grad)
            # 向CP发送本次更新的标准差(用于相关系数计算)
            conner[i].send(conner[i].conn['id{}'.format(ROLE_CP)], np.std(local_grad))

        # 接收参数，转化为张量，并更新模型
        for i in range(num_clients):
            updata_i = mpcer[i].restruct_recv(ROLE_SP0, ROLE_SP1)
        updatalist = []
        for j in range(len(updata_i)):
            updatalist.append(updata_i[j] / (prec * prec))

        for j in range(len(parmlist)):
            parmlist[j] -= updatalist[j]

        start = 0
        end = parm_lenth[parameters_name[0]]
        for j in range(len(parameters_name)):
            global_parameters[parameters_name[j]] = mindspore.Parameter(
                mindspore.Tensor(parmlist[start:end]).view_as(local_parameters[parameters_name[j]]),
                name=parameters_name[j]
            )
            if j < len(parameters_name) - 1:
                start += parm_lenth[parameters_name[j]]
                end += parm_lenth[parameters_name[j + 1]]
        param_not_load, _ = mindspore.load_param_into_net(net, global_parameters, strict_load=False)

        total_send += conner[0].record_send
        total_recv += conner[0].record_recv
        conner[0].CleanRecord()


        if (i + 1) % args['val_freq'] == 0:
            sum_accu = 0
            num = 0

            for data, label in testDataLoader:
                preds = net(data)
                preds = mindspore.ops.argmax(preds, dim=1)
                sum_accu += (preds.type_as(label) == label).float().mean()
                num += 1
            print("acc: ", sum_accu / num)