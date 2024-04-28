import os
import argparse
from tqdm import tqdm
import numpy as np
import mindspore as ms
import cppimport
import cppimport.import_hook

from Models import LeNet, resnet20
from myMPC import calculator
from connect import connecter
import SpeflGlobal

# 运行:
# python CP.py
# python SP.py -id 0
# python SP.py -id 1
# python Clients.py

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=40, help='numer of the clients')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_lenet', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=20, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=1, help='the way to allocate data to clients')

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

if __name__ == "__main__":

    print("CP start")
    device = ms.get_context("device_target")
    ms.set_context(device_target=device)
    # print("CP: " + device)

    args = parser.parse_args()
    args = args.__dict__

    conner = connecter(ROLE_CP, 0, int(args['num_of_clients']))
    mpcer = calculator(conner, m)

    net = None
    if args['model_name'] == 'mnist_lenet':
        net = LeNet()
    elif args['model_name'] == 'cifar10_resnet':
        net = resnet20()

    num_clients = int(args['num_of_clients'])
    clients_in_comm = ['client{}'.format(i) for i in range(num_clients)]

    global_parameters = {}
    parameters_name = []
    for name, param in net.parameters_and_names():
        global_parameters[name] = param.clone()
        parameters_name.append(name)

    parmlist = []
    for name in parameters_name:
        parmlist.extend(global_parameters[name].view(-1).tolist())
    for j in range(len(parmlist)):
        parmlist[j] = int(parmlist[j] * prec)
    
    mpcer.share_send(ROLE_SP0, ROLE_SP1, parmlist)

    conner.StartRecord()
    total_send = 0
    total_recv = 0

    for i in tqdm(range(args['num_comm'])):
        #向两个SP发送乘法三元组
        mpcer.getmask_test(ROLE_CP, num_clients, len(parmlist))
        #从每个用户接收其本地更新的标准差
        sigmaG = []
        for j in range(num_clients):
            ss = conner.recv(conner.conn['id{}'.format(ROLE_CLIENTS+j)])
            sigmaG.append(ss)

        #从两个SP接收混淆后的梯度，计算中位数，再发回其秘密共享
        R = {}
        Rmed = []
        for client in clients_in_comm:
            R[client] = mpcer.restruct_recv(ROLE_SP0,ROLE_SP1)

        for j in range(len(R["client1"])):
            temp = []
            for client in clients_in_comm:
                temp.append(R[client][j])
            Rmed.append(int(np.median(temp)))

        mpcer.share_send(ROLE_SP0, ROLE_SP1, Rmed)


        #计算相关系数公式的分子部分
        E = mpcer.restruct_recv(ROLE_SP0, ROLE_SP1)
        for j in range(len(E)):
            E[j] = E[j]/ len(parmlist)
            E[j] = (E[j]/ prec)/prec
        #计算中位数标准差
        sigmaM = mpcer.restruct_recv(ROLE_SP0, ROLE_SP1)
        sigmaM = sigmaM[0] / len(parmlist)
        sigmaM = np.sqrt(sigmaM)
        sigmaM = sigmaM / prec

        #计算相关系数与权重
        beta = []
        myu = []
        total = 0
        for j in range(num_clients):
            rho = (E[j]) / (sigmaM * sigmaG[j]) 
            if rho > 1:
                rho = 0.99
            
            myu.append(max(0, np.log(((1 + rho) / (1 - rho))).item()-0.5))
            if j < 0:
                myu[j] = 0
            total += myu[j]

        for j in range(num_clients):
            if total != 0:
                beta.append(myu[j]/total)
            else:
                beta.append(1/num_clients)
        
        beta_int = []
        for j in range(len(beta)):
            beta_int.append(int(beta[j]* prec))
        
        #向两个SP发送权重的秘密共享
        mpcer.share_send(ROLE_SP0, ROLE_SP1, beta_int)
        total_send += conner.record_send
        total_recv += conner.record_recv
        conner.CleanRecord()

