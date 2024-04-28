import os
import argparse
from tqdm import tqdm
import numpy as np
import cppimport
import cppimport.import_hook

from myMPC import calculator
from myThread import myThread
from connect import connecter
import SpeflGlobal

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-nc', '--num_of_clients', type=int, default=40, help='numer of the clients')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-id', '--role', type=int, help='identity of SP, 0 or 1')

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
    print("SP start")
    args = parser.parse_args()
    args = args.__dict__

    selfrole = args['role']
    otherrole = 1 - selfrole
    
    num_clients = int(args['num_of_clients'])
    clients_in_comm = ['client{}'.format(i) for i in range(num_clients)]
    conner = connecter(selfrole, 0, num_clients)
    mpcer = calculator(conner, m)

    #发送初始参数
    parmlist_share = mpcer.share_recv(ROLE_CP)
    thread = []
    for i in range(num_clients):
        thread.append(myThread(mpcer, i, 'restruct_send', parmlist_share))
        thread[i].start()
    for i in range(num_clients):
        thread[i].join()

    history_updata = None

    conner.StartRecord()
    total_send = 0
    total_recv = 0

    for i in tqdm(range(args['num_comm'])):
        #接收乘法三元组
        mask = mpcer.getmask_test(selfrole, num_clients, len(parmlist_share))
        G_share = {}
        R_share = {}
        garble = None
        garble_v = None

        #接收用户梯度共享
        thread = []
        for j in range(num_clients):
            thread.append(myThread(mpcer, j, 'share_recv', None))
            thread[j].start()
        for j in range(num_clients):
            thread[j].join()
            G_share_v = thread[j].getresult()
            G_share['client{}'.format(j)] = G_share_v

        #梯度加混淆后发送给CP
        for j in range(num_clients):
            G_share_v = G_share['client{}'.format(j)]
            if garble == None:
                garble_v = m.get_garble(len(G_share_v))
                garble = list(garble_v)
            R_share_v = m.vector_add(m.VectorInt(G_share_v), garble_v)
            R_share['client{}'.format(j)] = list(R_share_v)
            mpcer.restruct_send(ROLE_CP, list(R_share_v))

        #梯度中位数去混淆
        Rmed_share_v = mpcer.share_recv(ROLE_CP)
        Gmed_share = list(m.vector_sub(m.VectorInt(Rmed_share_v), garble_v))

        #计算相关系数公式分子部分
        G_submean = []
        for client in clients_in_comm:
            mea = int(np.mean(G_share[client]))
            G_submean.extend(list(m.vecnum_sub(m.VectorInt(G_share[client]), mea)))
        
        mea = int(np.mean(Gmed_share))
        Gmed_submean = list(m.vecnum_sub(m.VectorInt(Gmed_share), mea))
        E = mpcer.vecmat_mul_partner(otherrole, Gmed_submean, G_submean, mask[0]['Med_mask'], mask[0]['G_mask'], mask[0]['M_G_mask'])
        mpcer.restruct_send(ROLE_CP, E)

        #计算梯度中位数标准差
        sigmaM = mpcer.vector_squ_partner(otherrole, Gmed_submean, mask[0]['Med_mask'], mask[0]['M_M_mask'])
        sigmaM_share = []
        sigmaM_share.append(sigmaM)
        mpcer.restruct_send(ROLE_CP, sigmaM_share)

        #接收权重并聚合
        beta_share = mpcer.share_recv(ROLE_CP)
        G_T_share = []
        for k in range(len(parmlist_share)):
            for j in range(num_clients):
                G_T_share.append(G_share['client{}'.format(j)][k])
        agg = mpcer.vecmat_mul_partner(otherrole, beta_share, G_T_share, mask[0]['B_mask'], mask[0]['G_mask'], mask[0]['B_G_mask'])
        thread = []
        for j in range(num_clients):
            thread.append(myThread(mpcer, j, 'restruct_send', agg))
            thread[j].start()
        for j in range(num_clients):
            thread[j].join()

        total_send += conner.record_send
        total_recv += conner.record_recv
        conner.CleanRecord()

