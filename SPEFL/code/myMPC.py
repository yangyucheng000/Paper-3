import pickle
import numpy as np
import random
import time
import SpeflGlobal

ROLE_SP0 = SpeflGlobal.ROLE_SP0
ROLE_SP1 = SpeflGlobal.ROLE_SP1
ROLE_CP = SpeflGlobal.ROLE_CP
ROLE_CLIENTS = SpeflGlobal.ROLE_CLIENTS
prec = SpeflGlobal.prec

class calculator(object):
    '''
    隐私计算器
    conn:网络连接器
    m:cpp代码执行器
    '''
    def __init__(self, conn, m):
        self.conn = conn
        self.m = m

    #将data秘密共享给target0, target1
    def share_send(self, target0, target1, data):
        grab = self.m.get_garble(len(data))
        share0 = list(grab)
        share1 = list(self.m.vector_sub(self.m.VectorInt(data),grab))
        self.conn.send(self.conn.conn['id{}'.format(target0)], share0)
        self.conn.send(self.conn.conn['id{}'.format(target1)], share1)

    #从source接收秘密共享
    def share_recv(self, source):
        return self.conn.recv(self.conn.conn['id{}'.format(source)])

    #向target发送需要重建的共享
    def restruct_send(self, target, data):
        self.conn.send(self.conn.conn['id{}'.format(target)], data)

    #从source0, source1接收秘密共享并重建
    def restruct_recv(self, source0, source1):
        share0 = self.conn.recv(self.conn.conn['id{}'.format(source1)])
        share1 = self.conn.recv(self.conn.conn['id{}'.format(source0)])
        return list(self.m.vector_add(self.m.VectorInt(share0), self.m.VectorInt(share1)))
    
    '''
    与另一参与方一起计算向量X的内积
    another:另一方的角色编号
    X_share:向量X的本地秘密共享
    X_mask, res_mask:隐私计算掩码
    '''
    def vector_squ_partner(self, another, X_share, X_mask, res_mask):
        E_share = list(self.m.vector_sub(self.m.VectorInt(X_share), self.m.VectorInt(X_mask)))
        
        if another == 0:
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
        else:
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)

        E = self.m.vector_add(self.m.VectorInt(E_share), self.m.VectorInt(E_share_))

        temp = []
        temp.append(int(self.m.vector_mul(self.m.VectorInt(X_share), E)))
        res = []
        res.append(temp[0])
        res = self.m.vector_add(self.m.VectorInt(res), self.m.VectorInt(temp))
        res = self.m.vector_add(res, self.m.VectorInt(res_mask))
        if another == 0:
            temp = []
            temp.append(int(self.m.vector_mul(E, E)))
            res = self.m.vector_sub(res, self.m.VectorInt(temp))

        return list(res)[0]
    
    '''
    与另一参与方一起计算向量X与矩阵Y的乘积
    another:另一方的角色编号
    X_share:X的本地秘密共享
    Y_share:Y的本地秘密共享
    X_mask, Y_mask, res_mask:隐私计算三元组
    '''
    def vecmat_mul_partner(self, another, X_share, Y_share, X_mask, Y_mask, res_mask):
        A_share = X_mask
        B_share = Y_mask
        C_share = res_mask

        E_share = list(self.m.vector_sub(self.m.VectorInt(X_share), self.m.VectorInt(A_share)))
        F_share = list(self.m.vector_sub(self.m.VectorInt(Y_share), self.m.VectorInt(B_share)))
        if another == 0:
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
        else:
            E_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            F_share_ = self.conn.recv(self.conn.conn['id{}'.format(another)])
            self.conn.send(self.conn.conn['id{}'.format(another)], E_share)
            self.conn.send(self.conn.conn['id{}'.format(another)], F_share)

        E = (self.m.vector_add(self.m.VectorInt(E_share), self.m.VectorInt(E_share_)))
        F = (self.m.vector_add(self.m.VectorInt(F_share), self.m.VectorInt(F_share_)))

        res = self.m.vecmat_mul(self.m.VectorInt(X_share), F)
        temp = self.m.vecmat_mul(E, self.m.VectorInt(Y_share))
        res = self.m.vector_add(res, temp)
        res = self.m.vector_add(res, self.m.VectorInt(C_share))
        if another == 0:
            temp = self.m.vecmat_mul(E, F)
            res = self.m.vector_sub(res, temp)
        return list(res)

    #生成乘法三元组
    def getmask_test(self, role, num_clients, grad_len):
        res = []
        if role == ROLE_CP:
            Med_mask = list(self.m.get_garble(grad_len))
            G_mask = list(self.m.get_garble(grad_len * num_clients))
            M_G_mask = []
            M_G_mask.extend(list(self.m.vecmat_mul(self.m.VectorInt(Med_mask), self.m.VectorInt(G_mask))))
            M_M_mask = []
            M_M_mask.append(self.m.vector_mul(self.m.VectorInt(Med_mask), self.m.VectorInt(Med_mask)))
            G_G_mask = []
            for i in range(num_clients):
                G_single = G_mask[i*grad_len:(i+1)*grad_len]
                G_G_mask.append(self.m.vector_mul(self.m.VectorInt(G_single), self.m.VectorInt(G_single)))

            B_mask = list(self.m.get_garble(num_clients))
            B_G_mask = []
            B_G_mask.extend(list(self.m.vecmat_mul(self.m.VectorInt(B_mask), self.m.VectorInt(G_mask))))

            self.share_send(ROLE_SP0, ROLE_SP1, Med_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, M_G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, M_M_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, G_G_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, B_mask)
            self.share_send(ROLE_SP0, ROLE_SP1, B_G_mask)

        else:
            temp = {}
            temp['Med_mask'] = self.share_recv(ROLE_CP)
            temp['G_mask'] = self.share_recv(ROLE_CP)
            temp['M_G_mask'] = self.share_recv(ROLE_CP)
            temp['M_M_mask'] = self.share_recv(ROLE_CP)
            temp['G_G_mask'] = self.share_recv(ROLE_CP)
            temp['B_mask'] = self.share_recv(ROLE_CP)
            temp['B_G_mask'] = self.share_recv(ROLE_CP)

            res.append(temp)
        return res
