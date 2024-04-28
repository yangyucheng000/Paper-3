import torch.nn as nn
import torch
import numpy as np
# import time
import copy
import argparse
import sys

from FedSystem.FLCore.client.clientbase import Client

class clientDPFedAvg(Client):
    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)

        self.old_model = copy.deepcopy(args.model)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.client_learning_rate)

        self.dp_norm = args.dp_norm
        self.dp_sigma = args.dp_sigma


    def set_parameters(self, model):  # 覆盖model.parameters()的操作；是get/set这种类型的操作
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()  # 深拷贝

        for new_param, old_param in zip(model.parameters(), self.old_model.parameters()):
            old_param.data = new_param.data.clone()  # 深拷贝
    

    def train(self):

        print(self.id,end=' ')

        sys.stdout.flush()

        self.model.train()

        trainloader = self.load_train_data_minibatch(iterations=self.local_iterations,poisson=False)

        for _ in range(self.local_epochs):
            # x,y是一个batch
            i=0
            for x,y in trainloader:
                # print(i)
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                
                output = self.model(x)  
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i+=1
    
    def l2_norm_calculate(self):
        l2_norm = 0
        for params in self.model.parameters():
            if params.requires_grad:
                l2_norm += params.data.norm(2).item()**2
        return l2_norm** .5
    
    def clip(self,l2_norm):
        clip_coef = min(1.,self.dp_norm/(l2_norm + 1e-6))
        for params in self.model.parameters():
            if params.requires_grad:
                params.data.mul_(clip_coef)

    def add_noise(self,client_num):
        for params in self.model.parameters():
            params.data.add_(self.dp_norm*self.dp_sigma/(client_num ** .5+1e-6)*torch.randn_like(params.data))

    # 此函数运行结束后，model中的参数的就是要返回给server的梯度
    def dp_update(self,client_num):
        # 新旧模型参数相减，结果保存在新的模型中
        for params,old_params in zip(self.model.parameters(),self.old_model.parameters()):
            # params.data = (params.data - old_params.data).clone()
            if params.requires_grad:
                params.data.sub_(old_params.data)

        # 获取l2范数
        l2_norm = self.l2_norm_calculate()
        self.clip(l2_norm)
        self.add_noise(client_num)
        # print('finish')

        
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # global_rounds
    parser.add_argument('-gr', "--global_rounds", type=int, default=20,
                        help="Global Round in the DPFL")
    
    # local_iterations
    parser.add_argument('-li', "--local_iterations", type=int, default=1,
                        help="DP-FedSGD need li")
    
    # batch sample retio of Poisson sampling
    parser.add_argument('-bsr', "--batch_sample_ratio", type=float, default=0.05,
                        help="The ratio of Poisson sampling")
    
    # sigma
    parser.add_argument('-dps', "--dp_sigma", type=float, default=2.0)

    # 裁剪范数
    parser.add_argument('-dpn', "--dp_norm", type=float, default=0.1,
                        help = 'the norm of clip')
    
    # 数据集
    parser.add_argument('-data', "--dataset", type=str, default="mnist")  # mnsit, Cifar10, fmnist

    # algorithm
    parser.add_argument('-algo', "--algorithm", type=str, default="DPFL")

    # local_learning_rate
    # DPSGD 学习率在0.1这个级别，再低一个数量级acc升不了，再高一个数量级loss指数爆炸
    # 传统SGD 学习率取0.01/0.001这个级别，太高了不利于收敛
    parser.add_argument('-clr', "--client_learning_rate", type=float, default=0.5,
                        help="client learning rate")
    
    # global learning rate
    parser.add_argument('-glr','--global_learning_rate',type = float,default= 0.5,
                        help='glr may different with lr')
    
    # batch_size (DP相关的使用泊松采样，这个batch size用不上了)
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)

    # num_clients
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    
    # local_epochs
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    

    # device
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    
    # 分类任务的类别数
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    
    # 所用模型
    parser.add_argument('-m', "--model", type=str, default="cnn")

    # 经过多少轮进行一次测试
    parser.add_argument('-eg', "--eval_gap", type=int, default=10,
                        help="Rounds gap for evaluation")  
    
    
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")  # 客户端参加的比例(client drift程度)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable,using cpu")
        args.device = "cpu"


    client = clientDPFedAvg(args,0,0,0)
    client.train()
