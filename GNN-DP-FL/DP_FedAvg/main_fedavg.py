import torch
import argparse
from FedSystem.FLCore.model.CNN import FedAvgCNN
from FedSystem.FLCore.server.serverfedavg import FedAvg

import numpy as np



torch.manual_seed(0)

def run(args):
    if "mnist" in args.dataset:
        args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)

    server = FedAvg(args)
    server.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    # global_rounds
    parser.add_argument('-gr', "--global_rounds", type=int, default=500,
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
    parser.add_argument('-clr', "--client_learning_rate", type=float, default=0.01,
                        help="client learning rate")
    
    # global learning rate
    parser.add_argument('-glr','--global_learning_rate',type = float,default= 0.01,
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
    
    # 
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")  # 客户端参加的比例(client drift程度)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable,using cpu")
        args.device = "cpu"

    

    run(args)
    



