import torch.nn as nn
import numpy as np
import time
from FedSystem.FLCore.client.clientbase import Client

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)

    def train(self):
        # print(f"Clinet {self.id} is training……")
        trainloader = self.load_train_data_minibatch(iterations=1,poisson=True)

        self.model.train()

        max_local_epochs = self.local_epochs

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
            # for x,y in trainloader[0]:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)  # 前向传播
                loss = self.loss(output, y)
                self.optimizer.zero_grad()  # 梯度缓存清零，以确保每个训练批次的梯度都是从头开始计算的
                loss.backward()  # 对损失值 `loss` 进行反向传播，计算模型参数的梯度
                # loss.backward有很多操作: 1.内部有逐样本求梯度 2.逐样本的梯度裁剪
                self.optimizer.step()
                
        # print(f"Clinet {self.id} had trained.")