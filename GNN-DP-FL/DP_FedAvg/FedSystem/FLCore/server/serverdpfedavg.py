import torch.nn as nn
import torch
import copy
from FedSystem.FLCore.server.serverbase import server
from FedSystem.FLCore.client.clientdpfedavg import clientDPFedAvg
import sys

class serverDPFedAvg(server):
    def __init__(self,args):
        super().__init__(args)

        self.set_clients(clientDPFedAvg)

        # self.loss_fn = nn.CrossEntropyLoss()
        self.global_learning_rate = args.global_learning_rate

    def aggregate_parameters(self,client_model):
        tmp_model = copy.deepcopy(client_model[0])
        for client in client_model[1:]:
            for tmp_params,params in zip(tmp_model.parameters(),client.parameters()):
                tmp_params.data.add_(params.data)
        # 除以client数量,乘以学习率
        for tmp_params in tmp_model.parameters():
            tmp_params.data.mul_(self.global_learning_rate/(self.num_join_clients))
        
        for params,tmp_params in zip(self.global_model.parameters(),tmp_model.parameters()):
            # # 这种方式不同于论文中的
            # # 给之前的参数和本轮的参数加了一个权重
            # tmp_params.data.mul_(0.99)
            # params.data.mul_(0.01)
            params.data.add_(tmp_params.data)
            # params.data.add_(tmp_params.data)
        

    
    def train(self):

        # self.send_models()

        for i in range(self.global_rounds+1):

            print('global round:',i,'trained client: ',end = '') 
            sys.stdout.flush()

            self.selected_clients = self.select_clients()
            # self.selected_clients = copy.deepcopy(self.clients)

            client_model = []
            for j,client in enumerate(self.selected_clients):
                client.train()

                if j == len(self.selected_clients)-1:
                    print('')

                client.dp_update(self.num_join_clients)

                client_model.append(copy.deepcopy(client.model))
            
            self.aggregate_parameters(client_model)
            self.send_models()

            if i%self.eval_gap==0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model by global")
                self.evaluate()



    
