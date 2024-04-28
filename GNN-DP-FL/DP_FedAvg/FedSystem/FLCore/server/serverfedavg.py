from FedSystem.FLCore.client.clientfedavg import clientAVG
from FedSystem.FLCore.server.serverbase import server

class FedAvg(server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientAVG)

    def train(self):

        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:  # 几轮测试一次全局模型
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

        print('')
        print('Finish')
        self.evaluate()
