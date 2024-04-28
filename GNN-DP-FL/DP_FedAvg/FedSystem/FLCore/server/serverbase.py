import numpy as np
from FedSystem.utils.data_utils import read_client_data
import copy
from random import sample

class server():
    def __init__(self,args):
        self.args = args
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.global_learning_rate
        self.global_model = copy.deepcopy(args.model)  # 深拷贝
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio  # Ratio of clients per round
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm

        self.clients = []
        self.selected_clients = []

        self.rs_test_acc = []  # result
        self.rs_test_auc = []
        self.rs_train_loss = []
        
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.eval_gap = args.eval_gap
    
    def set_clients(self, clientObj):  # clientObj是一个类对象了
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,id=i,train_samples=len(train_data),test_samples=len(test_data))
            self.clients.append(client)
            self.clients[i].train_samples = len(train_data)  # 按数据集大小来决定权重

    def select_clients(self):
        if self.random_join_ratio:  # 这组判断确定current_num_join_clients（数量）
            self.current_num_join_clients = \
                np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        selected_clients.sort(key=lambda x: x.id)

        return selected_clients  
    
    def send_models(self):  # sever->client
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)      
    
    def receive_models(self):  # client->server
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):  # 权重归一化
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():  # 全局模型置为0，方便后面累加了，上一行深拷贝只是为了model shape
            param.data.zero_()
        
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def test_metrics(self):  # 中心方拿到准确率的方式是从客户端这里拿，拿到以后加权
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns = c.test_metrics()
            tot_correct.append(ct * 1.0)
            # tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        # print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    
    
    