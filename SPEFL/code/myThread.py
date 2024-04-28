import threading
ROLE_SP0 = 0
ROLE_SP1 = 1
ROLE_CP = 2
ROLE_CLIENTS = 3
class myThread (threading.Thread):
    def __init__(self, mpc, num, task, data):
        threading.Thread.__init__(self)
        self.m = mpc
        self.task = task
        self.num = num
        self.data = data

    def run(self):
        if self.task == 'share_recv':
            self.result = self.m.share_recv(ROLE_CLIENTS + self.num)
        if self.task == 'get_data':
            self.result = self.m.share_recv(ROLE_CLIENTS + self.num)
        if self.task == 'restruct_send':
            self.m.restruct_send(ROLE_CLIENTS + self.num, self.data)
        if self.task == 'share_recv_sec':
            self.result = self.m.share_recv_sec(ROLE_CLIENTS + self.num)
    def getresult(self):
        return self.result  # todo: 'myThread' object has no attribute 'result'