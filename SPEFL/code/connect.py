import socket
import pickle
import struct
import SpeflGlobal

ROLE_SP0 = SpeflGlobal.ROLE_SP0
ROLE_SP1 = SpeflGlobal.ROLE_SP1
ROLE_CP = SpeflGlobal.ROLE_CP
ROLE_CLIENTS = SpeflGlobal.ROLE_CLIENTS

CP_IP  = SpeflGlobal.CP_IP
CP_PORT  = SpeflGlobal.CP_PORT
SP0_IP  = SpeflGlobal.SP0_IP
SP0_PORT  = SpeflGlobal.SP0_PORT
SP1_IP  = SpeflGlobal.SP1_IP
SP1_PORT  = SpeflGlobal.SP1_PORT

class connecter(object):
    '''
    网络连接器
    identity:角色,CP,SP或者用户
    sernum:用户编号
    sernum:用户总数量
    '''
    def __init__(self, identity, sernum, clientnum):
        self.record_flag = False
        self.record_send = 0
        self.record_recv = 0
        self.role = identity

        if identity == ROLE_CP:
            self.conn = {}
            sock = socket.socket()
            sock.bind((CP_IP, CP_PORT))
            sock.listen()
            for i in range(clientnum + 2):
                sour, addr = sock.accept()
                id = self.recv(sour)
                self.conn[id] = sour

        if identity == ROLE_SP0:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((CP_IP, CP_PORT))
            self.send(sock, 'id{}'.format(identity))
            self.conn = {}
            self.conn['id2'] = sock

            sock = socket.socket()
            sock.bind((SP0_IP, SP0_PORT))
            sock.listen()
            for i in range(clientnum + 1):
                sour, addr = sock.accept()
                id = self.recv(sour)
                self.conn[id] = sour

        if identity == ROLE_SP1:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((CP_IP, CP_PORT))
            self.send(sock, 'id{}'.format(identity))
            self.conn = {}
            self.conn['id2'] = sock

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SP0_IP, SP0_PORT))
            self.send(sock, 'id{}'.format(identity))
            self.conn['id0'] = sock

            sock = socket.socket()
            sock.bind((SP1_IP, SP1_PORT))
            sock.listen()
            for i in range(clientnum):
                sour, addr = sock.accept()
                id = self.recv(sour)
                self.conn[id] = sour

        if identity == ROLE_CLIENTS:
            self.conn = {}
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((CP_IP, CP_PORT))
            self.send(sock, 'id{}'.format(sernum+identity))
            self.conn['id2'] = sock

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SP0_IP, SP0_PORT))
            self.send(sock, 'id{}'.format(sernum+identity))
            self.conn['id0'] = sock

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SP1_IP, SP1_PORT))
            self.send(sock, 'id{}'.format(sernum+identity))
            self.conn['id1'] = sock

    def send(self, target, data):
        data_pickle = pickle.dumps(data)

        headers = {'data_size': len(data_pickle)}
        head_pickle = pickle.dumps(headers)

        target.send(struct.pack('i', len(head_pickle)))
        target.send(head_pickle)
        target.sendall(data_pickle)
        if self.record_flag == True:
            self.record_send += len(data_pickle)

    def recv(self, source):
        head = source.recv(4)
        head_pickle_len = struct.unpack('i', head)[0]

        head_pickle = pickle.loads(source.recv(head_pickle_len))
        data_len = head_pickle['data_size']

        recv_data = source.recv(data_len, socket.MSG_WAITALL)
        data = pickle.loads(recv_data)
        if self.record_flag == True:
            self.record_recv += data_len
        return data

    def StartRecord(self):
        self.record_flag = True

    def CleanRecord(self):
        self.record_send = 0
        self.record_recv = 0
