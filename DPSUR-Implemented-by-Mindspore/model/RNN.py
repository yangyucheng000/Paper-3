import mindspore.nn as nn
import mindspore.ops.operations as P

class RNN_Relu(nn.Cell):
    def __init__(self, max_words=20000, emb_size=100):
        super(RNN_Relu, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.embedding = nn.Embedding(max_words, emb_size)
        self.fc0 = nn.Dense(emb_size, 32)
        self.rnn = nn.LSTM(32, 32, num_layers=1, bidirectional=True, has_bias=True)
        self.fc1 = nn.Dense(64, 16)
        self.fc2 = nn.Dense(16, 2)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.embedding(x)
        x = self.fc0(x)
        x = self.relu(x)
        x, _ = self.rnn(x)
        x = P.ReduceMean()(x, (1, 2))
        x = self.fc1(x)
        out = self.relu(x)
        pred = self.fc2(out)
        return pred


class RNN_Tanh(nn.Cell):
    def __init__(self, max_words=20000, emb_size=100):
        super(RNN_Tanh, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.embedding = nn.Embedding(max_words, emb_size)
        self.fc0 = nn.Dense(emb_size, 32)
        self.rnn = nn.LSTM(32, 32, num_layers=1, bidirectional=True, has_bias=True)
        self.fc1 = nn.Dense(64, 16)
        self.fc2 = nn.Dense(16, 2)
        self.tanh = nn.Tanh()

    def construct(self, x):
        x = self.embedding(x)
        x = self.fc0(x)
        x = self.tanh(x)
        x, _ = self.rnn(x)
        x = P.ReduceMean()(x, (1, 2))
        x = self.fc1(x)
        out = self.tanh(x)
        pred = self.fc2(out)
        return pred
