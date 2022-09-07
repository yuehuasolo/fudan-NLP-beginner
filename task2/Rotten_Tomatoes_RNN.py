"""
    通过 RNN 对 Rotten Tomatoes review 做分类

    label:
    0 - 消极
    1 - 有点消极
    2 - 中性
    3 - 有点积极
    4 - 积极

"""
import math
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence     # 优化GRU的输入，把非0的都打包起来
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


BATCH_SIZE = 256
N_CHAR = 128    # input_size
HIDDEN_SIZE = 100
EPOCH = 100
N_LAYERS = 1
USE_GPU = False


class ReviewDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'Rotten_Tomatoes_dataset/train.tsv' if is_train_set else 'Rotten_Tomatoes_dataset/test.tsv'
        data_all = pd.read_csv(filename, sep='\t')
        data_index = data_all.keys()     # ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']

        self.phrases = list(data_all[data_index[2]])
        self.sentiments = list(data_all[data_index[3]])

        self.sentiments_list = sorted(set(self.sentiments))  # [0, 1, 2, 3, 4]
        self.sentiments_list_len = len(self.sentiments_list)  # 分类的个数

        self.sentiments_len = len(self.sentiments)  # 156060
        self.phrases_len = len(self.phrases)    # 156060
        # print(self.phrases_len, self.sentiments_len)

    # 实例化之后获得的是哪些数据
    def __getitem__(self, index):
        return self.phrases[index], self.sentiments[index]

    def __len__(self):
        return self.phrases_len  # 或者返回 self.sentiments 也是一样的，都只是为了确定数据 长度而已; 但测试集没有，会报错

    # 分类个数
    def get_sentiments_num(self):
        return self.sentiments_list_len


train_set = ReviewDataset(is_train_set=True)
# 在训练集中划分验证集合
# num_train_samples = int(0.8 * len(train_set))
# train_samples = SubsetRandomSampler(torch.arange(0, num_train_samples))
# num_val_samples = int(0.2 * len(train_set))
# val_samples = SubsetRandomSampler(torch.arange(num_train_samples, num_train_samples + num_val_samples))
train_size = int(len(train_set) * 0.8)
val_size = int(len(train_set) * 0.2)

train_samples, val_samples = torch.utils.data.random_split(train_set, [train_size, val_size])
train_loader = DataLoader(dataset=train_samples, batch_size=BATCH_SIZE, shuffle=False)
sentiments_len = train_set.get_sentiments_num()

# test_set = ReviewDataset(is_train_set=True)
test_loader = DataLoader(dataset=val_samples, batch_size=BATCH_SIZE, shuffle=False)


# Model
class RNN_Classifier(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size, n_layers, bidirectional=True):
        super(RNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, hidden_size)      # 输入所在词典的长度; 需要映射成几维(作为下一次输入用，故看下一次输入的大小)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional, dropout=0.5)

        self.linear = nn.Linear(hidden_size * self.n_direction, output_size)

        self.dropout = nn.Dropout(p=0.5)    # 可用可不用

    #   **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_direction, batch_size, self.hidden_size)
        return create_tensor(hidden)

    # GRU网络的计算
    def forward(self, inputs, seq_len):
        # embedding 需要的 输入格式 是(seq_len, batch); 输出后的格式为 (seq_len, batch, hidden_size); hidden_size是embedding后想要映射的维度
        inputs = inputs.t()     # 传入的输入x是 batch x seq_len; ==> 转置 seq_len x batch
        batch_size = inputs.size(1)

        # GRU 计算需要的输入
        hidden_0 = self.init_hidden(batch_size)
        embedding = self.embedding(inputs)

        gru_input = pack_padded_sequence(embedding, seq_len)    # seq_len是排好序的所有输入的 list(tensor)

        outputs, hidden = self.gru(gru_input, hidden_0)

        if self.n_direction == 2:
            hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)
        else:
            hidden_cat = hidden[0]

        linear_outputs = self.linear(hidden_cat)
        return linear_outputs


model = RNN_Classifier(N_CHAR, sentiments_len, HIDDEN_SIZE, N_LAYERS)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def phrase2ascii_list(phrase):
    arr = [ord(c) for c in phrase]
    return arr, len(arr)


def create_tensor(tensor):
    if USE_GPU:
        devices = torch.device('cuda:0')
        tensor = tensor.to(devices)
    return tensor


def make_tensor(phrases, sentiments):
    # 处理数据
    sequence_and_len = [phrase2ascii_list(phrase) for phrase in phrases]
    phrase_sequence = [s1[0] for s1 in sequence_and_len]
    seq_len = torch.LongTensor([s1[1] for s1 in sequence_and_len])
    sentiments = sentiments.long()

    # 转成同样长度的 tensors; 长度不够的要padding到一样的长度
    seq_tensor = torch.zeros([len(phrase_sequence), seq_len.max()]).long()
    for index, (seq, s_len) in enumerate(zip(phrase_sequence, seq_len)):
        seq_tensor[index, :s_len] = torch.LongTensor(seq)

    # 排序
    seq_len, seq_index = seq_len.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[seq_index]
    sentiments = sentiments[seq_index]

    return create_tensor(seq_tensor), seq_len, create_tensor(sentiments)


def time_since(since):
    second = time.time() - since
    minute = math.floor(second / 60)
    second -= minute * 60
    return '%dm %ds' % (minute, second)


def train(epoch, start):
    total_loss = 0
    for i, (phrases, sentiments) in enumerate(train_loader, 1):
        inputs, seq_len, labels = make_tensor(phrases, sentiments)
        optimizer.zero_grad()

        outputs = model(inputs, seq_len)    # 需要seq_len序列是为了优化gru输出

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch: {epoch}', end='')
            print(f'[{i * len(inputs)} / {train_size}]', end='')  # len(train_set) 中的len()是重载的
            print(f'loss={total_loss / (i * len(inputs))}')

    return total_loss


def test():
    corrects = 0
    totals = val_size
    with torch.no_grad():
        for i, (phrases, sentiments) in enumerate(test_loader, 1):
            inputs, seq_len, labels = make_tensor(phrases, sentiments)
            outputs = model(inputs, seq_len)
            _, predicted = torch.max(outputs, dim=1)
            corrects += (predicted == labels).sum().item()

        percent = '%.2f %%' % (100*corrects/totals)
        print(f'Test set: Accuracy {corrects} / {totals} {percent}')

    return corrects / totals


if __name__ == '__main__':
    start_time = time.time()
    print('Training for %d epochs.. \n' % EPOCH)
    acc_list = []
    for epoch in range(1, EPOCH + 1):
        train(epoch, start_time)
        acc = test()
        acc_list.append(acc)
        print('\n')

    x = np.arange(1, len(acc_list)+1, 1)
    y = np.array(acc_list)
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()  # 画格子
    plt.show()


torch.nn.ReLU()