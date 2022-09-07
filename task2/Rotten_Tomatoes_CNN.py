"""
    通过 CNN 对 Rotten Tomatoes review 做分类 ==> 注意下卷积核

    label:
    0 - 消极
    1 - 有点消极
    2 - 中性
    3 - 有点积极
    4 - 积极

"""
import math
import time

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# parameter
BATCH_SIZE = 3
USE_GPU = False
EPOCH = 100


class RTDataset(Dataset):
    def __init__(self):
        filename = 'Rotten_Tomatoes_dataset/train.tsv'
        # 提取数据
        data_all = pd.read_csv(filename, '\t')
        data_index = data_all.keys()  # ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']

        # 拿到所有的 Phrase 和 Sentiment
        self.phrases = list(data_all[data_index[2]])
        self.sentiments = list(data_all[data_index[3]])

        # 拿到词频的字典
        self.word_index_dict = self.get_word_dict()

        # 所有的phrase 转换成 数字向量
        self.phrases2num_list = self.phrase2num_all()

    def __getitem__(self, index):
        return self.phrases[index], self.sentiments[index]

    def __len__(self):
        return len(self.phrases)    # 156060

    def get_padding_len(self):
        phrase_len_list = []
        for phrase in self.phrases:
            phrase = phrase.split(' ')
            phrase_len_list.append(len(phrase))  # 仅仅只是为了将来找到最长的句子好做 padding 用的
        print(max(phrase_len_list))  # 最长的词向量，同时也是padding长度 ==> 52 (后面可以优化，选择一个折中的长度)
        return max(phrase_len_list)

    def get_word_dict(self):
        # 统计词频 (然后排序)
        word_num_dict = {}      # word_num_dict = { ('单词':个数), ... }
        for phrase in self.phrases:
            for word in phrase.split(' '):
                # word = word.lower()     # 变成小写
                if word not in word_num_dict:   # 统计所有的单词个数，组成字典(即 统计词频)
                    word_num_dict[word] = 1
                else:
                    word_num_dict[word] += 1
        word_num_dict = sorted(word_num_dict.items(), key=lambda x: x[1], reverse=True)

        print(len(word_num_dict))   # 16531 ==> 18227    也是将来Embedding层的input_size
        # print(word_num_dict)     # [('the', 51220), (',', 42006), ..., ]
        # print(word_num_dict[0], word_num_dict[1])   # ('the', 51220) (',', 42006)

        # 将排好序的字典 加索引 (其实就是数字化,为后面Embedding用)
        word_index_dict = dict()
        for index, word_num in enumerate(word_num_dict):
            word_index_dict[word_num[0]] = index

        # print(word_index_dict)      # {'the': 0, ',': 1, 'a': 2, 'of': 3, 'and': 4, ... ,}
        # print(word_index_dict['and'])   # 4
        # print(word_index_dict.keys())

        return word_index_dict

    # 将传进来的 phrase 转换成 数字
    def phrase2num_per(self, sentence):
        num_list = []
        for word in sentence.split(' '):
            if word in self.word_index_dict.keys():
                num_list.append(self.word_index_dict[word])
        return num_list

    # 将所有的phrase 转成数字
    def phrase2num_all(self):
        num_lists = []
        for phrase in self.phrases:
            phrase_num = self.phrase2num_per(phrase)
            num_lists.append(phrase_num)
        return num_lists


data_set = RTDataset()
word_dict = data_set.get_word_dict()

padding_len = data_set.get_padding_len()
train_size = int(len(data_set)*0.8)
val_size = int(len(data_set)*0.2)

train_samples, val_samples = torch.utils.data.random_split(data_set, [train_size, val_size])

train_loader = DataLoader(dataset=train_samples, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=val_samples, batch_size=BATCH_SIZE, shuffle=False)


class CNN_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Classifier, self).__init__()
        self.embedding = nn.Embedding(18227, 5)     # [vocab_size, embedding_size]
        self.conv1 = nn.Conv2d(1, 20, (2, 5))
        self.conv2 = nn.Conv2d(1, 20, (3, 5))
        self.conv3 = nn.Conv2d(1, 20, (4, 5))

        self.dropout = nn.Dropout(p=0.5)

        self.linear = nn.Linear(60, 5)

    def max_pooling(self, x, num):
        x = F.max_pool2d(x, (padding_len-num+1, 1))
        x = x.squeeze(3)
        return x

    def forward(self, x):
        # batch_size = x.size(0)      # [batch_size, seq_len]
        x = self.embedding(x)       # 将 [batch_size, seq_len] ==> [batch_size, seq_len, embedding_size]
        x = x.unsqueeze(1)      # add channel(=1) become [batch, channel(=1), seq_len, embedding_size]
        x1 = self.max_pooling(F.relu(self.conv1(x)), 2).squeeze(2)
        x2 = self.max_pooling(F.relu(self.conv2(x)), 3).squeeze(2)
        x3 = self.max_pooling(F.relu(self.conv3(x)), 4).squeeze(2)
        output = torch.cat([x1, x2, x3], dim=1)
        # output = output.view(-1, 60)
        return self.linear(output)


model = CNN_Classifier()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def create_tensor(tensor):
    if USE_GPU:
        devices = torch.device('cuda:0')
        tensor = tensor.to(devices)
    return tensor


def change_phrase2num(phrase):
    word_num_list = []
    for word in phrase.split(' '):
        if word in word_dict.keys():
            word_num_list.append(word_dict[word])
    return word_num_list


# 这里处理的数据量 是一个batch_size; 先拿数据，再padding; pytorch中有自动padding的 pad_sequence()
def make_tensor(phrases, sentiments):
    phrase_sequence = [change_phrase2num(phrase) for phrase in phrases]
    phrase_len = [len(phrase) for phrase in phrase_sequence]
    sentiments = sentiments.long()

    # padding
    # seq_tensor = torch.zeros([len(phrase_sequence), max(phrase_len)]).long()      # 这个好一点，在同一个batch里面的padding一样
    seq_tensor = torch.zeros([len(phrase_sequence), padding_len]).long()
    for index, (p_seq, p_len) in enumerate(zip(phrase_sequence, phrase_len)):
        seq_tensor[index, :p_len] = torch.LongTensor(p_seq)

    return create_tensor(seq_tensor), create_tensor(sentiments)


def time_since(start):
    second = time.time() - start
    minute = math.floor(second / 60)
    second -= minute * 60
    return '%dm %ds' % (minute, second)


def train(epoch, start):
    total_loss = 0
    for i, (phrase, sentiment) in enumerate(train_loader, 1):

        inputs, labels = make_tensor(phrase, sentiment)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f'[{time_since(start)}]  Epoch: {epoch}', end='')
            print(f'[{i * len(inputs)} / {train_size}]', end='')
            print(f'[loss: {total_loss / i * len(inputs)}]')
    return total_loss


def test():
    correct = 0
    totals = val_size
    for i, (phrases, sentiments) in enumerate(test_loader, 1):
        inputs, labels = make_tensor(phrases, sentiments)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)    # outputs.shape ==> (?, 5)
        correct += (predicted == labels).sum().item()

    percent = '%2f %%' % (100*correct/totals)
    print(f'Test set: Accuracy {correct} / {totals} {percent}')

    return correct / totals


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

