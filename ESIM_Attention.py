"""
任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考ESIM（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

1.参考
    1.1 《神经网络与深度学习》 第7章
    1.2 Reasoning about Entailment with Neural Attention https://arxiv.org/pdf/1509.06664v1.pdf
    1.3 Enhanced LSTM for Natural Language Inference https://arxiv.org/pdf/1609.06038v3.pdf

2.数据集：https://nlp.stanford.edu/projects/snli/

3.实现要求：Pytorch

4.知识点：
    4.1 注意力机制
    4.2 token2token attetnion
"""
import math
import re
import time

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt

BATCH_SIZE = 256
EPOCH = 100
snli_path = '../input/estmdata/snli_1.0_dev.txt'
glove_path = '../input/estmdata/glove.6B.50d.txt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 获取训练数据 和 后面需要的glove预训练向量字典
def data_process(snli_path, glove_path):
    with open(snli_path, 'r') as f:
        snli_lines = f.readlines()  # 将每一行变成list中的一个元素
    snli_data = snli_lines[1:]

    with open(glove_path, 'r') as f:
        glove_lines = f.readlines()
    glove_dict = dict()
    for i in range(len(glove_lines)):
        line = glove_lines[i].split()  # list类型，总长度是51，0位是单词
        glove_dict[line[0]] = [float(line[j]) for j in range(1, 51)]
    return snli_data, glove_dict


class snliDataset(Dataset):
    def __init__(self, sentence1, sentence2, labels):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.labels = labels

    def __getitem__(self, index):
        return self.sentence1[index], self.sentence2[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# 处理一个batch大小的数据格式, 最后都会padding到同样的大小
def collate_fn(batch_data):
    sentence1, sentence2, labels = zip(*batch_data)
    sentence1 = [torch.LongTensor(sen) for sen in sentence1]
    padding_sen1 = pad_sequence(sentence1, padding_value=0, batch_first=True)
    sentence2 = [torch.LongTensor(sen) for sen in sentence2]
    padding_sen2 = pad_sequence(sentence2, padding_value=0, batch_first=True)
    return torch.LongTensor(padding_sen1), torch.LongTensor(padding_sen2), torch.LongTensor(labels)


class Random_Embedding:
    """
        -- 将数据文本转换成embedding(即 数字向量化)
        step:  筛选数据，获取字典(不考虑词频)，数字向量化
    """

    def __init__(self, data):
        self.word_dict = dict()
        self.dict_len = 0
        self.pad_len = 0  # 记录最长的句子，后面padding用的
        temp_data = [item.split('\t') for item in data]
        self.data = [[item[5], item[6], item[0]] for item in temp_data]  # sentence1, sentence2, gold_label
        self.data.sort(key=lambda x: len(x[0].split()))  # 根据item[5] 即sentence1 单词数量，升序排序
        self.label_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.label_word2vec = [self.label_dict[item[2]] for item in self.data]
        self.sentence1_word2vec = list()
        self.sentence2_word2vec = list()

    # 获取词典(不考虑词频，之前做CNN分类考虑了，这里可以后续再比较)
    def get_words_dict(self):
        pattern = '[A-Za-z\']+'  # 匹配前面的表达式一次或者多次
        for item in self.data:
            for i in range(2):
                sentence = item[i]
                sentence = sentence.lower()  # 全部统一小写，否则同一个词最后会导致 不一样的embedding
                words = re.findall(pattern, sentence)  # 找到句子中所有单词，等价于 words = [sen.split() for sen in sentence]
                for word in words:  # 此处后续可以改为比较词频顺序再做实验比较
                    if word not in self.word_dict:
                        self.word_dict[word] = len(self.word_dict) + 1  # 在字典的最后加上
        self.dict_len = len(self.word_dict)  # 所有句子遍历完后，确定最终字典的长度

    # 根据上面求出的字典，求出sentence的 word2vec
    def get_index(self):
        pattern = '[A-Za-z\']+'
        for item in self.data:
            sentence1 = item[0]
            sentence1 = sentence1.lower()
            words = re.findall(pattern, sentence1)
            self.sentence1_word2vec.append([self.word_dict[word] for word in words])  # 转换为数字向量
            self.pad_len = max(self.pad_len, len(words))  # 记录当前最长的句子长度

            sentence2 = item[1]
            sentence2 = sentence2.lower()
            words = re.findall(pattern, sentence2)
            self.sentence2_word2vec.append([self.word_dict[word] for word in words])
            self.pad_len = max(self.pad_len, len(words))  # 在sentence2中再次比较，确保全局最长
        self.dict_len += 1


# 预训练的embedding
class Glove_Embedding:
    def __init__(self, data, trained_dict):
        self.word_dict = dict()
        self.dict_len = 0
        self.pad_len = 0
        self.glove_dict = trained_dict  # （新增）
        temp_data = [item.split('\t') for item in data]  # 拆分了每一个句子的单词
        self.data = [[item[5], item[6], item[0]] for item in temp_data]  # 选取sentence1, sentence2, gold_label
        self.data.sort(key=lambda x: len(x[0].split()))
        self.label_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.label_word2vec = [self.label_dict[item[2]] for item in self.data]
        self.sentence1_word2vec = list()
        self.sentence2_word2vec = list()
        self.embedding = list()  # （新增）存放的是glove预训练模型中的参数; 后续训练模型时候可以直接传进去

    def get_word_dict(self):
        self.embedding.append([0] * 50)
        for item in self.data:
            for i in range(2):
                sentence = item[i]
                sentence = sentence.lower()
                words = sentence.split()
                for word in words:
                    if word not in self.word_dict:
                        self.word_dict[word] = len(self.word_dict) + 1
                        if word in self.glove_dict:
                            self.embedding.append(self.glove_dict[word])
                        else:
                            self.embedding.append([0] * 50)
        self.dict_len = len(self.word_dict)

    def get_index(self):
        for item in self.data:
            sentence1 = item[0]
            sentence1 = sentence1.lower()
            words = sentence1.split()
            self.sentence1_word2vec.append([self.word_dict[word] for word in words])
            self.pad_len = max(self.pad_len, len(words))

            sentence2 = item[1]
            sentence2 = sentence2.lower()
            words = sentence2.split()
            self.sentence2_word2vec.append([self.word_dict[word] for word in words])
            self.pad_len = max(self.pad_len, len(words))
        self.dict_len += 1


# 处理拿到的dataloader， 划分 训练集 和 验证集
def get_process_dataloader(dataset):
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.2)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, val_dataloader


snli_data, glove_dict = data_process(snli_path, glove_path)

# ====================
# random_embedding 数据
random_emb = Random_Embedding(snli_data)
random_emb.get_words_dict()
random_emb.get_index()

random_vocab_size = random_emb.dict_len  # 字典长度，建立模型的时候里面的建立 embedding参数需要用到
random_s1 = random_emb.sentence1_word2vec
random_s2 = random_emb.sentence2_word2vec
random_y = random_emb.label_word2vec
random_dataset = snliDataset(random_s1, random_s2, random_y)
random_train_dataloader, random_val_dataloader = get_process_dataloader(random_dataset)

# ====================
# glove_embedding 数据
glove_emb = Glove_Embedding(snli_data, glove_dict)
glove_emb.get_word_dict()
glove_emb.get_index()

glove_vocab_size = glove_emb.dict_len  # 字典长度，建立模型的时候里面的建立 embedding参数需要用到
weight = glove_emb.embedding  # glove的预训练数据
glove_s1 = glove_emb.sentence1_word2vec
glove_s2 = glove_emb.sentence2_word2vec
glove_y = glove_emb.label_word2vec
glove_dataset = snliDataset(glove_s1, glove_s2, glove_y)
glove_train_dataloader, glove_val_dataloader = get_process_dataloader(glove_dataset)

"""
    准备ESIM(Enhanced Sequential Inference Model)的四个核心部分:
        step1: ==> Input Encoding
        step2: ==> Local Inference Modeling
        step3: ==> Inference Composition
        step4: ==> Predicted

"""


# 核心就是将 输入x ==> wordEmbedding ==> BiLSTM
class Input_Encoding(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, weight=None, num_layers=1, bidirectional=True):
        super(Input_Encoding, self).__init__()
        self.input_size = input_size  # 一个单词对应的特征个数
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.5).to(device)
        # self.num_direction = 2 if bidirectional else 1

        if weight is None:
            # 没有预训练好的参数，就随机初始化
            x = nn.init.xavier_normal_(torch.Tensor(vocab_size, input_size))
            self.embedding = nn.Embedding(vocab_size, input_size, _weight=x).to(device)
        else:
            self.embedding = nn.Embedding(vocab_size, input_size, _weight=weight).to(device)

        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional, batch_first=True).to(device)

    # Input_Encoding 模块核心，先将x进行embedding再通过bilstm后得到的有序信息的输出作为 新的x
    def forward(self, x):
        x = torch.LongTensor(x).to(device)  # 传进来的x.shape [seq_len, batch, input_size], 但是batch应该在第0位，这里为了好理解
        x = self.embedding(x)
        x = self.dropout(x)
        self.bilstm.flatten_parameters()
        x, _ = self.bilstm(x)  # _ 是 (h_n, c_n)， lstm返回的隐层状态 和 序列信息状态
        #         print('input encoding x', x.shape)
        return x  # **output** of shape `(seq_len, batch, num_directions * hidden_size), 但是这里batch会在 0位置


# 通过Attention Mechanism 来计算两个句子之间的序列关系
class Local_Inference_Modeling(nn.Module):
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1).to(device)
        self.softmax_2 = nn.Softmax(dim=2).to(device)

    def forward(self, a_bar, b_bar):
        """
            :param a_bar: [batch, seq_len_a, hidden_a]
            :param b_bar: 转置 b_bar_T: [batch, hidden_b, seq_len_b]
            :return:

            计算attention打分函数(点积模型), 为两句话之间分析用; 比如(1,1)和(1,2)的关系 比(1,0) 和 (0,1)的紧密,前者内积大，后者正交内积为0
            (这里计算 e 时候的 seq_len是不带softmax信息的)
        """
        e = torch.matmul(a_bar, b_bar.transpose(1, 2))  # [batch, seq_len_a, seq_len_b] ,seq_len 大小是一样的
        #         print('e', e.shape)

        """
            -- 分析两个句子之间的联系; 
            -- softmax(e)是标准化后的权重，然后乘以另一句话中每个单词特征来表示当前单词特征的可能性
            -- [batch, seq_len_a, num_direction * hidden_b], 这里的seq_len_a已经带有了 softmax信息
        """
        a_tilde = self.softmax_2(e).bmm(b_bar)  # 对e里面的b_bar的 seq_len用softmax，后续才知道b_bar哪些重要
        b_tilde = self.softmax_1(e).transpose(1, 2).bmm(a_bar)  # [batch, seq_len_b, hidden_b * num_directions]
        #         print('a_tilde', a_tilde.shape)
        #         print('b_tilde', b_tilde.shape)

        # 对加权编码的值和原本输入的编码值进行比较，做差异性计算，计算新旧序列之间的差和积，把所有信息拼接起来
        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde],
                        dim=-1)  # [batch, seq_len, 4 * num_directions * hidden]
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1)
        #         print('m_a', m_a.shape)

        return m_a, m_b


# 通过上面的 Local Inference Modeling 拿到的全局信息通过 BiLSTM捕获局部推理信息和上下文信息
class Inference_Composition(nn.Module):
    def __init__(self, input_size, m_hidden_size, hidden_size, num_layers=1):
        super(Inference_Composition, self).__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True).to(device)
        self.dropout = nn.Dropout(p=0.5).to(device)
        self.linear = nn.Linear(m_hidden_size, input_size).to(device)  # 做线性变换是为了数据的输入格式

    def forward(self, m_a, m_b):
        temp_a = self.linear(F.relu(m_a))  # 做relu是为了减少过拟合的风险, linear为了转成lstm输入格式[batch, seq_len, input_size]
        temp_a = self.dropout(temp_a)
        self.bilstm.flatten_parameters()
        v_a, _ = self.bilstm(temp_a)  # [batch, seq_len, num_directions * hidden_size]
        #         print('v_a.shape', v_a.shape)

        temp_b = self.linear(F.relu(m_b))
        temp_b = self.dropout(temp_b)
        self.bilstm.flatten_parameters()
        v_b, _ = self.bilstm(temp_b)

        # 池化, 因为论文中提到了 算总的和对序列很敏感，池化后的效果最好
        # 论文没有具体讲怎么池化，但是从分析中知道，第1维，也就是seq_len是带softmax信息的(决定着当前单词与另一个句子之间所有关系)
        # 而且这样池化后不影响输出的 hidden_size, [b, s, n*h] max(1) ==> [b, n*h],中间的seq_len全用最大
        v_a_avg = v_a.sum(1) / v_a.shape[1]
        v_a_max = v_a.max(1)[0]
        v_b_avg = v_b.sum(1) / v_b.shape[1]
        v_b_max = v_b.max(1)[0]

        # [batch, seq_len, 4 * num_directions * hidden_size]
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=-1)
        #         print('v.shape', v.shape)
        return v


# 最后将上面得到的 v 用多层感知机进行分类
class Prediction(nn.Module):
    def __init__(self, v_size, temp_size, output_size=4):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(v_size, temp_size), nn.Tanh(),
                                 nn.Linear(temp_size, output_size)).to(device)

    def forward(self, v):
        return self.mlp(v)


# 组装上面的四个步骤，就是ESIM模型了
class ESIM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=1, output_size=4, weight=None):
        super(ESIM, self).__init__()
        self.input_encoding = Input_Encoding(input_size=input_size, hidden_size=hidden_size, vocab_size=vocab_size,
                                             weight=weight, num_layers=num_layers)
        self.local_inference_modeling = Local_Inference_Modeling()
        self.inference_composition = Inference_Composition(input_size=input_size, m_hidden_size=8 * hidden_size,
                                                           hidden_size=hidden_size, num_layers=num_layers)
        self.predicted = Prediction(v_size=8 * hidden_size, temp_size=hidden_size, output_size=output_size)

    # a 是premise， b是 hypothesis
    def forward(self, a, b):
        a_bar = self.input_encoding(a)
        b_bar = self.input_encoding(b)

        m_a, m_b = self.local_inference_modeling(a_bar, b_bar)

        v = self.inference_composition(m_a, m_b)

        output = self.predicted(v)

        return output


input_size = 50
hidden_size = 50


random_model = ESIM(input_size, hidden_size, random_vocab_size)
glove_model = ESIM(input_size, hidden_size, glove_vocab_size, weight=torch.tensor(weight, dtype=torch.float))

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(random_model.parameters(), lr=0.001)
optimizer2 = optim.Adam(glove_model.parameters(), lr=0.001)


def time_since(start):
    second = time.time() - start
    minute = math.floor(second / 60)
    second -= minute * 60
    return '%dm %ds' % (minute, second)


# 训练模型并 计算和统计损失
def train(epoch, start, model, optimizer, train_dataloader, dataset, data_rate):
    total_loss = 0
    count = 0
    model.train()
    for i, (sentence1, sentence2, labels) in enumerate(train_dataloader, 1):
        optimizer.zero_grad()
        outputs = model(sentence1, sentence2)
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if i % 100 == 0:
            print(f'[{time_since(start)}]  Epoch: {epoch}', end='')
            print(f'[{i * len(labels)} / {int(len(dataset) * data_rate)}]', end='')  # 用来观察 当前已训练与总数的比较
            print(f'[loss: {total_loss / i * len(labels)}]')

    return total_loss / count  # 平均损失


# 测试模型精度
def test(model, val_dataloader, optimizer, dataset, data_rate):
    correct = 0
    totals = int(len(dataset) * data_rate)
    with torch.no_grad():
        model.eval()
        for i, (sentence1, sentence2, labels) in enumerate(val_dataloader, 1):
            optimizer.zero_grad()
            outputs = model(sentence1, sentence2)
            _, predicted = torch.max(outputs, dim=1)  # outputs.shape ==> [?, 4]
            correct += (predicted == labels.to(device)).sum().item()

        percent = '%2f %%' % (100 * correct / totals)
#         print(f'Test set: Accuracy {correct} / {totals} = {percent}')

    return correct / totals


def compare_draw():
    start_time = time.time()
    print('Training for %d epochs.. \n' % EPOCH)

    # 比较训练集的 损失和精度
    random_train_loss_list = list()
    glove_train_loss_list = list()
    random_train_accuracy_list = list()
    glove_train_accuracy_list = list()

    # 比较测试集的 损失和精度
    random_val_loss_list = list()
    glove_val_loss_list = list()
    random_val_accuracy_list = list()
    glove_val_accuracy_list = list()

    for epoch in range(1, EPOCH + 1):
        ran_tra_loss = train(epoch, start_time, random_model, optimizer1, random_train_dataloader, random_dataset, 0.8)
        ran_tra_acc = test(random_model, random_train_dataloader, optimizer1, random_dataset, 0.8)
        random_train_loss_list.append(ran_tra_loss)
        random_train_accuracy_list.append(ran_tra_acc)

        ran_val_loss = train(epoch, start_time, random_model, optimizer1, random_val_dataloader, random_dataset, 0.2)
        ran_val_acc = test(random_model, random_val_dataloader, optimizer1, random_dataset, 0.2)
        random_val_loss_list.append(ran_val_loss)
        random_val_accuracy_list.append(ran_val_acc)

        glo_tra_loss = train(epoch, start_time, glove_model, optimizer2, glove_train_dataloader, glove_dataset, 0.8)
        glo_tra_acc = test(glove_model, glove_train_dataloader, optimizer2, glove_dataset, 0.8)
        glove_train_loss_list.append(glo_tra_loss)
        glove_train_accuracy_list.append(glo_tra_acc)

        glo_val_loss = train(epoch, start_time, glove_model, optimizer2, glove_val_dataloader, glove_dataset, 0.2)
        glo_val_acc = test(glove_model, glove_val_dataloader, optimizer2, glove_dataset, 0.2)
        glove_val_loss_list.append(glo_val_loss)
        glove_val_accuracy_list.append(glo_val_acc)
#         print('\n')  # 分开好观察

    x = list(range(1, EPOCH + 1))
    plt.subplot(2, 2, 1)
    plt.plot(x, random_train_loss_list, 'r--', label='random')
    plt.plot(x, glove_train_loss_list, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title('Train Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(2, 2, 2)
    plt.plot(x, random_val_loss_list, 'r--', label='random')
    plt.plot(x, glove_val_loss_list, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title('Test Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(2, 2, 3)
    plt.plot(x, random_train_accuracy_list, 'r--', label='random')
    plt.plot(x, glove_train_accuracy_list, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title('Train Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)

    plt.subplot(2, 2, 4)
    plt.plot(x, random_val_accuracy_list, 'r--', label='random')
    plt.plot(x, glove_val_accuracy_list, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title('Test Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)

    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)  # 自动更新画布大小
    plt.savefig('plot.jpg')
    plt.show()


compare_draw()
