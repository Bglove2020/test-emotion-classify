import re
import jieba
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pywt
from numpy import concatenate
import math
from tqdm import tqdm
from pylab import mpl
import matplotlib.gridspec as gridspec
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
#读入文本文件

data_txt= []
data_label= []
with open('OCEMOTION.txt', 'r',encoding='utf-8') as f:
    content = f.read()
lines = content.rsplit('\n')
pattern = re.compile(r'[^\u4e00-\u9fa5\d\w\s\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u3010\u2014\u3011\u2022\u300c\u300d!#$%&\'\"()*+,-.·/:;<=>?@\[\\\]\^_`{|}~]+')
jl ={}
for line in lines:
    list_tmp = line.strip().split("\t")

    # if(len(list_tmp)==3 and len(list_tmp[1])<=100 and(list_tmp[2]=="sadness" or list_tmp[2]=="happiness")):
    if (len(list_tmp) == 3 and len(list_tmp[1])<=100):
        if pattern.search(list_tmp[1]) != None:
            continue
        data_txt.append(list_tmp[1])
        data_label.append(list_tmp[2])

print("剔除数据前数据量：",len(lines),"剔除颜文字后数据量：",len(data_label),"删除数据量：",len(lines)-len(data_label))

label_dict={"sadness":0,"happiness":1,"like":2,"anger":3,"fear":4,"surprise":5,"disgust":6}
for index,i in enumerate(data_label):
    data_label[index]=label_dict[i]

word_freq = Counter()
# 分词并更新词频计数器
for text in data_txt:
    seg_list = jieba.cut(text)
    word_freq.update(seg_list)
# 将计数器对象转为词典数组
dictionary = dict(word_freq)
print(len(dictionary))
word_to_index = {word: index + 1 for index, word in enumerate(dictionary)}

#将文本转化为向量：
data_num =[]
len_max =0
for text in data_txt:
    tmp =[]
    for i,ci in enumerate(jieba.cut(text)):
        tmp.append(word_to_index[ci])
    if len(tmp)>len_max:
        len_max=len(tmp)
    data_num.append(tmp)
#可以一行代码代替：
#data_indexed = [[word_to_index[word] for word in jieba.cut(text)] for text in data_txt]

#0初始化向量，再进行尾部填充序列
data_num_init = [[0 for j in range(len_max)] for i in range(len(data_num))]
for index,seq in enumerate(data_num):
    data_num_init[index][-len(seq):]=seq

#构建dataset
train_size = int(len(data_num_init) * 0.8)
test_size = len(data_num_init) - train_size
# 划分数据
# 按比例划分数据集
test_size = 0.1  # 测试集占总数据集的比例
random_state = 45  # 设置随机种子，保证可重复性

# 将数据集划分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_num_init,data_label, test_size=test_size, random_state=random_state)

# X_train, y_train = data_num_init[:train_size], data_label[:train_size]
# X_test, y_test = data_num_init[train_size:],data_label[train_size:]

X_train, y_train = torch.Tensor(X_train),torch.Tensor(y_train)
X_test, y_test = torch.Tensor(X_test),torch.Tensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

#mylstm(128,64,1,output_size,0.0)
class mylstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(mylstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x的形状为(batch_size, seq_len, num_features:256)
        # h_n==c_n且形状都为(layer_num, batch_size, num_features)
        out,(h_n,c_n) = self.lstm(x)
        # out的形状为(batch_size, seq_len, hidden_size)
        # 取最后一个时间步的输出，并送入线性层
        # print("---------------------------------------------")
        # print(out[:, -1, :])
        h_n =h_n.reshape(-1,64)
        out = self.fc(h_n)
        #这里测试两个线性层的效果并不是很好，降低为一个线性层
        # out = nn.ReLU()(out)
        # out = self.fc2(out)
        # 输出的形状为(batch_size, output_size)
        return out

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        #添加padding很关键，不然会导致序列变短，并且每层添加的padding应该是不一样的，因为空洞卷积的存在，会使得每此卷积跨过很远的距离，所以不填充的话，序列长度会依次递减很多。
        padding = dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = F.relu(self.conv1d(x))
        y = self.dropout(y)
        return y


class TCN(nn.Module):
    def __init__(self, max_dict_index,input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn_blocks = nn.ModuleList()

        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            #tcn中的第一层卷积的输入通道数应该等于输入数据的特征数
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            tcn_block = TCNBlock(in_channels, out_channels, kernel_size, dilation_size, dropout)
            self.tcn_blocks.append(tcn_block)

    def forward(self, x):
        # x的形状为(batch_size, seq_len)

        x = x.permute(0, 2, 1)  # 将输入的最后两个维度交换，变成(batch_size, num_features, seq_len)

        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        return x

class TCN_LSTM(nn.Module):
    def __init__(self, max_dict_index,input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN_LSTM, self).__init__()
        self.tcn = TCN(max_dict_index,input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.lstm =mylstm(228,64,1,output_size,0.2)
        self.emb = nn.Embedding(max_dict_index, input_size)
    def forward(self, x):
        # x的形状为(batch_size, seq_len, num_features)
        x =x.long()
        x =self.emb(x)
        x_1= self.tcn(x)
        x_1 = x_1.permute(0, 2, 1)  # 将输入的最后两个维度交换，变成(batch_size, seq_len, num_features)

        # print(x.size(),x_1.size())
        x=torch.cat([x,x_1],2)
        x=self.lstm(x)
        # 输出的形状为(batch_size, output_size)
        return x
# 构建 TCN 模型
batch_size = 128
max_dict_index =44474

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
model = TCN_LSTM(max_dict_index, input_size=128, output_size=7, num_channels=[128, 128, 128, 128, 100, 100,100], kernel_size=3, dropout=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
#交叉熵损失直接包括了softmax层和对标签的
criterion=nn.CrossEntropyLoss()

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    acc_jl = 0
    #tqdm调用在可迭代对象上，可以显示for循环运行了多久，还剩多久
    loader_xs = tqdm(loader)

    for i, (inputs, labels) in enumerate(loader_xs):
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.size(),labels.size())
        # print(outputs.dtype,labels.dtype)
        labels =labels.long()
        yuce_labels = torch.argmax(outputs, dim=1)

        for index,yuce_label in enumerate(yuce_labels):
            if yuce_label==labels[index]:
                acc_jl+=1

        loss = criterion(outputs, labels)
        # print(type(loss),loss.dtype,loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset),acc_jl/len(loader.dataset)

def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    loader_xs=tqdm(loader)
    acc_jl=0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader_xs):
            outputs = model(inputs)
            labels = labels.long()
            yuce_labels = torch.argmax(outputs, dim=1)

            for index, yuce_label in enumerate(yuce_labels):
                if yuce_label == labels[index]:
                    acc_jl += 1

            labels = labels.long()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset),acc_jl/len(loader.dataset)


num_epochs = 30
best_loss=100
jl_loss_train =[]
jl_loss_test =[]
for epoch in range(num_epochs):
    train_loss,train_acc_rate = train(model, train_dataloader, criterion, optimizer)
    jl_loss_train.append(train_loss)

    test_loss,test_acc_rate = test(model, test_dataloader, criterion)
    jl_loss_test.append(test_loss)
    print("-------------{}--------------".format(epoch))
    print("train_loss: {:.6f} train_acc_rate:{:.6f}".format(train_loss, train_acc_rate))
    print("test_loss: {:.6f} test_acc_rate:{:.6f}".format(test_loss,test_acc_rate))
    if(train_loss<best_loss):
        best_loss=train_loss
        torch.save(model.state_dict(), 'tcn_lstm_01_128_6ceng.pth')


# 创建批次号码
batch_numbers = range(0,num_epochs)

# 绘制损失-批次图
plt.plot(batch_numbers, jl_loss_train)
plt.xlabel('batch_numbers')
plt.ylabel('train_loss')
plt.title('train_loss - batch_numbers')
plt.show()
# #保存模型参数
# model = TCN_LSTM(input_size=6, output_size=1, num_channels=[256, 256, 256, 128, 128, 64], kernel_size=5, dropout=0.0)
# # 加载模型的状态字典
# # model.load_state_dict(torch.load('tcn_lstm_01_128_6ceng.pth'))