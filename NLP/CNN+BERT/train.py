import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_process
import torch.utils.data as Data 
import model
import time

# 指定训练设备
device = torch.device('cuda')

# 读取数据
df_train = data_process.load_data('train')
print("训练集数据量:{}".format(len(df_train)))

# 参数设定
PAD = ['<PAD>']
pad_size = 64
batch_size = 64
epoch = 60

# 数据处理
df_train['rev_text'] = df_train['text'].apply(
    lambda x:data_process.revise_sentence(x, pad_size, PAD)
)
print("去除停用词完毕")
sentences = list(df_train['rev_text'])
labels = list(df_train['stars'])

num_classes = len(set(labels))
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Embedding
input_batch, target_batch = data_process.make_data(sentences, labels, word2idx)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)
print("EMbedding完毕")

# 划分数据集
train_dataset = Data.TensorDataset(input_batch, target_batch)


train_loader = Data.DataLoader(
    dataset=train_dataset,  # 数据，封装进Data.TensorDataset()类的数据
    batch_size=batch_size,  # 每块的大小
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
)

# 创建模型
model = model.TextCNN(num_classes, vocab_size, word2idx).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 记录训练参数
train_loss_his = []
train_totalaccuracy_his = []
test_totalaccuracy_his = []
# Training
print("----------训练开始-----------")
start_time = time.time()
for epoch in range(epoch):
    loss_epoch = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        # 记录训练损失值
        loss_epoch = loss.cpu().detach().numpy()
        train_loss_his.append(loss_epoch)
        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 打印训练损失
    if (epoch + 1) % 10 == 0:
        print('Epoch: {:4},  loss: {:.6f}'.format(epoch+1, loss_epoch))
# 训练完毕，保存训练得到的模型权重
torch.save(model.state_dict(), './data/model_weight.pth')


plt.plot(train_loss_his, label='Train Loss')
plt.legend(loc='best')
plt.xlabel('Steps') 
plt.savefig("./data/fig.png")
