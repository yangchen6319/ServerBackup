import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import data_process
import model
from sklearn.model_selection import train_test_split

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 读取数据
df = pd.read_csv('./data/labeledTrainData.tsv', sep='\t', header=0)
print('一共有{}条数据'.format(len(df)))
# 数据处理
data_use = df.iloc[0:65]
data_use['sep_review'] = data_use['review'].apply(
    lambda x: data_process.separate_sentence(x)
)
sentences = list(data_use['sep_review'])
labels = list(data_use['sentiment'])

for i in range(len(sentences)):
    sentences[i] = data_process.revise_size(sentences[i])

# 设置一些参数
num_classes = len(set(labels))  # num_classes=2
batch_size = 64
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# embeding
input_batch, target_batch = data_process.make_data(sentences, labels, word2idx)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(
    target_batch
)

# 划分训练集，测试集
x_train, x_test, y_train, y_test = train_test_split(
    input_batch, target_batch, test_size=0.2, random_state=0
)

train_dataset = Data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
test_dataset = Data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
dataset = Data.TensorDataset(input_batch, target_batch)

train_loader = Data.DataLoader(
    dataset=train_dataset,  # 数据，封装进Data.TensorDataset()类的数据
    batch_size=batch_size,  # 每块的大小
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多进程（multiprocess）来读数据
)
test_loader = Data.DataLoader(
    dataset=test_dataset,  # 数据，封装进Data.TensorDataset()类的数据
    batch_size=batch_size,  # 每块的大小
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多进程（multiprocess）来读数据
)

model = model.TextCNN(num_classes, vocab_size, word2idx).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(30):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 训练完毕，保存训练得到的模型权重
torch.save(model.state_dict(), './data/model_weight.pth')

test_acc_list = []
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        #         loss = criterion(output, target)
        #         test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加

        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()

# test_loss /= len(test_loader.dataset)
# test_loss_list.append(test_loss)
test_acc_list.append(100.0 * correct / len(test_loader.dataset))
print(
    'Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
    )
)
