import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv('labeledTrainData.tsv',sep='\t',header=0)
print('一共有{}条数据'.format(len(df)))

from nltk.corpus import stopwords
import nltk
def separate_sentence(text):
    disease_List = nltk.word_tokenize(text)
    #去除停用词
    filtered = [w for w in disease_List if(w not in stopwords.words('my_english'))]
    #进行词性分析，去掉动词、助词等
    Rfiltered =nltk.pos_tag(filtered)
    #以列表的形式进行返回，列表元素以（词，词性）元组的形式存在
    filter_word = [i[0] for i in Rfiltered]
    return " ".join(filter_word)


df['sep_review'] = df['review'].apply(lambda x:separate_sentence(x))

sentences = list(use_df['sep_review'])
labels = list(use_df['sentiment'])

PAD = ' <PAD>'  # 未知字，padding符号用来填充长短不一的句子
pad_size =  64     # 每句话处理成的长度(短填长切)

for i in range(len(sentences)):
    sen2list = sentences[i].split()
    sentence_len = len(sen2list)
    if sentence_len<pad_size:
        sentences[i] += PAD*(pad_size-sentence_len)
    else:
        sentences[i] = " ".join(sen2list[:pad_size])

# TextCNN Parameter
num_classes = len(set(labels))  # num_classes=2
batch_size = 64
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

def make_data(sentences, labels):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    targets = []
    for out in labels:
        targets.append(out) # To using Torch Softmax Loss function
    return inputs, targets
input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

from sklearn.model_selection import train_test_split
# 划分训练集，测试集
x_train,x_test,y_train,y_test = train_test_split(input_batch,target_batch,test_size=0.2,random_state = 0)

train_dataset = Data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
test_dataset = Data.TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
dataset = Data.TensorDataset(input_batch, target_batch)

train_loader = Data.DataLoader(
    dataset=train_dataset,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=batch_size,      # 每块的大小
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多进程（multiprocess）来读数据
)
test_loader = Data.DataLoader(
    dataset=test_dataset,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=batch_size,      # 每块的大小
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多进程（multiprocess）来读数据
)

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.filter_sizes = (2, 3, 4)
        self.embed = 300
        self.num_filters = 256
        self.dropout = 0.5
        self.num_classes = num_classes
        self.n_vocab = vocab_size
        #通过padding_idx将<PAD>字符填充为0，因为他没意义哦，后面max-pooling自然而然会把他过滤掉哦
        self.embedding = nn.Embedding(self.n_vocab, self.embed, padding_idx=word2idx['<PAD>'])
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
        
    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = TextCNN().to(device)
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

        pred = output.max(1, keepdim=True)[1]                           # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()

# test_loss /= len(test_loader.dataset)
# test_loss_list.append(test_loss)
test_acc_list.append(100. * correct / len(test_loader.dataset))
print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
