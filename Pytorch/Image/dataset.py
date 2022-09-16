from cgi import test
from tkinter.tix import Tree
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image

train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())

# 手写一个dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        # dataset的关键在于这里：要返回一个什么样的数据给dataloader！！！！！！！！！！！
        # 通常有如下两种形式：
        # 1、 如上形式：返回原数据和对应的标签，这里原数据的格式为tensor[1,28,28],对应标签为tensor[1]!
        # 那么dataloader会返回tensor[64,1,28,28]和tensor[64]
        # 2、第二种情况，需要返回的数据比较多，为了不搞乱，我们往往采用一个dict把所有数据装进去然后返回给dataloader
        # 那么dataloader也会返回一个dict，dict中每一项都是一个batch的数据叠在一起

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(type(train_features))
print(type(train_labels))
print(train_features.size())
print(train_labels.size())

for step, batch_data in enumerate(train_dataloader):
    print(len(batch_data))
    print(batch_data[0].size())
    break
'''
总结一下dataset和dataloader:
dataset:
作用：制定从序列到数据的映射规则，简单来说：就是给个idx就给你返回对应的数据，同时也规定了数据返回的格式
要求：实现__len__和__getitem__方法，这两个东西，一个规定了序列长度，另一个规定了映射规则
dataloader:
作用：制定整个数据训练的顺序，就是数据以什么样的顺序送入模型

这里千万要注意！！！！！！！！！
数据的格式是由dataset规定的，dataloader只是把dataset返回的数据放在了一个tensor中


'''
for train1,train2 in train_dataloader:
    print(train1.size())
    print(train2.size())
    break