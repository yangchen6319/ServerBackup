import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_process


# 指定训练设备
device = torch.device('cuda')

# 读取数据
df_train = data_process.load_data('train')
print("训练集数据量：{}".format(len(df_train)))

df_head = df_train.head()

# 参数设定
PAD = ['<PAD>']
pad_size = 64


# 数据处理
df_head['rev_text'] = df_head['text'].apply(
    lambda x:data_process.revise_sentence(x, pad_size, PAD)
)


print(df_head.head())