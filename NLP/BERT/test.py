from torch.utils.data import DataLoader
from model import *
from data_set import *
from dataProcessor import *
import matplotlib.pyplot as plt
import time
from transformers import BertTokenizer
from transformers import logging


# 加载训练数据
datadir = "./data"
bert_dir = "./pretrain_model"

my_processor = MyPro()
label_list = my_processor.get_labels()
train_data = my_processor.get_train_examples(datadir)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

train_data = train_data[:1]
text= "The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
train_features = convert_examples_to_features(train_data, label_list, 80, tokenizer)
print(train_features[0].input_ids)
print(train_features[0].input_mask)
print(train_features[0].segment_ids)
print(train_features[0].label_id)