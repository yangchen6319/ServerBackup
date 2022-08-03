from matplotlib.pyplot import text
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords

# 读取数据
def load_data(dir):
    dir = os.path.join('data', dir) + ".txt"
    df = pd.read_csv(dir, sep='\t', header=None, names=['stars', 'text'])
    return df


# 分词
def revise_sentence(text, pad_size, PAD):
    text = str(text)
    disease_list = nltk.word_tokenize(text)
    # 去除停用词和符号
    filtered = [w for w in disease_list if (w not in stopwords.words('english'))]
    # 修改sentence长度
    if len(filtered) < pad_size:
        filtered += PAD * (pad_size - len(filtered))
    else:
        filtered = filtered[:pad_size]
    return " ".join(filtered)


def separate_sentence(text):
    text = str(text)
    disease_List = nltk.word_tokenize(text)
    # 去除停用词
    filtered = [w for w in disease_List if (w not in stopwords.words('english'))]
    print(filtered)
    # 进行词性分析，
    # Rfiltered =nltk.pos_tag(filtered)
    # print(Rfiltered)
    # 以列表的形式进行返回，列表元素以（词，词性）元组的形式存在
    # filter_word = [i[0] for i in Rfiltered]
    return " ".join(filtered)


# 简单的文本embedding
def embedding(sentences, labels, word2idx):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    targets = []
    for out in labels:
        targets.append(out)
    return inputs, targets


df_data = load_data('train')
text1 = str(df_data.iloc[0, 1])
result = revise_sentence(text1, 14, ['<PAD>'])
print(result)
