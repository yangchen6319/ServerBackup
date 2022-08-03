# 数据处理模块
from nltk.corpus import stopwords
import nltk


def separate_sentence(text):
    disease_List = nltk.word_tokenize(text)
    #去除停用词
    filtered = [w for w in disease_List if(w not in stopwords.words('english'))]
    #进行词性分析，去掉动词、助词等
    Rfiltered =nltk.pos_tag(filtered)
    #以列表的形式进行返回，列表元素以（词，词性）元组的形式存在
    filter_word = [i[0] for i in Rfiltered]
    return " ".join(filter_word)


PAD = '<PAD>'
pad_size = 64

def revise_size(text):
    sen2list = text.split()
    sentence_len = len(sen2list)
    if sentence_len<pad_size:
        text += PAD*(pad_size-sentence_len)
    else:
        text = " ".join(sen2list[:pad_size])
    return text

def make_data(sentences, labels, word2idx):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    targets = []
    for out in labels:
        targets.append(out)
    return inputs, targets

