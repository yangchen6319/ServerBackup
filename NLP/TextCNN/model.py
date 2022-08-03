import torch.nn as nn
import torch
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, word2idx):
        super(TextCNN, self).__init__()
        self.filter_sizes = (2, 3, 4)
        self.embed = 300
        self.num_filters = 256
        self.dropout = 0.5
        self.num_classes = num_classes
        self.n_vocab = vocab_size
        # 通过padding_idx将<PAD>字符填充为0，因为他没意义哦，后面max-pooling自然而然会把他过滤掉哦
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