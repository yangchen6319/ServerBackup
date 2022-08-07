from torch import nn
import os
from transformers import BertModel


class ClassifierModel(nn.Module):
    def __init__(self, bert_dir, dropout_prob=0.1):
        super(ClassifierModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(
            config_path
        ), 'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir)

        self.bert_config = self.bert_module.config

        self.dropout_layer = nn.Dropout(dropout_prob)
        out_dims = self.bert_config.hidden_size
        self.obj_classifier = nn.Linear(out_dims, 2)

    def forward(self, input_ids, input_mask, segment_ids, label_id=None):

        bert_outputs = self.bert_module(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids
        )

        seq_out, pooled_out = bert_outputs[0], bert_outputs[1]
        # 对反向传播及逆行截断
        x = pooled_out.detach()
        out = self.obj_classifier(x)
        return out
