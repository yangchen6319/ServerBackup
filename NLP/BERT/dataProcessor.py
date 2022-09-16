from cgitb import text
from lib2to3.pgen2 import token
from this import d
from tkinter.tix import Tree
import pandas as pd
import os


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text, label=None):
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # dicts = []
        data = pd.read_csv(input_file, sep='\t', names=['label', 'review'])
        return data


class MyPro(DataProcessor):
    '''自定义数据读取方法，针对json文件

    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'train.txt')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'dev.txt')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'test.txt')), 'test')

    def get_labels(self):
        return [0, 1, 2, 3, 4]

    def _create_examples(self, data, set_type):
        examples = []
        for index, row in data.iterrows():
            # guid = "%s-%s" % (set_type, i)
            text = row['review']
            label = row['label']
            examples.append(
                InputExample(text=text, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    '''Loads a data file into a list of `InputBatch`s.

    Args:
        examples      : [List] 输入样本，句子和label
        label_list    : [List] 所有可能的类别，0和1
        max_seq_length: [int] 文本最大长度
        tokenizer     : [Method] 分词方法

    Returns:
        features:
            input_ids  : [ListOf] token的id，在chinese模式中就是每个分词的id，对应一个word vector
            input_mask : [ListOfInt] 真实字符对应1，补全字符对应0
            segment_ids: [ListOfInt] 句子标识符，第一句全为0，第二句全为1
            label_id   : [ListOfInt] 将Label_list转化为相应的id表示
    '''
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        # 使用tokenizer类中的encode_plus方法进行分词&编码
        encode_dict = tokenizer(example.text, 
                                padding = "max_length",
                                truncation = True,
                                max_length = max_seq_length,
                                add_special_tokens=True)

        input_ids = encode_dict['input_ids']
        input_mask = encode_dict['attention_mask']
        segment_ids = encode_dict['token_type_ids']

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features



