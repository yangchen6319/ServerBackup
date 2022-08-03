import pandas as pd
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
        data = pd.read_csv(input_file)
        return data


class MyPro(DataProcessor):
    '''自定义数据读取方法，针对json文件

    Returns:
        examples: 数据集，包含index、中文文本、类别三个部分
    '''

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'train_data.csv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'dev_data.csv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'test_data.csv')), 'test')

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, data, set_type):
        examples = []
        for index, row in data.iterrows():
            # guid = "%s-%s" % (set_type, i)
            text = row['review']
            label = row['label']
            examples.append(
                InputExample(text=text, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, show_exp=True):
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
        # 分词
        tokens = tokenizer.tokenize(example.text)
        # tokens进行编码
        encode_dict = tokenizer.encode_plus(text=tokens,
                                            max_length=max_seq_length,
                                            pad_to_max_length=True,
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        input_ids = encode_dict['input_ids']
        input_mask = encode_dict['attention_mask']
        segment_ids = encode_dict['token_type_ids']

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and show_exp:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features
