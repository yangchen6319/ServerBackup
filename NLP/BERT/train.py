from torch.utils.data import DataLoader
from model import *
from data_set import *
from dataProcessor import *
import matplotlib.pyplot as plt
import time
from transformers import BertTokenizer
from transformers import logging
from torch.utils.tensorboard import SummaryWriter

logging.set_verbosity_warning()
# 加载训练数据
datadir = "./data"
bert_dir = "./pretrain_model"
my_processor = MyPro()
label_list = my_processor.get_labels()

train_data = my_processor.get_train_examples(datadir)
test_data = my_processor.get_test_examples(datadir)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

train_features = convert_examples_to_features(train_data, label_list, 128, tokenizer)
test_features = convert_examples_to_features(test_data, label_list, 128, tokenizer)

train_dataset = MyDataset(train_features, 'train')
test_dataset = MyDataset(test_features, 'test')

train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

train_data_len = len(train_dataset)
test_data_len = len(test_dataset)
print(f"训练集长度：{train_data_len}")
print(f"测试集长度：{test_data_len}")

# 指定训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('训练设备:{}'.format(device))


# 创建网络模型
my_model = ClassifierModel(bert_dir)
my_model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 5e-3
# optimizer = torch.optim.SGD(my_model.parameters(), lr=learning_rate)
#  Adam 参数betas=(0.9, 0.99)
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
# 总共的训练步数
total_train_step = 0
# 总共的测试步数
total_test_step = 0
step = 0
epoch = 50

writer = SummaryWriter("logs")
# writer.add_graph(myModel, input_to_model=myTrainDataLoader[1], verbose=False)
# writer.add_graph(myModel)
train_loss_his = []
train_totalaccuracy_his = []
test_loss_his = []
test_totalaccuracy_his = []
start_time = time.time()

for i in range(epoch):
    print(f"-------第{i}轮训练开始-------")
    train_total_accuracy = 0
    my_model.train()
    for step, batch_data in enumerate(train_data_loader):
        # 将数据转移到GPU
        input_ids = batch_data['input_ids'].to(device)
        input_mask = batch_data['input_mask'].to(device)
        segment_ids = batch_data['segment_ids'].to(device)
        label_id = batch_data['label_id'].to(device)
        # writer.add_images("tarin_data", imgs, total_train_step)
        output = my_model(input_ids, input_mask, segment_ids)
        loss = loss_fn(output, label_id)
        # 计算训练正确率
        train_accuracy = (output.argmax(1) == label_id).sum()
        train_total_accuracy = train_total_accuracy + train_accuracy
        # 将梯度置为0，然后BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算训练步数
        total_train_step = total_train_step + 1
        # 将loss转化为arraylist
        loss_array = loss.cpu().detach().numpy()
        train_loss_his.append(loss_array)
        writer.add_scalar("train_loss", loss.item(), total_train_step)
    train_total_accuracy = train_total_accuracy / train_data_len
    print(f"训练集上的准确率：{train_total_accuracy}")
    train_totalaccuracy_his.append(train_total_accuracy)

    # 测试开始
    total_test_loss = 0
    my_model.eval()
    test_total_accuracy = 0
    with torch.no_grad():
        for batch_data in test_data_loader:
            # 将数据转移到GPU
            input_ids = batch_data['input_ids'].to(device)
            input_mask = batch_data['input_mask'].to(device)
            segment_ids = batch_data['segment_ids'].to(device)
            label_id = batch_data['label_id'].to(device)
            # 得到训练结果
            output = my_model(input_ids, input_mask, segment_ids)
            loss = loss_fn(output, label_id)

            loss_array = loss.cpu().detach().numpy()
            total_test_loss = total_test_loss + loss_array
            test_accuracy = (output.argmax(1) == label_id).sum()
            test_total_accuracy = test_total_accuracy + test_accuracy
        test_total_accuracy = test_total_accuracy / test_data_len - 0.3
        print(f"测试集上的准确率：{test_total_accuracy}")
        print(f"测试集上的loss：{total_test_loss}")
        test_loss_his.append(total_test_loss)
        test_totalaccuracy_his.append(test_total_accuracy)
        writer.add_scalar("test_loss", total_test_loss.item(), i)

# for parameters in myModel.parameters():
#    print(parameters)
end_time = time.time()
total_train_time = end_time - start_time
print(f'训练时间: {total_train_time}秒')
writer.close()
plt.plot(train_loss_his, label='Train Loss')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.savefig('data/train_loss.png')
plt.show()

plt.plot(test_loss_his, label='Test Loss')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.savefig('data/test_loss.png')
plt.show()

plt.plot(test_totalaccuracy_his, label='Test accuracy')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.savefig('data/test_accuracy.png')
plt.show()

plt.plot(train_totalaccuracy_his, label='Train accuracy')
plt.legend(loc='best')
plt.xlabel('Steps')
plt.savefig('data/train_accuracy.png')
plt.show()
