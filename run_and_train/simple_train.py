from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_prepare.data_set import *
import logging
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from torch.nn import init
from models.Bert_softmax import BertNER
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from evaluate_model import get_f1

# 先加载分词好的数据
file = open(config.saved_ids_data_path, "rb")
word_lists,tag_lists=pickle.load(file)
file.close()
logging.info("--------已加载好训练数据!--------")

# 分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(word_lists[:1000], tag_lists[:1000], test_size=config.train_test_split_size, random_state=0)

# 构建dataloader
train_dataset=MyDataSet(x_train,y_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
test_dataset=MyDataSet(x_test,y_test)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
logging.info("--------已构建好DataLoader!--------")

# 这里加载模型
model=BertNER()
logging.info("--------已准备好模型!--------")

param_optimizer = list(model.classifier.named_parameters())
optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)#, correct_bias=False)
logging.info("--------已构建优化器!--------")

for epoch in range(1,config.epoch_num+1):
    for batch_num,batch_data in enumerate(tqdm(train_dataloader)):
        #print(batch_num,'begin')
        x,y,lengths =batch_data
        output_logits=model(x,lengths)
        loss=model.get_loss(output_logits ,y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num%20==0:
            print("loss:", loss)
            print("f1_score:",get_f1(model,test_dataloader))


        #print(batch_num,loss)

