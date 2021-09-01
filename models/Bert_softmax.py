from transformers import BertModel, BertTokenizer
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import data_prepare.config as config

class BertNER(Module):
    def __init__(self):
        super(BertNER, self).__init__()
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        #self.init_weights()

    def forward(self, input_data, lengths ):

        # 求出attention_mask
        max_len = int(max(lengths))
        attention_mask = torch.Tensor()
        # length = length.numpy()
        # 与每个序列等长度的全1向量连接长度为最大序列长度-当前序列长度的全0向量。
        for len_ in lengths:
            attention_mask = torch.cat((attention_mask, torch.Tensor([[1] * len_ + [0] * (max_len - len_)])), dim=0)


        input_data=input_data.long()
        attention_mask=attention_mask.long()

        outputs = self.bert(input_data,
                            attention_mask=attention_mask,)
        sequence_output = outputs[0]
        #print("sequence_output,shape:",sequence_output.shape)

        origin_sequence_output=[]
        # 去掉special token, 待改进
        for idx, sentence in enumerate(sequence_output):
            #print(len(sentence),lengths[idx])
            sentence=torch.index_select(sentence,0,torch.arange(1,lengths[idx]-1))
            origin_sequence_output.append(sentence)

        #print('origin_sequence_output:',len(origin_sequence_output),len(origin_sequence_output[0]),len(origin_sequence_output[1]))
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # 得到判别值
        # (batch_size,seq,label_num)
        logits = self.classifier(padded_sequence_output)

        return logits

    def get_loss(self,output_logits,labels):

        labels=labels.long()
        #output_logits=output_logits.long()
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                # 只留下label存在的位置计算loss
                # (batch_size*seq,)
                active_loss = loss_mask.reshape(-1) == 1
                #print('active_loss',active_loss.shape)
                # (batch_size*seq, label_nums)
                active_logits = output_logits.reshape(-1, self.num_labels)#
                #print('active_logits',active_logits.shape)
                active_logits = active_logits[active_loss]
                # (batch_size*seq,)
                active_labels = labels.reshape(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(output_logits.reshape(-1, self.num_labels), labels.reshape(-1))
        #print('loss',loss.shape)
        return loss




