import pickle
import config
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

class MyDataSet(Dataset):

    def __init__(self,wordlists,taglists):
        self.wordlists = wordlists
        self.taglists =taglists

    def __getitem__(self, item):
        return torch.Tensor(self.wordlists[item]), torch.Tensor(self.taglists[item])

    def __len__(self):
        return len(self.wordlists)


def collate_fn(batch):
    """
    :param batch: (batch_num, ([sentence_len, word_embedding], [sentence_len]))
    :return:
    """
    a,b=len(batch),len(batch[0])
    x_list = [x[0] for x in batch]
    y_list = [x[1] for x in batch]
    lengths = [len(item[0]) for item in batch]
    x_list = pad_sequence(x_list, padding_value=0)
    y_list = pad_sequence(y_list, padding_value=-1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    sen_list=[tokenizer.decode(x) for x in x_list  ]





    return x_list.transpose(0, 1), y_list.transpose(0, 1), lengths