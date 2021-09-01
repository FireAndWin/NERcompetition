import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from config import label2id
import config
import pickle

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
word_lists = []
tag_lists = []
with open(config.train_data_path, 'r', encoding='utf-8') as f:
    word_list = []
    tag_list = []

    for i,line in enumerate(f.readlines()):
        if i==0:
            continue
        if len(line)<=2 or line==',\n':

            word_list=tokenizer.convert_tokens_to_ids(word_list)
            word_list.insert(0,101)
            word_list.append(102)
            word_lists.append(word_list[:])
            tag_lists.append(tag_list[:])
            word_list.clear()
            tag_list.clear()

        else:
            word, tag = line.strip('\n').split(',')
            word_list.append(word)

            tag_id=label2id[tag]
            tag_list.append(tag_id)


with open(config.saved_ids_data_path,'wb') as f:
    pickle.dump((word_lists,tag_lists),f)

print("结束")

