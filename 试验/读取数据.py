import pandas as pd
import torch
from transformers import BertModel, BertTokenizer

label2id={
    'M-GPE':0,
    'M-ORG':1,
    'M-PER':2,
    'M-LOC':3,
    'S-ORG':4,
    'E-ORG':5,
    'S-PER':6,
    'E-LOC':7,
    'B-LOC':8,
    'B-ORG':9,
    'S-GPE':10,
    'S-LOC':11,
    'B-PER':12,
    'E-GPE':13,
    'O':14,
    'B-GPE':15,
    'E-PER':16,
}

word_lists = []
tag_lists = []
with open(r"D:\workspace\competition\01\data\char_ner_train.csv", 'r', encoding='utf-8') as f:
    word_list = []
    tag_list = []

    for i,line in enumerate(f.readlines()):
        if i==0:
            continue
        if len(line)<=2 or line==',\n':

            word_lists.append(word_list[:])
            tag_lists.append(tag_list[:])
            word_list.clear()
            tag_list.clear()

        else:
            word, tag = line.strip('\n').split(',')
            word_list.append(word)

            tag_id=label2id[tag]
            tag_list.append(tag_id)


''''''

sentences=[]#存储所有的句子
for s in word_lists:
    sentence="".join(s)
    sentences.append(sentence)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
outputs=tokenizer(sentences[:3],padding=True, return_tensors='pt')
ids=outputs['input_ids']
print(ids.size())
