import torch
from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

outputs=tokenizer(['我','去','吃','放'],return_tensors='pt')
print(outputs.size)