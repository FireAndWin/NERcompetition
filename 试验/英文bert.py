import torch
from transformers import BertModel, BertTokenizer


'''
1.tokenizer,已有内部的id
2.做词嵌入 id vector

[1,2,3]
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%%

input_ids = tokenizer.encode('hello world bert!')
print(tokenizer.convert_ids_to_tokens(input_ids))
ids = torch.LongTensor(input_ids)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
granola_ids = ids.unsqueeze(0)
out = model(input_ids=granola_ids)
b=out[0]
a=1
