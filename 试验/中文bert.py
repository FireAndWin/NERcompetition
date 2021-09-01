import torch
from transformers import BertModel, BertTokenizer


'''
1.tokenizer,已有内部的id
2.做词嵌入 id vector

[1,2,3]
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

#%%

#encoded_inputs = tokenizer(['一二三四五','一二三四五六','一二三四五六七八'])
encoded_inputs = tokenizer.convert_tokens_to_ids(['1','2','3'])
encoded_inputs.insert(0,101)
encoded_inputs.append(102)
print(encoded_inputs)
print(tokenizer.decode(encoded_inputs))

# model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
# granola_ids = encoded_inputs['input_ids']
# out = model(input_ids=granola_ids,
#             attention_mask=encoded_inputs['attention_mask'])
# b=out[0]
# a=1
