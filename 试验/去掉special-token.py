import torch
from transformers import BertModel, BertTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# encoded_inputs = tokenizer(["一","一二"],padding=True)
#
# # for ids in encoded_inputs["input_ids"]:
# #     print(tokenizer.decode(ids))
#
# granola_ids = encoded_inputs['input_ids']
#
# print(granola_ids)
'''
PAD:0
CLS:101
SEP:102
'''

# a=torch.tensor([[101, 1,2,3,4,5,6,7,8, 102, 0], [101, 2,3,1,2,3,4,5,6,7, 102]])
# for b in a:
#     b=b[b!=101]
#     b=b[b!=102]
#     b=b[b!=0]
#     print(b)
# print(a==1)


# a=torch.tensor([[101, 1,2,3,4,5,6,7,8, 102, 0], [101, 2,3,1,2,3,4,5,6,7, 102]])
# for b in a:
#     b=b[b!=101][b!=102][b!=0]
#     print(b)
# print(a==1)

a=torch.tensor([[
                    [1,2],
                    [3,4],
                    [5,6]
                ],

                [
                    [7,8],
                    [9,10],
                    [11,12]
                ]

               ])

print(a.shape)
print(torch.index_select(a,0,torch.tensor([1])))
print(torch.index_select(a[0],0,torch.arange(1,2+1)))