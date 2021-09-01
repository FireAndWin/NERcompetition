import numpy as np
from sklearn.metrics import f1_score
import torch

def get_f1(model, test_dataLoader):
    with torch.no_grad():

        all_y_pred=[]
        all_y_true=[]
        for idx, batch_data in enumerate(test_dataLoader):
            x, labels, lengths = batch_data
            logits = model.forward(x,lengths)

            # 最大值简单解码
            decoded_outputs = np.argmax(logits, axis=2)

            # 思路是先展平,然后根据loss_mask选出非PADDING值,最后送入f1计算
            # loss_mask,概念和get_loss里的那个变量一样
            loss_mask = labels.gt(-1).reshape(-1)

            # 将输出和标签展开
            y_pred = decoded_outputs.reshape(-1)[loss_mask]
            y_true = labels.reshape(-1)[loss_mask]

            all_y_pred.append(y_pred)
            all_y_true.append(y_true)

        all_y_true=np.concatenate(all_y_true,axis=0)
        all_y_pred = np.concatenate(all_y_pred, axis=0)

        f1_macro = f1_score(all_y_true, all_y_pred, average='macro')
        f1_micro = f1_score(all_y_true, all_y_pred, average='micro')

        return f1_macro, f1_micro