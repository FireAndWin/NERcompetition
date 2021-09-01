
train_data_path=r"D:\workspace\competition\01\data\char_ner_train.csv"
saved_ids_data_path=r"D:\workspace\competition\static\saved_ids.pkl"

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
num_labels=17
hidden_dropout_prob= 0.1
hidden_size=768
epoch_num=10
batch_size=25
learning_rate=3e-5
# 训练集、验证集划分比例
train_test_split_size = 0.005