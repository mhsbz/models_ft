import pandas

eval = pandas.read_csv("./data_eval.csv").dropna()

eval.loc[eval['label'] == "shop", 'label'] = 0
eval.loc[eval['label'] == "author", 'label'] = 1
eval.loc[eval['label'] == "brand", 'label'] = 2


train = pandas.read_csv("./data_train.csv").dropna()

train.loc[train['label'] == "shop", 'label'] = 0
train.loc[train['label'] == "author", 'label'] = 1
train.loc[train['label'] == "brand", 'label'] = 2

eval.to_csv("./data_eval.csv", index=False)
train.to_csv("./data_train.csv", index=False)