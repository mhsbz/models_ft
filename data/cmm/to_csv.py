import json

import pandas as pd

with open('data_eval_shuffled.json', 'r', encoding='utf-8') as file:
    eval = json.load(file)

with open('data_train_shuffled.json', 'r', encoding='utf-8') as file:
    train = json.load(file)

data_list = []
for item in eval+train:

    text = item["messages"][1]["content"]
    label = item["messages"][2]["content"]

    tag = 0
    if label == "达人":
        tag = 1
    elif label == "店铺":
        tag = 2
    else:
        tag = 0

    data_list.append({
        "text": text,
        "label": tag
    })
    # break

data_map = {}
for item in data_list:
    data_map[item["text"]] = item["label"]

print(len(data_map))

all_list = []
for k,v in data_map.items():
    all_list.append({
        "text": k,
        "label": v
    })

print(len(all_list))

print(all_list[0])

df1 = pd.DataFrame(all_list[:int(len(all_list)*0.8)], columns=["text", "label"])
df2 = pd.DataFrame(all_list[int(len(all_list)*0.8):], columns=["text", "label"])


df1.to_csv("./data_train.csv", index=False)
df2.to_csv("./data_eval.csv", index=False)
