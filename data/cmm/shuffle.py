import json
import random

# # 读取JSON文件
# with open('./data_train_shuffled.json', 'r', encoding='utf-8') as file:
#     train = json.load(file)
#
# # 打乱数组
# random.shuffle(train)
#
# print(len(train))
#
# with open('./data_eval_shuffled.json', 'r', encoding='utf-8') as file:
#     eva = json.load(file)
#
# train.append(random.shuffle(eva))
#
# random.shuffle(train)
#
#
# for idx in train:
#     print()
#
# # 将打乱后的数据写回文件
# with open('data_all_shuffled.json', 'w', encoding='utf-8') as file:
#     json.dump(train, file, ensure_ascii=False, indent=4)



with open('data_eval_shuffled.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

data_list = []
for item in data:

    item["messages"][0]["content"] = "你是一个有用的机器人助手，我会给你一个词语你需要帮我判断这个名词是哪个类别的,可选的类别列表有['达人','店铺','品牌']"
    print(item["messages"][0]["content"])

    data_list.append(item)
    # break


with open('data_eval_fixed_shuffled.json', 'w', encoding='utf-8') as file:
    json.dump(data_list, file, ensure_ascii=False, indent=4)