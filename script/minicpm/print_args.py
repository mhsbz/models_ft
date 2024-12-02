import torch

training_args = torch.load('/home/dxj/projects/models_ft/models/MiniCPM3-4B-adv_gen20241202_1236/checkpoint-1000/training_args.bin')

# 打印内容
print(training_args)