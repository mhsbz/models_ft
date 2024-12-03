from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载本地模型和分词器
model_path = './results'  # 本地模型路径
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-chinese")
model = BertForSequenceClassification.from_pretrained(model_path).to(device)

# 准备输入数据
text = "三只羊"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)

# 进行推理
with torch.no_grad():
    outputs = model(**inputs.to(device))
    print(outputs.logits)
    predictions = torch.argmax(outputs.logits, dim=-1)

# 输出预测结果
print(f"Predicted class: {predictions.item()}")