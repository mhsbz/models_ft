import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (preds == labels).mean()}

# 参数设置
train_file = 'data_train.csv'  # 训练数据文件路径
test_file = 'data_eval.csv'    # 测试数据文件路径
model_name = 'google-bert/bert-base-chinese'  # 使用的中BERT模型
max_length = 32  # 输入序列的最大长度
batch_size = 256  # 批处理大小
num_epochs = 1  # 训练轮数
num_labels = 3  # 分类任务的类别数量

# 加载数据集
dataset = load_dataset('csv', data_files={'train': train_file, 'test': test_file})

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained(model_name)
# 数据预处理函数
def preprocess_function(examples):
   ipt = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
   return ipt

# 对数据集进行预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 初始化模型
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("cuda")

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=num_epochs,     # 训练轮数
    per_device_train_batch_size=batch_size,  # 批处理大小
    per_device_eval_batch_size=256,   # 评估时的批处理大小
    warmup_steps=500,                # 热身步骤
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    metric_for_best_model="accuracy",
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=1000,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 评估模型
trainer.evaluate()

# 使用模型进行预测
# predictions = trainer.predict(tokenized_dataset['test'])
# print(predictions.predictions)  # 输出预测结果
