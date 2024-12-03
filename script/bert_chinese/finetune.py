"""
训练bert做情感分类
"""
import json
from datetime import datetime

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

with open("/home/dxj/projects/models_ft/data/cmm/data_eval_fixed_shuffled.json", "r",
          encoding='utf-8') as f:
    data = json.load(f)

print("len data:", len(data))

train_data = {"text": [], "label": []}
test_data = {"text": [], "label": []}

for idx, item in enumerate(tqdm(data)):
    label = 0
    if item["messages"][2]["content"] == "达人":
        label = 1
    elif item["messages"][2]["content"] == "店铺":
        label = 2
    elif item["messages"][2]["content"] == "品牌":
        label = 3

    row = {
        "text": item["messages"][1]["content"],
        "label": label,
    }
    if idx < 30000:
        train_data["text"].append(row["text"])
        train_data["label"].append(row["label"])
    else:
        test_data["text"].append(row["text"])
        test_data["label"].append(row["label"])

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-chinese')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def encode_examples(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (preds == labels).mean()}


train_encoded_dataset = train_dataset.map(encode_examples, batched=True)
test_encoded_dataset = test_dataset.map(encode_examples, batched=True)

print(train_encoded_dataset)
print(test_encoded_dataset)

data_collator = DataCollatorWithPadding(tokenizer)

datefmt = datetime.now().strftime("%Y%m%d%H%M")

num_epochs = 1
learn_rate = 2e-5
batch_size = 16
lr_scheduler_type = "cosine"

output_dir = f"./{datefmt}_e{num_epochs}_lrs{lr_scheduler_type}_lr{learn_rate}_bs{batch_size}"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=8,
    num_train_epochs=num_epochs,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=10,
    metric_for_best_model="accuracy",
    learning_rate=learn_rate,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb",
    optim="adamw_hf",
    warmup_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded_dataset,
    eval_dataset=test_encoded_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
