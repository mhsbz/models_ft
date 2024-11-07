"""
训练bert做情感分类
"""
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification,DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

dataset = load_dataset("XiangPan/waimai_10k")

train_test_split = dataset['train'].train_test_split(test_size=0.1)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',local_files_only=True)
model = BertForSequenceClassification.from_pretrained('bert-base-chinese',local_files_only=True)

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased',local_files_only=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def encode_examples(examples):
    resp = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return resp

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (preds == labels).mean()}


train_encoded_dataset = train_dataset.map(encode_examples, batched=False)
test_encoded_dataset = test_dataset.map(encode_examples, batched=False)

print(train_encoded_dataset)
print(test_encoded_dataset)


data_collator = DataCollatorWithPadding(tokenizer)

datefmt = datetime.now().strftime("%Y%m%d%H%M")

num_epochs = 1
learn_rate = 2e-5
batch_size = 16
lr_scheduler_type = "cosine"



output_dir = f"/Users/dxj/Desktop/self-project/model_ft/models/waimai_10k_bert/{datefmt}_e{num_epochs}_lrs{lr_scheduler_type}_lr{learn_rate}_bs{batch_size}"

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

# trainer.evaluate()




