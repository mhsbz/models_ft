import json
import platform
from datetime import datetime
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, model_max_length: int = 1024 * 4):
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignored_index = -100

        item = self._process_data(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for _id in item["label_ids"]:
            if _id == self.ignored_index:
                continue
            labels.append(_id)
        print("label:", self.tokenizer.decode(labels))

    def _process_data(self, example) -> Dict[str, torch.Tensor]:
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = [self.ignored_index]

        for message in example["messages"]:
            role = message["role"]
            content = message["content"]

            ## content_ids是词汇表列表
            content_ids = self.tokenizer.apply_chat_template([message])

            if role == "user":
                if self.tokenizer.eos_token_id == 73440:
                    input_ids += self.tokenizer.apply_chat_template([message], add_generation_prompt=True)
                    label_ids += [self.ignored_index] * len(
                        self.tokenizer.apply_chat_template([message], add_generation_prompt=True))
                else:
                    input_ids += content_ids
                    label_ids += [self.ignored_index] * len(content_ids)

            elif role == "system":
                input_ids += content_ids
                label_ids += [self.ignored_index] * len(content_ids)

            elif role == "assistant":
                if self.tokenizer.eos_token_id == 73440:
                    input_ids += self.tokenizer.encode(content,add_special_tokens=False)
                    label_ids += self.tokenizer.encode(content,add_special_tokens=False)
                else:
                    input_ids += content_ids
                    label_ids += content_ids

        input_ids.append(self.tokenizer.eos_token_id)
        label_ids.append(self.tokenizer.eos_token_id)

        ## truncate to max len
        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]

        attention_mask = [1] * len(input_ids)

        ## pad to max len
        input_ids += [self.tokenizer.eos_token_id] * (self.model_max_length - len(input_ids))
        label_ids += [self.ignored_index] * (self.model_max_length - len(label_ids))

        attention_mask += [0] * (self.model_max_length - len(attention_mask))

        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        attention_mask = torch.LongTensor(attention_mask)
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self._process_data(self.data[idx])


def train(
        model_path: str = "openbmb/MiniCPM-2B-sft-bf16",
        output_dir: str = "models/MiniCPM-2B-sft-bf16/",
        train_data_path: str = "/home/mark/projects/models_ft/data/AdvertiseGenChatML/train.json",
        eval_data_path: str = "/home/mark/projects/models_ft/data/AdvertiseGenChatML/dev.json",
        device_map: str = "cuda",
        max_steps: int = 0,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_max_length: int = 1024 * 4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int = 1,
        learning_rate: float = 5e-5,

        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        targe_modules: List[str] = None,
):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = output_dir + f"{ts}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("tokenizer.pad_token", tokenizer.pad_token)
    print("tokenizer.eos_token_id", tokenizer.eos_token)
    print("tokenizer.length", len(tokenizer))

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map=device_map,
                                                 trust_remote_code=True)

    # model.resize_token_embeddings(len(tokenizer))

    if not targe_modules:
        print("model.config.architectures--", model.config.architectures)
        targe_modules = ["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj"]
        if model.config.architectures == ["MiniCPMForCausalLM"]:
            targe_modules = ["q_proj", "kv_proj", ]

    lora_config = LoraConfig(
        init_lora_weights="gaussian",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targe_modules,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        bf16=True,
        gradient_accumulation_steps=4,
        # warmup_steps=100,
        num_train_epochs=num_train_epochs,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        eval_steps=100,
        seed=42,
        eval_strategy="steps",
        logging_steps=10,
        warmup_ratio=0.1,
        save_steps=10000,
        max_grad_norm=0.3,
        max_steps=max_steps,
    )

    train_dataset = SupervisedDataset(data_path=train_data_path, tokenizer=tokenizer, model_max_length=model_max_length)
    eval_dataset = SupervisedDataset(data_path=eval_data_path, tokenizer=tokenizer, model_max_length=model_max_length)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    return


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if platform.system() == "Darwin":
        device = "mps"

    train(model_path="openbmb/MiniCPM3-4B",
          train_data_path="/home/dxj/projects/models_ft/data/AdvertiseGenChatML/train.json",
          eval_data_path="/home/dxj/projects/models_ft/data/AdvertiseGenChatML/dev.json",
          output_dir="/home/dxj/projects/models_ft/models/MiniCPM3-4B-adv_gen",
          max_steps=2500,
          num_train_epochs=1,
          device_map=device,
          model_max_length=256,
          learning_rate=2e-5,
          per_device_train_batch_size=4,
          per_device_eval_batch_size=4
          )
