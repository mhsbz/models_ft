from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
torch.manual_seed(0)

base_path = 'openbmb/MiniCPM-1B-sft-bf16'
lora_path = '../../models/MiniCPM-1B-sft-bf1620241107_1502/checkpoint-1000'
tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)

model = PeftModel.from_pretrained(model,lora_path)
responds, history = model.chat(tokenizer, "a", temperature=0.1, top_p=0.8)
print(responds)
