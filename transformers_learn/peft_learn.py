from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.save_pretrained("./peft_model")
