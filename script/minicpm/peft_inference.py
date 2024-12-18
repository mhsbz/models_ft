import json

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

path = "openbmb/MiniCPM3-4B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


base_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
peft_model = PeftModel.from_pretrained(base_model,"/home/dxj/projects/LLaMA-Factory/saves/minicpm/adv_gen")

model = peft_model.merge_and_unload()
# model = base_model
## lora model:   蕾丝花边宫廷裙，优雅的蕾丝花边，将裙摆的层次感勾勒的非常清晰，宫廷的气息也随之而来。泡泡袖的袖口，更显甜美可爱，让穿着者更加的温柔可人。
## base model:   这款连衣裙采用蕾丝花边刺绣工艺，打造出宫廷风的优雅气质，展现出女性柔美的气质。大裙摆的裙摆设计，让裙子的视觉效果更加的华丽，穿起来更加的显瘦。裙子的泡泡袖设计，让穿起来更加的舒适。
## data content: 宫廷风的甜美蕾丝设计，清醒的蕾丝拼缝处，刺绣定制的贝壳花边，增添了裙子的精致感觉。超大的裙摆，加上精细的小花边设计，上身后既带着仙气撩人又很有女人味。泡泡袖上的提花面料，在细节处增加了浪漫感，春日的仙女姐姐。浪漫蕾丝布满整个裙身，美丽明艳，气质超仙。

messages = [
    # {"role":"user","content":"类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"},
    {"role": "user", "content": "类型#裙*材质#蕾丝*风格#宫廷*图案#刺绣*图案#蕾丝*裙型#大裙摆*裙下摆#花边*裙袖型#泡泡袖"},
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

model_outputs = model.generate(
    model_inputs,
    max_new_tokens=1024,
    top_p=0.8,
    temperature=0.7
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]

print(responses)
