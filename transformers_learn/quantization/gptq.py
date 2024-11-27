from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained(model_id, config=gptq_config, trust_remote_code=True).to("cuda")

quantized_model.save_pretrained("Qwen/Qwen2.5-1.5B-Instruct-quantized")
