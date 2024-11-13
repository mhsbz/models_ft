from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
)

model.to("cpu")

token_ids = tokenizer.encode("你好,我是一个聊天机器人", add_special_tokens=True, max_length=10, padding=True,
                             truncation=True)

# for item in token_ids:
#     print(item, tokenizer.decode(item))
#
# print(token_ids)
tokens = tokenizer.convert_ids_to_tokens(token_ids)

messages = [
    {
        "role": "system",
        "content": "You are Mohan, created by duan xu jie. You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "你是谁？"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

pt_token_ids = tokenizer([text], add_special_tokens=True, max_length=128, padding="max_length",
                         truncation=True, return_tensors="pt")

# print(pt_token_ids)
print(tokenizer.decode(pt_token_ids['input_ids'][0]))

# print(pt_token_ids)
#
# for item in pt_token_ids['input_ids']:
#     print(item, tokenizer.decode(item))

generated_ids = model.generate(**pt_token_ids, max_new_tokens=512, temperature=0.9)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(pt_token_ids.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
