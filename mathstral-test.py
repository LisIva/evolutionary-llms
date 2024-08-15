from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from time import time

# checkpoint = "mistralai/Mathstral-7b-v0.1"
checkpoint = os.path.join(os.getcwd(), 'mathstral_model')
checkpoint_token = os.path.join(os.getcwd(), 'mathstral_tokenizer')

t1 = time()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_token)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

prompt = [{"role": "user", "content": "What is x in equation x^2 + 2x + 5 = 0?"}]
tokenized_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

out = model.generate(**tokenized_prompt, max_new_tokens=512)
print(tokenizer.decode(out[0]))
t2 = time()
print(f"Time spent, m: {(t2-t1) / 60:.2f}")
