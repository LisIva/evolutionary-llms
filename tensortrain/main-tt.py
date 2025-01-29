from openai import OpenAI
import numpy as np
import os
import sys
from pathlib import Path
np.set_printoptions(threshold=sys.maxsize)
import creds


PARENT_PATH = Path().absolute().parent
MODEL = "OMF-Qwen/Qwen2-Math-72B-Instruct"#OMF-Qwen/Qwen2-Math-72B-Instruct qwen/qwen-2-72b-instruct
PROMPT_NAME = "tensor-train1"

client = OpenAI(
    api_key=creds.api_key,
    base_url="https://api.vsegpt.ru/v1",)

file_path = os.path.join(f"{PROMPT_NAME}.txt")
with open(file_path, 'r') as file:
    prompt_raw = file.read()

messages = [{"role": "user", "content": prompt_raw}, ]

response_big = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.2,
    n=1,
    max_tokens=1500, # максимальное число ВЫХОДНЫХ токенов
    extra_headers={ "X-Title": "EPDE-LLM"},
)

response = response_big.choices[0].message.content
print("Response:", response)

# info(prompt_raw, response)
