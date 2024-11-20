from openai import OpenAI
from langchain_core.prompts import PromptTemplate
import numpy as np
import os
import sys
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
np.set_printoptions(threshold=sys.maxsize)
from data.models_desc import models
import creds
from promptconstructor.combine_txts import get_simple_prompt

# qwen/qwen-2-72b-instruct         0.14  / 1000 inp. symb. & 0.14  / 1000 out. symb.  32768 cont
# qwen/qwen-2.5-72b-instruct       0.6   / 1000 inp. symb. & 0.6   / 1000 out. symb.  128000 cont
# mistralai/mixtral-8x7b-instruct  0.045 / 1000 inp. symb. & 0.045 / 1000 out. symb.  32768 cont
# mistralai/mixtral-8x22b-instruct 0.15  / 1000 inp. symb. & 0.15  / 1000 out. symb.  65536 cont
# OMF-Qwen/Qwen2-Math-72B-Instruct 0.25  / 1000 inp. symb. & 0.25  / 1000 out. symb.

PARENT_PATH = Path().absolute().parent
MODEL = "qwen/qwen-2-72b-instruct"

# minimum для simple burgers ~14500 tokens
# frac{du}{dt} = c[0] \cdot t + c[1] \cdot \frac{du}{dx} + c[2] \cdot u + c[3] \cdot x
def info(prompt, response):
    def cost(prompt, response):
        return len(prompt) / 1000 * models[MODEL]["in_price"] + \
               len(response) / 1000 * models[MODEL]["out_price"]

    system_prompt_price = len("You are a helpful assistant") / 1000 * models[MODEL]["in_price"]
    print(f"\n\nPrice: {cost(prompt, response):.5f}")
    print(f"Price with system prompt: {cost(prompt, response) + system_prompt_price:.5f}")
    print(f"Len(in_symbols): {len(prompt)}")
    print(f"Length of tokens, total: {response_big.usage.prompt_tokens + response_big.usage.completion_tokens}")
    print(f"Len(out_symbols): {len(response)}")


client = OpenAI(
    api_key=creds.api_key,
    base_url="https://api.vsegpt.ru/v1",
)
prompt = get_simple_prompt()
messages = []
messages.append({"role": "user", "content": prompt})

response_big = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.5,
    n=1,
    max_tokens=1500, # максимальное число ВЫХОДНЫХ токенов
    extra_headers={ "X-Title": "My App"},
)

response = response_big.choices[0].message.content
# num_comp_tokens = response_big.usage.prompt_tokens
# num_comp_tokens = response_big.usage.completion_tokens
print("Response:", response)

info(prompt, response)
