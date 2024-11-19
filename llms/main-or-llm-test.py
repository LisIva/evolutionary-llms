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

# qwen/qwen-2-72b-instruct         0.14  / 1000 inp. symb. & 0.14  / 1000 out. symb.  32768 cont
# qwen/qwen-2.5-72b-instruct       0.6   / 1000 inp. symb. & 0.6   / 1000 out. symb.  128000 cont
# mistralai/mixtral-8x7b-instruct  0.045 / 1000 inp. symb. & 0.045 / 1000 out. symb.  32768 cont
# mistralai/mixtral-8x22b-instruct 0.15  / 1000 inp. symb. & 0.15  / 1000 out. symb.  65536 cont
# OMF-Qwen/Qwen2-Math-72B-Instruct 0.25  / 1000 inp. symb. & 0.25  / 1000 out. symb.

PARENT_PATH = Path().absolute().parent
MODEL = "qwen/qwen-2-72b-instruct"
PROMPT_NAME = "space-sense6"

'''
1. провести эксперименты с u_t = u*u_x: 
2. разобрать https://medium.com/@mrk5199/how-to-compress-llm-contexts-with-langchain-2b58eb84f57b
'''


def load_resample_array():
    path = os.path.join(PARENT_PATH, "data", "simple_burg", "u.npy")
    u = np.load(path)

    x, t = np.linspace(-1000, 0, 101), np.linspace(0, 1, 101)
    xi, ti = np.linspace(-1000, 0, 32), np.linspace(0, 1, 32)
    grids = np.meshgrid(ti, xi, indexing='ij')
    test_points = np.array([grids[0].ravel(), grids[1].ravel()]).T

    interp = RegularGridInterpolator([t, x], u)
    u_res = interp(test_points, method='linear').reshape(32, 32)
    return (np.array2string(np.round(u_res, 2), separator=',', max_line_width=1000),
            np.array2string(np.round(ti, 2), separator=',', max_line_width=1000),
            np.array2string(np.round(xi, 2), separator=',', max_line_width=1000))


def info(prompt, response):
    def cost(prompt, response):
        return len(prompt) / 1000 * models[MODEL]["in_price"] + \
               len(response) / 1000 * models[MODEL]["out_price"]

    system_prompt_price = len("You are a helpful assistant") / 1000 * models[MODEL]["in_price"]
    print(f"\n\nPrice: {cost(prompt, response):.5f}")
    print(f"Price with system prompt: {cost(prompt, response) + system_prompt_price:.5f}")
    print(f"Len(in_symbols): {len(prompt)}")
    print(f"Len(out_symbols): {len(response)}")


# Len(in_symbols): 4937 compression 0.2 in space6
# Len(out_symbols): 1145
client = OpenAI(
    api_key=creds.api_key,
    base_url="https://api.vsegpt.ru/v1",
)

file_path = os.path.join(PARENT_PATH, "prompts", "text-llms", f"{PROMPT_NAME}.txt") # os.path.join(os.getcwd(), "u.npy")
file = open(file_path, 'r')
prompt_raw = file.read()
# u_res, tr, xr = load_resample_array()
# prompt_template = PromptTemplate.from_template(prompt_raw)
# prompt = prompt_template.format(u_array=u_res, x_array=xr, t_array=tr)

messages = []
messages.append({"role": "user", "content": prompt_raw})

response_big = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.5,
    n=1,
    max_tokens=1000, # максимальное число ВЫХОДНЫХ токенов
    extra_headers={ "X-Title": "My App"},
)

response = response_big.choices[0].message.content
print("Response:", response)

info(prompt_raw, response)
