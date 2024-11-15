from openai import OpenAI
from langchain_core.prompts import PromptTemplate
import numpy as np
import os
import sys
from transformers import AutoTokenizer
np.set_printoptions(threshold=sys.maxsize)


def format_arrays(u: np.ndarray = None):
    if type(u) is not np.ndarray:
        path_u = "D:\\Users\\Ivik12S\\Desktop\\PDE-Net 2.0\\matrices_burgers"
        path_u_full = os.path.join(path_u, 'u.npy')
        u = np.load(path_u_full)

    u_to_string = np.array2string(u, separator=',', max_line_width=1000)
    return u_to_string


file = open("experimental-prompt.txt", 'r')
prompt_raw = file.read()

u = np.round(np.random.random((5, 5)), 2)

u_array = format_arrays(u)
prompt_template = PromptTemplate.from_template(prompt_raw)
prompt = prompt_template.format(u_array=u_array)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
out1 = tokenizer.decode([220, 15, 13, 23,])
print()
