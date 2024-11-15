import numpy as np
import os
import sys
from langchain_core.prompts import PromptTemplate
np.set_printoptions(threshold=sys.maxsize)


def format_arrays(path):
    matrix = np.load(path)
    mx_to_string = np.array2string(matrix, separator=',', max_line_width=1000)
    return mx_to_string


path_u = "D:\\Users\\Ivik12S\\Desktop\\PDE-Net 2.0\\matrices_burgers"
file = open("complete-prompt.txt", 'r')
prompt_raw = file.read()

path_u_full = os.path.join(path_u, 'u.npy')
u_array = format_arrays(path_u_full)

prompt_template = PromptTemplate.from_template(prompt_raw)
prompt = prompt_template.format(u_array=u_array)

print()