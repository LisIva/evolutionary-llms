import os
from langchain_core.prompts import PromptTemplate
import numpy as np
import sys
from pathlib import Path
np.set_printoptions(threshold=sys.maxsize)


PARENT_PATH = Path().absolute().parent


def read_simple_burg():
    abs_path = os.path.join(PARENT_PATH, "promptconstructor", "burg_txu_derivs.txt")
    with open(abs_path, 'r') as myf:
        data = myf.read()
    return data


def get_simple_burg_prompt():
    head = "What is a possible function with the general equation form du/dt = F(t, x, u, du/dx)) that could be described with the set of points," \
           " that have the form of 't x u du/dx du/dt':\n"
    data = read_simple_burg()
    tail = "Example of output:\n" \
           "du/dt = c[0] * t + c[1] * du/dx, where c are some coefficients"
    return head + data + tail


def read_with_langchain(prompt_name, type='simple-burg', print_baselen=False):
    if type=='simple-burg':
        data = read_simple_burg()
        abs_path = os.path.join(PARENT_PATH, "prompts", "text-llms", prompt_name)
        with open(abs_path, 'r') as myf:
            prompt_raw = myf.read()

    prompt_template = PromptTemplate.from_template(prompt_raw)
    prompt = prompt_template.format(points_set=data)

    if print_baselen:
        print("Len of prompt without input data:", len(prompt)-len(data))
    return prompt


if __name__ == "__main__":
    prompt = read_with_langchain("points-set-prompt3.txt", print_baselen=True)
    print()