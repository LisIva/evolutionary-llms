import os
from langchain_core.prompts import PromptTemplate
import numpy as np
import sys
from pathlib import Path
from promptconstructor.info_prompts import prompt_complete_inf

PARENT_PATH = Path().absolute().parent

if 'pipeline' in PARENT_PATH.parts:
    PARENT_PATH = PARENT_PATH.parent
# right_side = params[0] * u * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"]
# string_form_of_the_equation = "du/dt = c[0] * du/dx + c[1] * u * du/dx"


def read_eq_data(name):
    abs_path = os.path.join(PARENT_PATH, "promptconstructor", f"{name}_txu_derivs.txt")
    with open(abs_path, 'r') as myf:
        data = myf.read()
    return data


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


def read_with_langchain(dir_name='burg', path=None):
    data = read_eq_data(dir_name)
    with open(path, 'r') as myf:
        prompt_raw = myf.read()

    prompt_template = PromptTemplate.from_template(prompt_raw)
    prompt = prompt_template.format(points_set=data,
                                    dots_order=prompt_complete_inf[dir_name]['dots_order'],
                                    left_deriv=prompt_complete_inf[dir_name]['left_deriv'],
                                    full_form=prompt_complete_inf[dir_name]['full_form'],)
    return prompt


def test_read_with_langchain(prompt_name="points-set-prompt3.txt", dir_name='burg'):
    data = read_eq_data(dir_name)
    # path = os.path.join(PARENT_PATH, "prompts", "text-llms", prompt_name)
    path = 'reset-for-continue.txt'
    with open(path, 'r') as myf:
        prompt_raw = myf.read()

    prompt_template = PromptTemplate.from_template(prompt_raw)
    prompt = prompt_template.format(points_set=data,
                                    dots_order=prompt_complete_inf[dir_name]['dots_order'],
                                    left_deriv=prompt_complete_inf[dir_name]['left_deriv'],
                                    full_form=prompt_complete_inf[dir_name]['full_form'],)
    return prompt


if __name__ == "__main__":
    # prompt = test_read_with_langchain()
    prompt1 = test_read_with_langchain(dir_name='wave')
    print()