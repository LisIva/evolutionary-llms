import os
from langchain_core.prompts import PromptTemplate
import numpy as np
import sys
from pathlib import Path

PARENT_PATH = Path().absolute().parent


def find_positions(path: str = 'llm-output.txt', encoding: str = None):
    with open(path, 'r', encoding=encoding) as myf:
        context = myf.read()
    begin_pos = context.find("def equation_v1(")
    end_pos = context.find("return (")
    return begin_pos, end_pos, context


def replace_evaluate_code():
    begin_pos, end_pos, context = find_positions(encoding="utf-8")
    text_to_replace = context[begin_pos:end_pos]

    # eval_path = os.path.join(PARENT_PATH, "evaluation", "evaluate_response_v2.py")
    begin_eval, end_eval, eval_file = find_positions('evaluator.py')
    new_eval_file = eval_file[:begin_eval] + text_to_replace + eval_file[end_eval:]
    with open('evaluator.py', 'w') as myf:
        myf.write(new_eval_file)



if __name__ == "__main__":
    # string = "print('This is a string')\ndef equation_v1(arg1, arg2):\n\t\tprint()\nreturn (1, 2, 3)\nprint('Function ended')"
    # print(string)
    # print('\n\n\n New string:\n')
    # begin_pos = string.find("def equation_v1(")
    # end_pos = string.find("return (")
    #
    # string_for_replace = "def equation_v1(arg1, arg2):\n\t\tx1 = 10\n\t\tx2 = 20\n"
    # new_string = string[:begin_pos] + string_for_replace + string[end_pos:]
    # print(new_string)
    replace_evaluate_code()
    # with open('my_script.py') as f:
    #     print(f.read())
