import os

import numpy as np
import sys
from pathlib import Path
np.set_printoptions(threshold=sys.maxsize)


PARENT_PATH = Path().absolute().parent


def get_simple_prompt():
    head = "What is a possible function with the general equation form du/dt = F(t, x, u, du/dx)) that could be described with the set of points," \
           " that have the form of 't x u du/dx du/dt':\n"
    abs_path = os.path.join(PARENT_PATH, "promptconstructor", "burg_txu_derivs.txt")
    with open(abs_path, 'r') as myf:
        data = myf.read()

    tail = "Example of output:\n" \
           "du/dt = c[0] * t + c[1] * du/dx, where c are some coefficients"

    return head + data + tail


if __name__ == "__main__":
    head = "What is a possible function with the general equation form du/dt = F(t, x, u, du/dx)) that could be described with the set of points," \
           " that have the form of 't x u du/dx du/dt':\n"
    with open("burg_txu_derivs.txt", 'r') as myf:
        data = myf.read()

    propmt = head + data
    print(os.getcwd())