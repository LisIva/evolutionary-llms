from typing import Tuple, Any

import numpy as np
from scipy.optimize import minimize
import os
import pandas as pd
from pathlib import Path
import re
import cv2

''' 
1. привести выдаваемую форму qwen к 'du/dt = c[0] * x * du/dx + ...'??? (могут быть логические проблемы у qwen о том 
    как оценивается уравнение). Иначе - проблемы с уникальностью уравнения в рамках exp_buffer

2. изучить устройство qwen и возможно написать add inf что модель должна поглядывать на выходы своей архитектуры
3. Идея: прописывать что 255 это max(u_true), а 0 относится к min(u_true)? [Возможно не сработает, 
        так как нейросеть сама делает кучу преобразований (возможно надо протестить на микропримерах?)]
        \\e.g. what is the approximate actual value of the function u with the pixel that
                                            is located exactly at the center of the image?\\
4. Также приводить тепловые карты для x и t
5. Удостовериться что qwen разбирается в пространственных отношениях (это слабое место gpt4o)'''

def load_derivs():
    path = os.path.join(os.getcwd(), "data", "simple_burg")
    u = np.load(os.path.join(path, "u.npy"))
    u_t = np.load(os.path.join(path, "du_dx0.npy"))
    u_x = np.load(os.path.join(path, "du_dx1.npy"))
    u_xx = np.load(os.path.join(path, "d^2u_dx1^2.npy"))
    u_tt = np.load(os.path.join(path, "d^2u_dx0^2.npy"))
    return u, u_t, u_x, u_tt, u_xx


def load_from_csv(name: str = "burgers_sln_100.csv"):
    path_full = os.path.join(os.getcwd(), "data", name)
    df = pd.read_csv(path_full, header=None)
    return df.values.T


def eq_pyth_to_text(equation: str, left_side: str):
    # equation = "params[0] * u * derivs_dict['du/dx'] + params[1] * derivs_dict['d^2u/dt^2'] * t + params[2] * x * cos(x)"
    # left_side = "d^2u/dt^2"
    idx = 0
    eq_text = []

    terms = equation.split(" + ")
    for term in terms:
        factors = term.split(" * ")
        factors2text = [f"c[{idx}]", ]
        for j in range(1, len(factors)):
            deriv = re.findall("derivs_dict\['(.+/.+)'\]", factors[j])
            if len(deriv) != 0:
                factors2text.append(deriv[0])
            else:
                deriv_no_quotes = re.findall("derivs_dict\[(.+/.+)\]", factors[j])
                if len(deriv_no_quotes) != 0:
                    factors2text.append(deriv_no_quotes[0])
                else:
                    factors2text.append(factors[j])
        eq_text.append(" * ".join(factors2text))
        idx += 1
        print()
    eq_str = f"{left_side} = " + " + ".join(eq_text)
    return eq_str


def loss_function(params, t, x, u, derivs_dict):
    u_pred, left_deriv_name = equation_v1(t, x, u, derivs_dict, params)
    return np.mean((u_pred - derivs_dict[left_deriv_name])**2)


def evaluate(data: dict) -> tuple[Any, Any]:
    """ Evaluate the constructed equation"""
    # Load true data observations
    inputs, derivs_dict = data['inputs'], data["derivs_dict"]

    # Optimize equation skeleton parameters
    loss_partial = lambda params: loss_function(params, *inputs, derivs_dict)
    params_initial_guess = np.array([1.0]*P)
    result = minimize(loss_partial, params_initial_guess, method='BFGS')
    optimized_params = result.x

    # Return evaluation score
    score = loss_function(optimized_params, *inputs, derivs_dict)
    return -score if not np.isnan(score) and not np.isinf(score) else None, result.x


def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict, params: np.ndarray):
    right_side = params[0] * derivs_dict["du/dt"] + params[1] * derivs_dict["du/dx"] + params[2] * u * t
    return (right_side, "d^2u/dt^2")


if __name__ == '__main__':

    P = 3
    # rs = "params[0] * u + params[1] * derivs_dict[du/dx] + params[2] * t * derivs_dict[du/dx]"
    # eq_str = eq_pyth_to_text(rs, "du/dt")

    x = np.linspace(-1000, 0, 101)
    t = np.linspace(0, 1, 101)
    grids = np.meshgrid(t, x, indexing='ij')

    u, u_t, u_x, u_tt, u_xx = load_derivs()

    data = {"inputs": [grids[0], grids[1], u], "derivs_dict": {"du/dt": u_t, "du/dx": u_x,
                                                               "d^2u/dt^2": u_tt, "d^2u/dx^2": u_xx}}
    score, params = evaluate(data)
    print(score)
