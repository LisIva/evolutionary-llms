from typing import Tuple, Any

import numpy as np
from scipy.optimize import minimize
import os
import pandas as pd
from pathlib import Path
import re
import cv2


def set_u_grid(size=64):
    x = np.linspace(0, 4, size)
    t = np.linspace(0, 10, size)
    grids = np.meshgrid(t, x, indexing='ij')
    u_ideal = grids[1] ** 2 + 100*np.cos(grids[0])
    return grids, u_ideal


def loss_function(params, x0, x1, u_true):
    u_pred = equation_v1(x0, x1, params)
    return np.mean((u_pred-u_true)**2)


def evaluate(data: dict) -> float:
    """ Evaluate equation of input and output observations."""
    inputs, outputs = data['inputs'], data['outputs']

    loss_partial = lambda params: loss_function(params, *inputs, outputs)
    params_initial_guess = np.array([1.0]*P)
    result = minimize(loss_partial, params_initial_guess, method='BFGS')
    optimized_params = result.x

    score = loss_function(optimized_params, *inputs, outputs)
    return -score if not np.isnan(score) and not np.isinf(score) else None


def equation_v1(t: np.ndarray, x: np.ndarray, params: np.ndarray) -> np.ndarray:
    right_side = (params[0] * t ** 2 + params[1] * x + params[2] * t + params[3] * t * x + params[4] * t * x**2 + params[5] * t * x**3 + params[6] * t * x**4 + params[7] * t * x**5 + params[8] * t * x**6 + params[9] * t * x**7) * np.sin(x)

    return right_side


if __name__ == '__main__':

    P = 10
    grids, u = set_u_grid(64)

    data = {"inputs": [grids[0], grids[1]], "outputs": u}
    score = evaluate(data)
    print(score)

# def equation_v1(t: np.ndarray, x: np.ndarray, params: np.ndarray) -> np.ndarray:
#     right_side = (params[0] * t ** 2 + params[1] * x + params[2] * t + params[3] * t * x + params[4] * t * x**2 + params[5] * t * x**3 + params[6] * t * x**4 + params[7] * t * x**5 + params[8] * t * x**6 + params[9] * t * x**7) * np.sin(x)
#     string_form_of_the_equation = "u = (c[0]*t^2 + c[1]*x + c[2]*t + c[3]*t*x + c[4]*t*x^2 + c[5]*t*x^3 + c[6]*t*x^4 + c[7]*t*x^5 + c[8]*t*x^6 + c[9]*t*x^7) * sin(x)"
#
#     return (right_side, string_form_of_the_equation)
