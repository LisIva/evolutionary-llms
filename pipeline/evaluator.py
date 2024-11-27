from typing import Tuple, Any

import numpy as np
from scipy.optimize import minimize
import os
from promptconstructor.array_to_txt import load_resample_array
import pandas as pd
from pathlib import Path
import re
import cv2


def loss_function(params, t, x, u, derivs_dict):
    u_pred = equation_v1(t, x, u, derivs_dict, params)[0]
    return np.mean((u_pred-derivs_dict["du/dt"])**2)


def evaluate(data: dict, P: int) -> tuple[Any, Any]:
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
    return score if not np.isnan(score) and not np.isinf(score) else None, result.x


def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):
    right_side = params[0] * u * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"]
    string_form_of_the_equation = "du/dt = c[0] * du/dx + c[1] * u * du/dx"
    len_of_params = 2
    return (right_side, string_form_of_the_equation, len_of_params)


def piped_evaluator():
    u, t, x = load_resample_array()
    u_t, _, _ = load_resample_array("du_dx0")
    u_x, _, _ = load_resample_array("du_dx1")
    grids = np.meshgrid(t, x, indexing='ij')
    data = {"inputs": [grids[0], grids[1], u], "derivs_dict": {"du/dt": u_t, "du/dx": u_x}}

    _, string_form_of_the_equation, P = equation_v1(*data['inputs'], data["derivs_dict"], np.zeros(1000))
    score, params = evaluate(data, P)
    return score, string_form_of_the_equation, params

if __name__=='__main__':
    score, string_form_of_the_equation, params = piped_evaluator()
    print()
