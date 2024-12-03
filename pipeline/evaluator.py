from typing import Any
import numpy as np
from scipy.optimize import minimize
from promptconstructor.array_to_txt import load_resample_array
from extract_llm_equation import write_equation_v1_fun


def define_eq(response):
    eq1_fun_text = write_equation_v1_fun(response)
    exec(eq1_fun_text, globals())


def loss_function(params, t, x, u, derivs_dict):
    u_pred = equation_v1(t, x, u, derivs_dict, params)[0]
    return np.mean((u_pred-derivs_dict["du/dt"])**2)


def evaluate(data: dict, P: int) -> tuple[Any, Any]:
    inputs, derivs_dict = data['inputs'], data["derivs_dict"]
    loss_partial = lambda params: loss_function(params, *inputs, derivs_dict)
    params_initial_guess = np.array([1.0]*P)
    result = minimize(loss_partial, params_initial_guess, method='BFGS')
    optimized_params = result.x
    score = loss_function(optimized_params, *inputs, derivs_dict)
    return score if not np.isnan(score) and not np.isinf(score) else None, result.x


def round_score(score):
    if score > 100:
        return int(score)
    elif score > 10:
        return np.round(score, 1)
    elif score > 1:
        return np.round(score, 2)
    else: return np.round(score, 3)


def piped_evaluator(response):
    define_eq(response)

    u, t, x = load_resample_array()
    u_t, _, _ = load_resample_array("du_dx0")
    u_x, _, _ = load_resample_array("du_dx1")
    grids = np.meshgrid(t, x, indexing='ij')
    data = {"inputs": [grids[0], grids[1], u], "derivs_dict": {"du/dt": u_t, "du/dx": u_x}}

    _, string_form_of_the_equation, P = equation_v1(*data['inputs'], data["derivs_dict"], np.zeros(100))
    score, params = evaluate(data, P)
    return round_score(score), string_form_of_the_equation, params



