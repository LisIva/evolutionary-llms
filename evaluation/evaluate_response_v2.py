from typing import Tuple, Any
import numpy as np
from scipy.optimize import minimize
from promptconstructor.array_to_txt import load_resample_array


def loss_function(params, t, x, u, derivs_dict):
    lam = 0.01
    l2 = lam * np.dot(params, params)
    l1 = lam * np.sum(np.abs(params))
    u_pred = equation_v1(t, x, u, derivs_dict, params)[0]
    return np.mean((u_pred-derivs_dict["du/dt"])**2) + l2


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
    return score if not np.isnan(score) and not np.isinf(score) else None, result.x


def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):
    # right_side = params[0] * u * derivs_dict["du/dx"] + params[1] * derivs_dict["du/dx"] # 1.64653
    # right_side = params[0] * u * derivs_dict["du/dx"] + params[1]  # 1.59917
    # right_side = params[0] * u * derivs_dict["du/dx"] # 1.67579
    # right_side = params[0] * u * derivs_dict["du/dx"] + params[1] * t * derivs_dict["du/dx"] # 1.66
    right_side = params[0] * u * derivs_dict["du/dx"] + params[1] * x * derivs_dict["du/dx"] # 1.351
# "du/dt = c[0] * u * du/dx + c[1] * x * du/dx": 1.34
    string_form_of_the_equation = ""
    return (right_side, string_form_of_the_equation)


if __name__ == '__main__':
    P = 2
    u, t, x = load_resample_array()
    u_t, _, _ = load_resample_array("du_dx0")
    u_x, _, _ = load_resample_array("du_dx1")
    grids = np.meshgrid(t, x, indexing='ij')
    data = {"inputs": [grids[0], grids[1], u], "derivs_dict": {"du/dt": u_t, "du/dx": u_x,
                                                               }}
    score, params = evaluate(data)
    print(score)
    print()
