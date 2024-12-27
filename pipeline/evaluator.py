from typing import Any
import numpy as np
from scipy.optimize import minimize
from promptconstructor.array_to_txt import load_resample_burg_array
from extract_llm_equation import write_equation_v1_fun
from solution_complexity import eval_complexity
from promptconstructor.array_to_txt import Data
from promptconstructor.info_prompts import prompt_complete_inf
optimization_track = {}

def define_eq(response):
    eq1_fun_text = write_equation_v1_fun(response)
    exec(eq1_fun_text, globals())


def loss_function(params, t, x, u, derivs_dict, left_side):
    u_pred = equation_v1(t, x, u, derivs_dict, params)[0]
    return np.mean((u_pred-derivs_dict[left_side])**2)


def evaluate(data: dict, P: int, left_side: str = 'du/dt') -> tuple[Any, Any]:
    inputs, derivs_dict = data['inputs'], data["derivs_dict"]
    loss_partial = lambda params: loss_function(params, *inputs, derivs_dict, left_side)
    params_initial_guess = np.array([1.0]*P)
    result = minimize(loss_partial, params_initial_guess, method='BFGS')
    optimized_params = result.x
    score = loss_function(optimized_params, *inputs, derivs_dict, left_side)
    return score if not np.isnan(score) and not np.isinf(score) else None, result.x


def round_score(score):
    if score > 100:
        return int(score)
    elif score > 10:
        return np.round(score, 1)
    elif score > 1:
        return np.round(score, 2)
    else: return np.round(score, 3)


def piped_evaluator(response, dir_name='burg', resample_shape=(20, 20)):
    define_eq(response)
    data_for_eval = Data(dir_name, resample_shape=resample_shape)
    data = data_for_eval.eval_data
    left_side = prompt_complete_inf[data_for_eval.dir_name]['left_deriv']
    _, string_form_of_the_equation, P = equation_v1(*data['inputs'], data["derivs_dict"], np.zeros(100))
    score, params = evaluate(data, P, left_side)
    rounded_score = round_score(score)
    try:
        complexity_score = eval_complexity(string_form_of_the_equation)
    except Exception as e:
        print(f"\nException while finding a complexity score")
    total_score = round_score(np.sqrt(score*score + complexity_score*complexity_score))
    optimization_track[string_form_of_the_equation] = (float(rounded_score), complexity_score)
    return total_score, string_form_of_the_equation, params
