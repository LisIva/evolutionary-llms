from typing import Any
import numpy as np
from scipy.optimize import minimize
from promptconstructor.array_to_txt import load_resample_burg_array
from extract_llm_responses import compose_equation_v1_fun
from solution_complexity import eval_complexity
from promptconstructor.array_to_txt import Data
from promptconstructor.info_prompts import prompt_complete_inf
from numpy import ndarray
from buffer_handler.eq_buffer import EqBuffer

eq_buffer = EqBuffer()


def define_eq(response):
    eq1_fun_text = compose_equation_v1_fun(response)
    exec(eq1_fun_text, globals())
    return eq1_fun_text


def loss_function(params, t, x, u, derivs_dict, left_side):
    u_pred = equation_v1(t, x, u, derivs_dict, params)[0]
    return np.mean((u_pred - derivs_dict[left_side]) ** 2)


def eval_metric(params, t, x, u, derivs_dict, left_side):
    u_pred = equation_v1(t, x, u, derivs_dict, params)[0]
    return np.mean(np.fabs(u_pred - derivs_dict[left_side]))


def evaluate(data: dict, P: int, left_side: str = 'du/dt') -> tuple[Any, Any, Any]:
    inputs, derivs_dict = data['inputs'], data["derivs_dict"]
    loss_partial = lambda params: loss_function(params, *inputs, derivs_dict, left_side)
    params_initial_guess = np.array([1.0]*P)
    result = minimize(loss_partial, params_initial_guess, method='BFGS')
    optimized_params = result.x
    # score = loss_function(optimized_params, *inputs, derivs_dict, left_side)
    loss_minimum = loss_function(optimized_params, *inputs, derivs_dict, left_side)
    score = eval_metric(optimized_params, *inputs, derivs_dict, left_side)
    return score if not np.isnan(score) and not np.isinf(score) else None, loss_minimum, result.x


def round_score(score):
    if score > 100:
        return int(score)
    elif score > 10:
        return np.round(score, 1)
    elif score > 1:
        return np.round(score, 2)
    else: return np.round(score, 3)


def piped_evaluator(response, dir_name='burg', resample_shape=(20, 20), debug_eval=False):
    if not debug_eval:
        eq_code = define_eq(response)
    data_for_eval = Data(dir_name, resample_shape=resample_shape)
    data = data_for_eval.eval_data
    left_side = prompt_complete_inf[data_for_eval.dir_name]['left_deriv']
    _, eq_text, P = equation_v1(*data['inputs'], data["derivs_dict"], np.zeros(100))
    score, loss, params = evaluate(data, P, left_side)
    try:
        complex_score = eval_complexity(eq_text)
    except Exception as e:
        print(f"\nException while finding a complexity score")

    u_t = data['derivs_dict'][left_side]
    relat_score = score / np.mean(np.fabs(u_t)) * 1000
    if not debug_eval:
        eq_buffer.push_record(eq_text, complex_score, relat_score, loss, eq_code)
    return round_score(relat_score), eq_text, params


# def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):
#     right_side = params[0] * derivs_dict["d^2u/dx^2"] + params[1] * derivs_dict[
#         "du/dx"]
#     string_form_of_the_equation = "du/dt = c[0] * t + c[1] * t * du/dx"
#     len_of_params = 2
#     return right_side, string_form_of_the_equation, len_of_params


if __name__ == '__main__':
    tot_score, string, params = piped_evaluator('', 'sindy-burg', debug_eval=True)

    print()