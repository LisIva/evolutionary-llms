from typing import Any
import numpy as np
from scipy.optimize import minimize
from promptconstructor.array_to_txt import load_resample_burg_array
from extract_llm_equation import write_equation_v1_fun
from solution_complexity import eval_complexity
from promptconstructor.array_to_txt import Data
from promptconstructor.info_prompts import prompt_complete_inf
optimization_track = {}
optimization_track_tripods = {}


def define_eq(response):
    eq1_fun_text = write_equation_v1_fun(response)
    exec(eq1_fun_text, globals())


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
        define_eq(response)
    data_for_eval = Data(dir_name, resample_shape=resample_shape)
    data = data_for_eval.eval_data
    left_side = prompt_complete_inf[data_for_eval.dir_name]['left_deriv']
    _, string_form_of_the_equation, P = equation_v1(*data['inputs'], data["derivs_dict"], np.zeros(100))
    score, loss, params = evaluate(data, P, left_side)

    try:
        complexity_score = eval_complexity(string_form_of_the_equation)
    except Exception as e:
        print(f"\nException while finding a complexity score")

    u_t = data['derivs_dict'][left_side]
    minim = np.abs(np.min(u_t))
    maxim = np.abs(np.max(u_t))
    mean = np.mean(np.fabs(u_t))
    mean2 = np.mean(u_t*u_t)
    if minim == 0:
        minim += 0.00001
    t1 = score / minim * 1000
    t2 = score / maxim * 1000

    relat_mean = score / mean * 1000
    optimization_track_tripods[string_form_of_the_equation] = (float(round_score(t1)), float(round_score(t2)), float(round_score(relat_mean)))

    total_score = (loss / mean2 * 1000 + relat_mean) / 2
    rounded_relat_score = round_score(relat_mean)

    rounded_tot_score = round_score(total_score)
    optimization_track[string_form_of_the_equation] = (float(rounded_tot_score), complexity_score)
    if not debug_eval:
        return rounded_relat_score, string_form_of_the_equation, params
    else:
        return score, string_form_of_the_equation, params, u_t


# def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):
#     right_side = params[0] * derivs_dict["d^2u/dx^2"]
#                  # + params[1] * np.cos(t) * np.sin(x) + params[2] * derivs_dict["d^3u/dx^3"]
#     string_form_of_the_equation = "du/dt = c[0] * t + c[1] * t * du/dx"
#     len_of_params = 1
#     return right_side, string_form_of_the_equation, len_of_params


# def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):
#     right_side = params[0] * derivs_dict["d^2u/dx^2"]
#                  # + params[1] * np.cos(t) * np.sin(x) + params[2] * derivs_dict["d^3u/dx^3"]
#     string_form_of_the_equation = "du/dt = c[0] * t + c[1] * t * du/dx"
#     len_of_params = 3
#     return right_side, string_form_of_the_equation, len_of_params


if __name__ == '__main__':
    # Понять почему такой низкий скор у kdv
    "du/dt = c[0] * du/dx + c[1]"
    tot_score, string, params, u_t = piped_evaluator('', 'wave', debug_eval=True)
    minim = np.abs(np.min(u_t))
    maxim = np.abs(np.max(u_t))
    mean = np.mean(np.fabs(u_t))
    t1 = tot_score / minim * 1000
    t2 = tot_score / maxim * 1000
    t3 = tot_score / mean * 1000
    # optimization_track_tripods[string] = (float(round_score(t1)), float(round_score(t2)), float(round_score(t3)))
    # 229, 105, 835
    print()