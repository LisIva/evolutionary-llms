import re
from promptconstructor.info_prompts import prompt_complete_inf
from pipeline.buffer_handler.code_parser import SympyConverter
import numpy as np
from sympy import expand, sympify
from pipeline.epde_translator.translation_structures import LLMPool
from pipeline.epde_translator.solution_translator import SolutionTranslator
import sympy
import warnings


class EpdeTranslator(object):
    def __init__(self, record_track: dict, populat_track: dict):
        self.record_track = record_track
        self.populat_track = populat_track
        self.llm_pool = LLMPool()

    def some(self):
        for sol_key in self.populat_track.keys():
            # sol_terms = strip(split_with_braces(sol_key))
            pass
            # вместо c[..] подставить реальные коэффы из record_track


# обработать случаи когда cos попадает в Pow и просто наравне с Symbol может быть
if __name__ == '__main__':
    pop_track = {'du/dt = c[0] * du/dx + c[1] * t * du/dx + c[2] * t * x': (1.6, 460.5686610664196), 'du/dt = c[0] * du/dx + c[1] * u + c[2] * d^2u/dx^2': (1.45, 484.1114426561667), 'du/dt = c[0] * du/dx + c[1] * u * du/dx': (1.2, 438.94292729549943), 'du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2': (1.95, 37.14800565887713), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2': (1.45, 38.90635312678824), 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2 + c[2] * du/dx * t': (2.15, 37.057907826954576), 'du/dt = c[0] * du/dx + c[1] * du/dt * d^2u/dx^2': (1.75, 542.9853705131861), 'du/dt = c[0] * du/dx + c[1] * t * du/dx': (1.2, 442.49077370655203)}
    rs4 = 'params[0] * derivs_dict["du/dx"] ** 3 + ((params[1] * derivs_dict["du/dx"] ** 2 * np.cos(params[2] * t +1)) * x**2) * u +params[3] * derivs_dict["du/dt"] * (t**2 + 2)'
    llm_pool = LLMPool()
    trans4 = SolutionTranslator(rs4, np.array([1.2, 5.678, 6.5421, 4.12]), llm_pool).translate()
    rs5 = 'params[0] * np.sqrt(u)'

    rs6 = '(arcsin((log(t) + 5 * x) * du_dt)) ** 2'
    express_pow = expand(sympify(rs6))
    term = express_pow.args[0]
    term_power = express_pow.args[1]
    base_name = str(term.func)
    term_str = str(term)
    t = 5
    allowed_namespace = {'np': np,}
    lambda_str = f"lambda t: np.{term_str}"
    np_fun = eval(lambda_str, allowed_namespace)

    # custom_trigonometric_eval_fun = {
    #     'cos(t)': lambda *grids, **kwargs: np.cos(grids[0]) ** kwargs['power'],
    #     'sin(x)': lambda *grids, **kwargs: np.sin(grids[1]) ** kwargs['power']}
    # custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
    #                                         eval_fun_params_labels=['power'])

    # trig_params_ranges = {'power': (1, 1)}


    expression1 = expand(sympify('sin(t)'))
    exb11 = expression1.is_Symbol
    ex1 = expression1.func.__name__
    ex111 = str(expression1.func)

    expression2 = sympify('du_dt')
    exb21 = expression2.is_Symbol
    ex2 = expression2.name
    trans5 = SolutionTranslator(rs5, np.array([1.2, ]), llm_pool).translate()
    print()