import re
from promptconstructor.info_prompts import prompt_complete_inf
from pipeline.buffer_handler.code_parser import SympyConverter
import numpy as np
from sympy import expand, sympify
import sympy
# class Individ(object):
#     def __init__(self, sympy_form, epde_form):
#         self.sympy_form = sympy_form

class Family(object):
    def __init__(self, family_type, tokens):
        self.family_type = family_type
        self.tokens = tokens


class LLMPool(object):
    def __init__(self):
        self.simple_tokens_pow = {'t': 0, 'x': 0, }
        self.max_deriv_orders = {'max_deriv_t': 1, 'max_deriv_x': 1}
        self.special_tokens_pow = {} # key: как епде будет выводить токен юзеру, val - лямбда-функция как его посчитать

    def set_token_pow(self, token, value):
        self.simple_tokens_pow[token] = value

    def set_max_d_order(self, token, value):
        self.max_deriv_orders[token] = value


class BaseTokenConverter(object):
    def __init__(self, token):
        self.token = str(token)

    def convert(self, pool: LLMPool):
        if self.token == 'u':
            return 'u'
        elif self.token == 't':
            pool.set_token_pow('t', 1)
            return 'x1'
        elif self.token == 'x':
            pool.set_token_pow('x', 1)
            return 'x2'
        elif self.token == 'du_dt':
            return 'du/dx1'
        elif self.token == 'du_dx':
            return 'du/dx2'
        elif len(self.token) == 7:
            return self.__convert_high_deriv(pool)

    def __convert_high_deriv(self, pool: LLMPool):
        var = self.token[-2]
        n = self.token[-1]

        if pool.max_deriv_orders[f'max_deriv_{var}'] < n:
            pool.set_max_d_order(f'max_deriv_{var}', n)
        return f'd^{n}u/dx1^{n}' if var == 't' else f'd^{n}u/dx2^{n}'


class FunConverter(object):
    def __init__(self, term, power, llm_pool):
        self.term = term
        self.power = power
        self.llm_pool = llm_pool

        if not self.is_valid_expression(self.term.args[0]):
            raise Exception('Cannot convert nested numpy functions in suggested solution')

    def is_valid_expression(self, expr):
        if isinstance(expr, sympy.Symbol):
            return True
        if isinstance(expr, sympy.Add):
            for arg in expr.args:
                if not self.is_valid_expression(arg):
                    return False
            return True
        if isinstance(expr, sympy.Mul):
            for arg in expr.args:
                if not self.is_valid_expression(arg):
                    return False
            return True
        if isinstance(expr, (sympy.Float, sympy.Integer)):
            return True
        return False


class MulConverter(object):
    def __init__(self, mul_term, llm_pool):
        self.mul_term = mul_term
        self.llm_pool = llm_pool

    def to_epde(self):
        mul_str = [str(self.mul_term.args[0]), ]
        for token in self.mul_term.args[1:]:
            if token.is_Symbol:
                epde_token = BaseTokenConverter(token).convert(self.llm_pool) + '{power: 1.0}'
            elif token.is_Pow:
                if token.args[0].is_Symbol:
                    if token.args[1].is_Half:
                        epde_token = BaseTokenConverter(token.args[0]).convert(self.llm_pool) \
                                     + '{power: 0.5}'
                    else:
                        epde_token = BaseTokenConverter(token.args[0]).convert(self.llm_pool) \
                                     + '{power: ' + str(token.args[1]) + '.0}'
                else:
                    pass
            else:
                pass
            mul_str.append(epde_token)
        return mul_str


class SolutionTranslator(object):
    def __init__(self, rs_code, params, llm_pool):
        sym_converter = SympyConverter(rs_code, params)

        self.rs_code = sym_converter.rs_code
        self.sympy_code = sym_converter.sympy_code
        self.llm_pool = llm_pool

    def translate(self):

        if self.sympy_code.is_Add:
            for mul_term in self.sympy_code.args:
                mul_conv = MulConverter(mul_term, self.llm_pool).to_epde()
        elif self.sympy_code.is_Mul:
            mul_conv = MulConverter(self.sympy_code, self.llm_pool).to_epde()
        else:
            raise Exception('Unexpected form in sympy structure: could not find Add or Mul to convert')
        print()


# 2. Написать еще {power: 1} и разобрать отдельно кейсы с ** (тут {power: n})
# 4. Pool возвращает необычные элементы (t, x, sin(t), ...) - тут важно что они есть и их надо отправить
#    в динамический код в епде токенс
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
    rs4 = 'params[0] * derivs_dict["du/dx"] ** 3 + ((params[1] * derivs_dict["du/dx"] ** 2 * np.cos(params[2] * u +1)) * x**2) * u +params[3] * derivs_dict["du/dt"] * (t**2 + 2)'
    llm_pool = LLMPool()
    # trans4 = SolutionTranslator(rs4, np.array([1.2, 5.678, 6.5421, 4.12]), llm_pool).translate()
    rs5 = 'params[0] * (np.arcsin(t) + u) ** 2'

    rs6 = '(arcsin((log(t) + 5 * x) * du_dt)) ** 2'
    express_pow = expand(sympify(rs6))
    term = express_pow.args[0]
    term_power = express_pow.args[1]
    base_name = str(term.func)
    term_str = str(term)
    t = 5
    allowed_namespace = {'np': np,}

    is_subset = term.free_symbols.issubset({sympy.symbols('t'), sympy.symbols('x')})
    contains_t = sympy.symbols('t') in term.free_symbols
    lambda_str = f"lambda t: np.{term_str}"
    # lambda_str = f"lambda t: np.{term_str}"
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