import re
from promptconstructor.info_prompts import prompt_complete_inf
from pipeline.buffer_handler.code_parser import SympyConverter
import numpy as np
from sympy import expand, sympify
import sympy
import warnings
from pipeline.epde_translator.translation_structures import LLMPool


class BaseTokenConverter(object):
    def __init__(self, token, power):
        self.token = str(token)
        self.power = power

    def convert(self, pool: LLMPool):
        if self.token == 'u':
            return 'u'
        elif self.token == 't':
            if pool.simple_tokens_pow['t'] < self.power:
                pool.set_token_pow('t', self.power)
            return 't'
        elif self.token == 'x':
            if pool.simple_tokens_pow['x'] < self.power:
                pool.set_token_pow('x', self.power)
            return 'x'
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
    def __init__(self, term, power):
        self.term = term
        self.power = power
        self.correct_function = True

        if not self.is_valid_expression(self.term.args[0]):
            self.correct_function = False
            warnings.warn('Cannot convert nested numpy functions in suggested solution', UserWarning)

        if not self.has_valid_variables():
            self.correct_function = False
            warnings.warn("Cannot convert a sympy function dependant on variables other than t and x", UserWarning)

    def convert(self, llm_pool):
        if self.correct_function:
            if llm_pool.special_tokens_pow.get(self.term) is not None:
                if llm_pool.special_tokens_pow[self.term] < self.power:
                    llm_pool.set_special_token_pow(self.term, self.power)
            else:
                llm_pool.set_special_token_pow(self.term, self.power)
            return str(self.term)
        else:
            return None

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

    def has_valid_variables(self):
        return self.term.free_symbols.issubset({sympy.Symbol('t'), sympy.Symbol('x')})


class MulConverter(object):
    def __init__(self, mul_term, llm_pool):
        self.mul_term = mul_term
        self.llm_pool = llm_pool

    def to_epde(self):
        mul_str = [str(self.mul_term.args[0]), ]
        for token in self.mul_term.args[1:]:
            if token.is_Symbol:
                epde_token = BaseTokenConverter(token, 1).convert(self.llm_pool) + '{power: 1.0}'
            elif token.is_Pow:
                if isinstance(token.args[1], sympy.Integer):
                    if token.args[0].is_Symbol:
                        converted_token = BaseTokenConverter(token.args[0], int(token.args[1])).convert(self.llm_pool)
                    else:
                        converted_token = FunConverter(token.args[0], int(token.args[1])).convert(self.llm_pool)
                        if converted_token is None:
                            return None
                    epde_token = converted_token + '{power: ' + str(token.args[1]) + '.0}'

                else:
                    warnings.warn("Floating power is not supported", UserWarning)
                    return None
            else:
                fun_token = FunConverter(token, 1).convert(self.llm_pool)
                if fun_token is None:
                    return None
                epde_token = fun_token + '{power: 1.0}'
            mul_str.append(epde_token)
        return ' * '.join(mul_str)


class SolutionTranslator(object):
    def __init__(self, rs_code, params, llm_pool, dir_name):
        sym_converter = SympyConverter(rs_code, params)
        self.left_deriv = prompt_complete_inf[dir_name]['left_deriv']
        self.rs_code = sym_converter.rs_code
        self.sympy_code = sym_converter.sympy_code
        self.llm_pool = llm_pool

    def translate(self):
        add_str = []
        left_side = ' = ' + BaseTokenConverter(self.left_deriv, 1).convert(self.llm_pool) + '{power: 1.0}'
        if self.sympy_code.is_Add:
            for mul_term in self.sympy_code.args:
                converted_term = MulConverter(mul_term, self.llm_pool).to_epde()
                if converted_term is None:
                    return None
                add_str.append(converted_term)
            return " + ".join(add_str) + left_side
        elif self.sympy_code.is_Mul:
            converted_term = MulConverter(self.sympy_code, self.llm_pool).to_epde()
            return None if converted_term is None else converted_term + left_side
        else:
            raise Exception('Unexpected form in sympy structure: could not find Add or Mul to convert')


if __name__ == '__main__':
    pop_track = {'du/dt = c[0] * du/dx + c[1] * t * du/dx + c[2] * t * x': (1.6, 460.5686610664196),
                 'du/dt = c[0] * du/dx + c[1] * u + c[2] * d^2u/dx^2': (1.45, 484.1114426561667),
                 'du/dt = c[0] * du/dx + c[1] * u * du/dx': (1.2, 438.94292729549943),
                 'du/dt = c[0] * u * du/dx + c[1] * du/dx + c[2] * d^2u/dx^2': (1.95, 37.14800565887713),
                 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2': (1.45, 38.90635312678824),
                 'du/dt = c[0] * u * du/dx + c[1] * d^2u/dx^2 + c[2] * du/dx * t': (2.15, 37.057907826954576),
                 'du/dt = c[0] * du/dx + c[1] * du/dt * d^2u/dx^2': (1.75, 542.9853705131861),
                 'du/dt = c[0] * du/dx + c[1] * t * du/dx': (1.2, 442.49077370655203)}
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
    allowed_namespace = {'np': np, }
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