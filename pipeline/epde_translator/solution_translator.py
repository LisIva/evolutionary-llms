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
        def set_tx_power(key):
            if pool.simple_tokens_pow[key][0] == 0:
                pool.set_token_pow(key, self.power, self.power)
            elif pool.simple_tokens_pow[key][0] > self.power:
                pool.set_token_pow(key, self.power, pool.simple_tokens_pow[key][1])
            elif pool.simple_tokens_pow[key][1] < self.power:
                pool.set_token_pow(key, pool.simple_tokens_pow[key][0], self.power)

        def set_derivs_pow(key):
            if pool.max_deriv_pow[key] < self.power:
                pool.max_deriv_pow[key] = self.power

        if self.token == 'u':
            set_derivs_pow('data_fun_pow')
            return 'u'
        elif self.token == 't':
            set_tx_power('t')
            return 't'
        elif self.token == 'x':
            set_tx_power('x')
            return 'x'
        else:
            set_derivs_pow('deriv_fun_pow')
            if self.token == 'du_dt':
                return 'du/dx1'
            elif self.token == 'du_dx':
                return 'du/dx2'
            elif len(self.token) == 7:
                return self.__convert_high_deriv(pool)

    def __convert_high_deriv(self, pool: LLMPool):
        var = self.token[-2]
        n = int(self.token[-1])

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

    def convert(self, pool):
        if self.correct_function:
            if pool.special_tokens_pow.get(self.term) is not None:
                if pool.special_tokens_pow[self.term][0] > self.power:
                    pool.set_special_token_pow(self.term, self.power, pool.special_tokens_pow[self.term][1])
                elif pool.special_tokens_pow[self.term][1] < self.power:
                    pool.set_special_token_pow(self.term, pool.special_tokens_pow[self.term][0], self.power)
            else:
                pool.set_special_token_pow(self.term, self.power, self.power)
            return str(self.term)
        else:
            return None

    def is_valid_expression(self, expr):
        if isinstance(expr, sympy.Symbol):
            return True
        elif isinstance(expr, sympy.Add):
            for arg in expr.args:
                if not self.is_valid_expression(arg):
                    return False
            return True
        elif isinstance(expr, sympy.Mul):
            for arg in expr.args:
                if not self.is_valid_expression(arg):
                    return False
            return True
        elif isinstance(expr, sympy.Pow):
            if self.is_valid_expression(expr.args[0]):
                return True
        elif isinstance(expr, (sympy.Float, sympy.Integer)):
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
        if self.llm_pool.factors_max_num < len(self.mul_term.args)-1:
            self.llm_pool.factors_max_num = len(self.mul_term.args)-1

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
    def __init__(self, rs_code, params, llm_pool: LLMPool, dir_name: str):
        sym_converter = SympyConverter(rs_code, params)
        self.left_deriv = prompt_complete_inf[dir_name]['left_deriv'].replace('/', '_')
        self.rs_code = sym_converter.rs_code
        self.sympy_code = sym_converter.sympy_code
        self.llm_pool = llm_pool

    def translate(self):
        cached_pool = self.llm_pool.deepcopy()
        left_side = ' = ' + BaseTokenConverter(self.left_deriv, 1).convert(cached_pool) + '{power: 1.0}'
        add_str = []

        if cached_pool.terms_max_num < len(self.sympy_code.args):
            cached_pool.terms_max_num = len(self.sympy_code.args)

        if self.sympy_code.is_Add:
            for mul_term in self.sympy_code.args:
                converted_term = MulConverter(mul_term, cached_pool).to_epde()
                if converted_term is None:
                    return None
                add_str.append(converted_term)
            self.llm_pool.from_copy(cached_pool)
            return " + ".join(add_str) + left_side

        elif self.sympy_code.is_Mul:
            converted_term = MulConverter(self.sympy_code, cached_pool).to_epde()
            if converted_term is None:
                return None
            else:
                self.llm_pool.from_copy(cached_pool)
                return converted_term + left_side
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
    rs1 = 'params[0] * derivs_dict["du/dx"] ** 3 * t + ((params[1] * derivs_dict["du/dx"] ** 2 * np.cos(params[2] * t +1)) * x**2) * u +params[3] * derivs_dict["du/dt"] * (t**2 + 2)'
    rs2 = 'params[0] * derivs_dict["d^2u/dt^2"] ** 2 * t**3 + (params[1] * derivs_dict["du/dx"] * np.arcsin(params[2] * t**2)) * x**3 * u**5'
    rs3 = 'params[0] * np.arcsin(1.67335*t**2)'

    llm_pool = LLMPool()
    st1 = SolutionTranslator(rs1, np.array([1.2, 5.678, 6.5421, 4.12]), llm_pool, 'sindy-burg').translate()
    st2 = SolutionTranslator(rs2, np.array([2.1, 10.1, 9.2]), llm_pool, 'sindy-burg').translate()
    st3 = SolutionTranslator(rs3, np.array([2.1, ]), llm_pool, 'sindy-burg').translate()
    epde_classes, lambda_strs = llm_pool.to_epde_classes()

    rs5 = 'params[0] * np.sqrt(u)'
    ex1 = sympify('sin(t)').func.__name__
    ex111 = str(sympify('sin(t)').func)
    print()