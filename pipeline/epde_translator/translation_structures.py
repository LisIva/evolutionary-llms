from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens
import numpy as np
import sympy as sp
import copy


class LLMPool(object):
    def __init__(self):
        self.simple_tokens_pow = {'t': (0, 0), 'x': (0, 0)}
        self.max_deriv_orders = {'max_deriv_t': 1, 'max_deriv_x': 1}
        self.max_deriv_pow = {'data_fun_pow': 1, 'deriv_fun_pow': 1}

        self.special_tokens_pow = {}  # key: как епде будет выводить токен юзеру, val - лямбда-функция как его посчитать
        self.factors_max_num = 1
        self.terms_max_num = 1

    def from_copy(self, cached_pool):
        self.simple_tokens_pow = cached_pool.simple_tokens_pow
        self.max_deriv_orders = cached_pool.max_deriv_orders
        self.special_tokens_pow = cached_pool.special_tokens_pow
        self.factors_max_num = cached_pool.factors_max_num
        self.terms_max_num = cached_pool.terms_max_num
        self.max_deriv_pow = cached_pool.max_deriv_pow

    def deepcopy(self):
        llmpool = LLMPool()
        llmpool.simple_tokens_pow = self.simple_tokens_pow.copy()
        llmpool.max_deriv_orders = self.max_deriv_orders.copy()
        llmpool.special_tokens_pow = copy.deepcopy(self.special_tokens_pow)
        llmpool.factors_max_num = self.factors_max_num
        llmpool.terms_max_num = self.terms_max_num
        llmpool.max_deriv_pow = self.max_deriv_pow.copy()
        return llmpool

    def to_epde_classes(self):
        epde_classes, lambda_strs = [], []
        for key in self.simple_tokens_pow.keys():
            if self.simple_tokens_pow[key][0] != 0:
                ct_converter = CustomTokenConverter(str(key), self.simple_tokens_pow[key])
                epde_classes.append(ct_converter.get_cache_token())

        for key in self.special_tokens_pow.keys():
            ct_converter = CustomTokenConverter(key, self.special_tokens_pow[key])
            e_class, l_str = ct_converter.get_custom_token()
            epde_classes.append(e_class)
            lambda_strs.append(l_str)
        return epde_classes, lambda_strs

    def set_token_pow(self, token, min_pow, max_pow):
        self.simple_tokens_pow[token] = (min_pow, max_pow)

    def set_max_d_order(self, token, max_order):
        self.max_deriv_orders[token] = max_order

    def set_special_token_pow(self, token, min_pow, max_pow):
        self.special_tokens_pow[token] = (min_pow, max_pow)


class CustomTokenConverter(object):
    def __init__(self, token, power):
        self.token = token
        self.power = power

    def get_cache_token(self):
        lambda_signature = 'grids[0]' if self.token == 't' else 'grids[1]'
        lambda_str = f"lambda *grids, **kwargs: ({lambda_signature}) ** kwargs['power']"
        eval_fun = {self.token: lambda_str}
        evaluator = CustomEvaluator(eval_fun, eval_fun_params_labels=['power'])
        params_ranges = {'power': (self.power[0], self.power[1])}
        return CustomTokens(token_type=self.token, token_labels=[self.token],
                            evaluator=evaluator, params_ranges=params_ranges,
                            params_equality_ranges={}, meaningful=True, unique_token_type=False)

    def get_custom_token(self):
        name = str(self.token)
        lambda_fun, lambda_str = self.__get_lambda_fun()
        eval_fun = {name: lambda_fun}
        evaluator = CustomEvaluator(eval_fun, eval_fun_params_labels=['power'])
        params_ranges = {'power': (self.power[0], self.power[1])}
        return CustomTokens(token_type=name, token_labels=[name],
                            evaluator=evaluator, params_ranges=params_ranges,
                            params_equality_ranges={}, meaningful=True, unique_token_type=False), lambda_str

    def __get_lambda_fun(self):
        lambda_signature = self.__transform_for_eval()
        lambda_str = f"lambda *grids, **kwargs: (np.{lambda_signature}) ** kwargs['power']"
        return eval(lambda_str, {'np': np, }), lambda_str

    def __transform_for_eval(self):
        replacements = {sp.Symbol('t'): sp.Symbol('grids[0]'), sp.Symbol('x'): sp.Symbol('grids[1]')}
        replaced_inner_expr = self.__replace_symbols(self.token.args[0], replacements)
        return str(self.token.func(replaced_inner_expr))

    def __replace_symbols(self, expr, replacements):
        if isinstance(expr, sp.Symbol):
            return replacements.get(expr, expr)
        if isinstance(expr, sp.Pow):
            return sp.Pow(self.__replace_symbols(expr.args[0], replacements), expr.args[1])
        if isinstance(expr, sp.Add):
            return sp.Add(*[self.__replace_symbols(arg, replacements) for arg in expr.args])
        if isinstance(expr, sp.Mul):
            return sp.Mul(*[self.__replace_symbols(arg, replacements) for arg in expr.args])
        if isinstance(expr, (sp.Float, sp.Integer)):
            return expr
        return expr


if __name__ == '__main__':

    allowed_namespace = {'np': np, }
    # lambda_str = f"lambda t: np.{term_str}"
    # np_fun = eval(lambda_str, allowed_namespace)
