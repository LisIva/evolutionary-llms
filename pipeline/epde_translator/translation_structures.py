from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens, CacheStoredTokens
import numpy as np
import sympy as sp


class LLMPool(object):
    def __init__(self):
        self.simple_tokens_pow = {'t': (0, 0), 'x': (0, 0) }
        self.max_deriv_orders = {'max_deriv_t': 1, 'max_deriv_x': 1}
        self.special_tokens_pow = {}  # key: как епде будет выводить токен юзеру, val - лямбда-функция как его посчитать
        self.epde_classes = []

    def to_epde_classes(self):
        for key in self.simple_tokens_pow.keys():
            if self.simple_tokens_pow[key][0] != 0:
                ct_converter = CustomTokenConverter(str(key), self.simple_tokens_pow[key])
                self.epde_classes.append(ct_converter.get_cache_token())

        for key in self.special_tokens_pow.keys():
            ct_converter = CustomTokenConverter(key, self.special_tokens_pow[key])
            self.epde_classes.append(ct_converter.get_custom_token())
        return self.epde_classes

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

    # def get_cache_token(self, grids):
    #     data = grids[0] if self.token == 't' else grids[1]
    #     return CacheStoredTokens(token_type=self.token, token_labels=[self.token],
    #                              token_tensors={self.token: data},
    #                              params_ranges={'power': (1, self.power)}, params_equality_ranges=None)

    def get_custom_token(self):
        name = str(self.token)
        eval_fun = {name: self.__get_lambda_fun()}
        evaluator = CustomEvaluator(eval_fun, eval_fun_params_labels=['power'])
        params_ranges = {'power': (self.power[0], self.power[1])}
        return CustomTokens(token_type=name, token_labels=[name],
                            evaluator=evaluator, params_ranges=params_ranges,
                            params_equality_ranges={}, meaningful=True, unique_token_type=False)

    def __get_lambda_fun(self):
        lambda_signature = self.__transform_for_eval()
        lambda_str = f"lambda *grids, **kwargs: (np.{lambda_signature}) ** kwargs['power']"
        return eval(lambda_str, {'np': np, })

    def __transform_for_eval(self):
        replacements = {sp.Symbol('t'): sp.Symbol('grids[0]'), sp.Symbol('x'): sp.Symbol('grids[1]')}
        replaced_inner_expr = self.__replace_symbols(self.token.args[0], replacements)
        return str(self.token.func(replaced_inner_expr))

    def __replace_symbols(self, expr, replacements):
        if isinstance(expr, sp.Symbol):
            return replacements.get(expr, expr)
        if isinstance(expr, sp.Add):
            return sp.Add(*[self.__replace_symbols(arg, replacements) for arg in expr.args])
        if isinstance(expr, sp.Mul):
            return sp.Mul(*[self.__replace_symbols(arg, replacements) for arg in expr.args])
        if isinstance(expr, (sp.Float, sp.Integer)):
            return expr
        return expr
        # custom_eval_fun = {
        #     'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
        # custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
        #                                         eval_fun_params_labels=['power'])
        # trig_params_ranges = {'power': (1, 1)}
        # trig_params_equal_ranges = {}
        #
        # custom_trig_tokens = CustomTokens(token_type='trigonometric',
        #                                   token_labels=['cos(t)sin(x)'],
        #                                   evaluator=custom_trig_evaluator,
        #                                   params_ranges=trig_params_ranges,
        #                                   params_equality_ranges=trig_params_equal_ranges,
        #                                   meaningful=True, unique_token_type=False)


if __name__ == '__main__':

    allowed_namespace = {'np': np, }
    # lambda_str = f"lambda t: np.{term_str}"
    # np_fun = eval(lambda_str, allowed_namespace)