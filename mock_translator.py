#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:50:10 2022

@author: maslyaev
"""

import numpy as np
import sys
import getopt
from collections import OrderedDict

import epde.globals as global_var

from epde.interface.prepared_tokens import TrigonometricTokens
from epde.evaluators import simple_function_evaluator, trigonometric_evaluator
from epde.interface.token_family import TokenFamily, TFPool
from epde.cache.cache import prepare_var_tensor, upload_simple_tokens, upload_grids
from epde.supplementary import define_derivatives
from epde.preprocessing.derivatives import preprocess_derivatives
from epde.interface.prepared_tokens import CacheStoredTokens
from epde.interface.equation_translator import translate_equation

def get_basic_var_family(var_name, deriv_names, deriv_orders):
    entry_token_family = TokenFamily(var_name, family_of_derivs = True)
    entry_token_family.set_status(demands_equation=True, unique_specific_token=False, 
                                  unique_token_type=False, s_and_d_merged = False, 
                                  meaningful = True)     
    entry_token_family.set_params(deriv_names, OrderedDict([('power', (1, 1))]),
                                  {'power' : 0}, deriv_orders)
    entry_token_family.set_evaluator(simple_function_evaluator, [])    
    return entry_token_family

def prepare_basic_inputs():
    grids = [np.linspace(0, 4*np.pi, 1000), np.linspace(0, 10, 1000)]
    var_name = 'u'
    u = np.sin(grids[0]) + 1.3 * np.cos(grids[0])#np.load('/home/maslyaev/epde/EPDE_main/tests/system/Test_data/fill366.npy')

    global_var.init_caches(set_grids = True)
    global_var.set_time_axis(0)
    global_var.grid_cache.memory_usage_properties(u, 3, None)
    global_var.tensor_cache.memory_usage_properties(u, 3, None)

    deriv_names, deriv_orders = define_derivatives(var_name, dimensionality=u.ndim, max_order = 1)

    method = 'poly'; method_kwargs = {'grid' : grids, 'smooth' : False}
    data_tensor, derivatives = preprocess_derivatives(u, method=method, method_kwargs=method_kwargs)
    derivs_stacked = prepare_var_tensor(u, derivatives, time_axis = global_var.time_axis)

    upload_grids(grids, global_var.grid_cache)
    upload_simple_tokens(deriv_names, global_var.tensor_cache, derivs_stacked)
    global_var.tensor_cache.use_structural()

    var_family = get_basic_var_family(var_name, deriv_names, deriv_orders)
    trig_tokens = TrigonometricTokens(dimensionality = 0, freq = (0.95, 1.05))
    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                             token_labels=['t', 'x'],
                                             token_tensors={'t': grids[0], 'x': grids[1]},
                                             params_ranges={'power': (1, 1)},
                                             params_equality_ranges=None)
    # trig_tokens.token_family
    pool = TFPool([var_family, custom_grid_tokens.token_family])

    return grids, pool


def mock_equation(text_form = ('(1.0 * u{power: 1} * sin{freq: 1, power: 1, dim: 0} + 1.) + 5.0 * u{power: 1} = '
                               'du/dx1{power: 1} * cos{freq: 1, power: 1, dim: 0}')):
    grids, mock_pool = prepare_basic_inputs()
    print('Mock pool families:', [family.tokens for family in mock_pool.families])
    
    return grids, translate_equation(text_form, mock_pool)


# custom_trigonometric_eval_fun =  {'cos' : lambda *grids, **kwargs:
#                                     np.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'],
#                                   'sin' : lambda *grids, **kwargs:
#                                     np.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}
# custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels = ['freq', 'dim', 'power'])
# trig_params_ranges = {'power' : (1, 1), 'freq' : (0.95, 1.05), 'dim' : (0, dimensionality)}
# trig_params_equal_ranges = {'freq' : 0.05}
#
# custom_trig_tokens = Custom_tokens(token_type = 'trigonometric',
#                                    token_labels = ['sin', 'cos'],
#                                        evaluator = custom_trig_evaluator,
#                                        params_ranges = trig_params_ranges,
#                                        params_equality_ranges = trig_params_equal_ranges,
#                                        meaningful = True, unique_token_type = False)


if __name__ == '__main__':
    text = '1.0 * u{power: 1} + 2.7744 * du/dx1{power: 1} * t{power: 1} = du/dx1{power: 1}'
    _, eq = mock_equation(text)
    print()
