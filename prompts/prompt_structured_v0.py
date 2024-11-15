# Generate the code for the function equation_v1. The purpose of equation_v1 is to describe the field u that will be provided below.
# Your task is to find the partial differential equation skeleton to which the function u(t, x) is a solution.
#
# ### Example: ###
#
# Input: u = np.array([[0., 4.], [24., 28.]]), x = np.array([0., 2.]), t = np.array([0., 2.])
# Output: d^2u/dt^2 = -2 * d^2u/dtdx + 3 * d^2u/dx^2
# Explanation: the data on u seems to be from the equation u(t, x) = 3t^3 + x^2.
# For which the governing partial differential equation is d^2u/dt^2 = -2 * d^2u/dtdx + 3 * d^2u/dx^2
#
# ### end of example ###
#
#
# Step-by-step guide for solving the task:
# 1. Analyze the data u, x and t.
# 2. Look at the set of available derivatives set_of_derivs for the differential equation construction.
# 3. Choose one derivative from the set as the balancing term (the term on the left side of the differential equation - u_left_derivative, will be returned by the function equation_v1).
# 4. Look at the experience buffer exp_buffer. If the buffer is not empty your task is to guess an equation that will have a higher score. The buffer may provide some hints about the form of the differential equation.
# 5. Construct the right side of the differential equation by guessing based on input data u, x, t, exp_buffer and set of available derivatives set_of_derivs
#
#
# Requirements:
# -The new structure of equation_v1 must be unique inside the experience buffer.
# -For constructing equation_v1 you can use any function of t, x, u or any available derivative of u except for the balancing term.
# -Equation_v1 must be found with the structure that maximizes the evaluate function.
# -Make sure to only output the function equation_v1.
# -Make sure the function equation_v1 uses at least one derivative as its return value (namely, equation_v1(...)[0] uses some derivatives).
#
#
# Additional information:
# -The experience buffer stores the optimization track with the best equations and their corresponding scores. The pairs equation-score are given in the form of a dictionary. The equations have their params already optimized.
# -The information on the allowed derivatives is given in set_of_derivs. Pay attention that set_of_derivs is a set of keys of derivs_dict.


import numpy as np
from scipy.optimize import minimize


def loss_function(params, t, x, u, derivs_dict, left_deriv_name):
    u_pred = equation_v1(t, x, u, derivs_dict, params)[0]
    return np.mean((u_pred-derivs_dict[left_deriv_name])**2)


def evaluate(data: dict) -> float:
    """ Evaluate the constructed equation"""
    # Load true data observations
    inputs, derivs_dict, left_deriv_name = data['inputs'], data["derivs_dict"], data['left_deriv_name']

    # Optimize equation skeleton parameters
    loss_partial = lambda params: loss_function(params, *inputs, derivs_dict, left_deriv_name)
    params_initial_guess = np.array([1.0]*P)
    result = minimize(loss_partial, params_initial_guess, method='BFGS')
    optimized_params = result.x

    # Return evaluation score
    score = loss_function(optimized_params, *inputs, derivs_dict, left_deriv_name)
    return -score if not np.isnan(score) and not np.isinf(score) else None


### Input data ###

exp_buffer = {{}} # curently empty
t = np.array({t_array})
x = np.array({x_array})
u_true = np.array(
{u_array})
set_of_derivs = {{"du/dx", "du/dt", "d^2u/dt^2"}}

### end of input data ###

"""An example of output:"""
def equation_v0(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):
    """Find mathematical function skeleton that describes u_true data the most.
    Args:
        x0: Time variable.
        x1: Space variable.
        u_true: Values of the input function
        params: Array of numeric parameters to be optimized.
    Return:
        A tuple of: numpy array representing the derivative of u_true (choose one); a string of chosen derivative.
    """
    return (params[0] * u * t + params[1] * x * derivs_dict["du/dx"] * derivs_dict["d^2u/dt^2"], "du/dt")

"""The function to complete:"""
def equation_v1(t: np.ndarray, x: np.ndarray, u: np.ndarray, derivs_dict: dict(), params: np.ndarray):
    """Your task is to generate code here with an example of the function given at equation_v0"""