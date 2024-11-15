import numpy as np
from scipy.optimize import minimize

# u(t, x) = x^2 + 5xt^3
def set_u_grid():
    np.random.seed(10)
    x = np.linspace(0, 4, 10)
    t = np.linspace(0, 10, 10)
    grids = np.meshgrid(t, x, indexing='ij')
    # np.savetxt("t_vals.txt", grids[0], delimiter=", ", newline='],\n [')
    # np.savetxt("x_vals.txt", grids[1], delimiter=", ", newline='],\n [')
    # np.savetxt("t_vals_small.txt", t, delimiter=", ", newline=', ')
    # np.savetxt("x_vals_small.txt", x, delimiter=", ", newline=', ')
    u_ideal = grids[1] ** 2 + 5 * grids[1] * grids[0] ** 3
    # np.savetxt("u_vals.txt", u_ideal, delimiter=", ", newline='],\n [')
    noise_lvl = 0.001
    noise_val = np.random.rand(u_ideal.shape[0], u_ideal.shape[1]) * noise_lvl
    u_noised = u_ideal + noise_val * np.fabs(u_ideal)
    return grids, u_noised, u_ideal


# def equation(x0: np.ndarray, x1: np.ndarray, params: np.ndarray) -> np.ndarray:
#     """Find mathematical function skeleton that describes u_true data the most.
#     Args:
#         x0: Time variable.
#         x1: Space variable.
#         params: Array of numeric parameters to be optimized.
#     Return:
#         A numpy array representing the field u_true.
#     """
#     return params[0] * x1 * x1 + params[1] * x1 * x0 * x0 * x0

def equation_v1(x0: np.ndarray, x1: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Your task is to generate code here with an example of the function given at equation_v0"""
    return (params[0] * x0 ** 4 + params[1] * x1 ** 4 + \
            params[2] * x0 ** 3 * x1 + params[3] * x0 * x1 ** 3 + \
            params[4] * x0 ** 3 + params[5] * x1 ** 3 + \
            params[6] * x0 ** 2 * x1 ** 2 + \
            params[7] * x0 ** 2 * x1 + params[8] * x0 * x1 ** 2 + \
            params[9] * x0 ** 2 + params[10] * x1 ** 2 + \
            params[11] * x0 + params[12] * x1 + \
            params[13])
    # return (params[0] * x1 ** 2 + params[1] * x1*x0 **3) #-8.627359884802176e-11



    # -1.6404197428419203e-06, -9.97516470972674e-07, -1.3744880080020078e-07
    # return params[0] * x0**2 + params[1] * x0 * x1 + params[2] * x1**2 + params[3] * x0 + params[4] * x1 + params[5]
    # return params[0] * x0**2 + params[1] * x0 * x1 + params[2] * x1 ** 2 + params[3] * x0 + params[4] * x1 + params[5]
    # return params[0] * x0 ** 3 + params[1] * x1 ** 3 + \
    #  params[2] * x0 ** 2 * x1 + params[3] * x0 * x1 ** 2 + \
    #  params[4] * x0 ** 2 + params[5] * x1 ** 2 + \
    #  params[6] * x0 + params[7] * x1 + \
    #  params[8]
    # return params[0] + params[1] * x0 + params[2] * x1 + params[3] * x0**2 + params[4] * x1**2 + \
    #        params[5] * x0 * x1 + params[6] * x0**3 + params[7] * x1**3 + params[8] * x0**2 * x1 + \
    #        params[9] * x0 * x1**2
    # return (params[0] * x0 ** 3 +
    #         params[1] * x1 ** 3 +
    #         params[2] * x0 ** 2 * x1 +
    #         params[3] * x0 * x1 ** 2 +
    #         params[4] * x0 ** 2 +
    #         params[5] * x1 ** 2 +
    #         params[6] * x0 +
    #         params[7] * x1 +
    #         params[8] * x0 * x1 * (x0 + x1) +
    #         params[9])

def loss_function(params, x0, x1, u_true):
    u_pred = equation_v1(x0, x1, params)
    return np.mean((u_pred-u_true)**2)


def evaluate(data: dict) -> float:
    """ Evaluate equation of input and output observations."""
    # Load true data observations
    inputs, outputs = data['inputs'], data['outputs']

    # Optimize equation skeleton parameters
    loss_partial = lambda params: loss_function(params, *inputs, outputs)
    params_initial_guess = np.array([1.0]*P)
    result = minimize(loss_partial, params_initial_guess, method='BFGS')
    optimized_params = result.x
    # [-1.03225598e-05 - 1.74914533e-04  5.00000919e+00 - 1.78342273e-05, 1.71279161e-04
    #  1.46158417e-03 - 8.03713138e-06 - 1.10510283e-04, 2.01764753e-04 - 7.62684762e-04  9.95918565e-01  7.38200921e-04,
    #  4.04775665e-03 - 8.72566194e-04]
    # Return evaluation score
    score = loss_function(optimized_params, *inputs, outputs)
    return -score if not np.isnan(score) and not np.isinf(score) else None


if __name__ == '__main__':

    P = 14

    grids, u_noised, u_ideal = set_u_grid()
    data = {"inputs": grids, "outputs": u_ideal}
    score = evaluate(data)
    print(score)
# exp_buffer = {"params[0] * x1": 500,
#               "params[0] * x0 ** 3 + params[1] * x1 ** 3 + \
#                 params[2] * x0 ** 2 * x1 + params[3] * x0 * x1 ** 2 + \
#                 params[4] * x0 ** 2 + params[5] * x1 ** 2 + \
#                 params[6] * x0 + params[7] * x1 + \
#                 params[8]": -19,}

