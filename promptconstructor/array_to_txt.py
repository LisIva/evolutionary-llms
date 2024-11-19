import numpy as np
import os
import sys
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
np.set_printoptions(threshold=sys.maxsize)


PARENT_PATH = Path().absolute().parent


def load_resample_array(name="u", shape=(20,20)):
    path = os.path.join(PARENT_PATH, "data", "simple_burg", f"{name}.npy")
    u = np.load(path)

    x, t = np.linspace(-1000, 0, 101), np.linspace(0, 1, 101)
    xi, ti = np.linspace(-1000, 0, shape[1]), np.linspace(0, 1, shape[0])
    grids = np.meshgrid(ti, xi, indexing='ij')
    test_points = np.array([grids[0].ravel(), grids[1].ravel()]).T

    interp = RegularGridInterpolator([t, x], u)
    u_res = interp(test_points, method='linear').reshape(shape[0], shape[1])
    return u_res,ti,xi


def write_file(t, x, u, u_t, u_x, show_symnum=False):
    # data = list(map(np.array2string, [t, x, u, u_t, u_x]))
    grids = np.meshgrid(t, x, indexing='ij')
    with open("burg_txu_derivs.txt", 'w') as myf:
        raveled_vals = list(map(np.ravel, [grids[0], grids[1], u, u_t, u_x]))
        for ti, xi, ui, u_ti, u_xi in zip(*raveled_vals):
            myf.write(f'{ti} {xi} {ui} {u_ti} {u_xi}\n')

    if show_symnum:
        with open("burg_txu_derivs.txt", 'r') as myf:
            content = myf.read()
            print("Количество символов:", len(content))


def local_round(t, x, u, u_t, u_x):
    t1 = np.round(t, 2)
    x1 = np.round(x, 1)
    u1 = np.round(u, 1)
    u_t1 = np.round(u_t, 1)
    u_x1 = np.round(u_x, 3)
    return t1, x1, u1, u_t1, u_x1


def get_simple_burg_data():
    u, t, x = load_resample_array()
    u_t, _, _ = load_resample_array("du_dx0")
    u_x, _, _ = load_resample_array("du_dx1")
    t, x, u, u_t, u_x = local_round(t, x, u, u_t, u_x)
    write_file(t, x, u, u_t, u_x)


if __name__ == "__main__":
    u, t, x = load_resample_array()
    u_t, _, _ = load_resample_array("du_dx0")
    u_x, _, _ = load_resample_array("du_dx1")
    t, x, u, u_t, u_x = local_round(t, x, u, u_t, u_x)
    write_file(t, x, u, u_t, u_x, show_symnum=True)
    print()

