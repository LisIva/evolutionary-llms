import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import cv2
import os


def F(u, v):
    return u * np.cos(u * v) + v


def plot_3dsurf(u, grids, title):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(grids[0], grids[1], u, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    ax.set_title(title)


def load_u():
    path = os.path.join(os.getcwd(), "data")
    u = np.load(os.path.join(path, "u.npy"))
    return u


def load_resample_array():
    path = os.path.join(os.getcwd(), "data", "u.npy")
    u = np.load(path)
    # u_res = cv2.resize(u, (32, 32), 0, 0, cv2.INTER_LINEAR).astype(np.float32)

    x = np.linspace(-1000, 0, 101)
    t = np.linspace(0, 1, 101)
    xi, ti = np.meshgrid(np.linspace(-1000, 0, 32), np.linspace(0, 1, 32), indexing='ij')
    test_points = np.array([ti.ravel(), xi.ravel()]).T

    interp = RegularGridInterpolator([t, x], u)
    u_res = interp(test_points, method='linear').reshape(32, 32)
    return (np.array2string(np.round(u_res, 2), separator=',', max_line_width=1000),
            np.array2string(np.round(ti, 2), separator=',', max_line_width=1000),
            np.array2string(np.round(xi, 2), separator=',', max_line_width=1000))


u = np.linspace(-1000, 0, 101)
v = np.linspace(0, 1, 101)

load_resample_array()
fit_points = [u, v]
# values = F(*np.meshgrid(*fit_points, indexing='ij'))

values = load_u()

ut, vt = np.meshgrid(np.linspace(-1000, 0, 32), np.linspace(0, 1, 32), indexing='ij')
# true_values = F(ut, vt)
test_points = np.array([ut.ravel(), vt.ravel()]).T

interp = RegularGridInterpolator(fit_points, values)
im = interp(test_points, method='linear').reshape(32, 32)
plot_3dsurf(im, [ut, vt], "some")
# for method in ['linear', 'nearest', 'slinear', 'cubic', 'quintic', 'custom']:
#     if method=="custom":
#         im = cv2.resize(values, (32, 32), 0, 0, cv2.INTER_LINEAR).astype(np.float32)
#     else:
#         im = interp(test_points, method=method).reshape(32, 32)
#
#     error = np.mean(np.abs(true_values - im))
#     plot_3dsurf(im, [ut, vt], title=f"{method}| {error:.2f}")

# plot_3dsurf(true_values, [ut, vt], title="true vals")
plt.show()