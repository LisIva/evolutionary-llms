import rasterio
from rasterio.enums import Resampling
import cv2
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator


def resample_img(path: str = "burg64x64.jpg"):
    with rasterio.open(path) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(dataset.count, 32, 32),
            resampling=Resampling.bilinear)
    u = cv2.cvtColor(data.T, cv2.COLOR_BGR2GRAY) / 255
    u_to_string = np.array2string(u, separator=',', max_line_width=1000)
    return u_to_string


def load_resample_array():
    path = os.path.join(os.getcwd(), "data", "simple_burg", "u.npy")
    u = np.load(path)

    x, t = np.linspace(-1000, 0, 101), np.linspace(0, 1, 101)
    xi, ti = np.linspace(-1000, 0, 32), np.linspace(0, 1, 32)
    grids = np.meshgrid(ti, xi, indexing='ij')
    test_points = np.array([grids[0].ravel(), grids[1].ravel()]).T

    interp = RegularGridInterpolator([t, x], u)
    u_res = interp(test_points, method='linear').reshape(32, 32)
    return (np.array2string(np.round(u_res, 2), separator=',', max_line_width=1000),
            np.array2string(np.round(ti, 2), separator=',', max_line_width=1000),
            np.array2string(np.round(xi, 2), separator=',', max_line_width=1000))