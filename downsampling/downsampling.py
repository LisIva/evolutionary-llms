import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
# np.random.seed(10)
from PIL import Image
import numpy as np
import cv2


def plot_3dsurf(u, grids):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(grids[0], grids[1], u, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    plt.show()


def make_grids(u):
    x = np.arange(u.shape[0])
    t = np.arange(u.shape[0])
    return np.meshgrid(t, x, indexing='ij')


def use_rasterio():
    # im_gray = cv2.imread('burg64x64.jpg', cv2.IMREAD_GRAYSCALE)

    # resample data to target shape
    with rasterio.open("img3.jpg") as dataset:
        data = dataset.read(
            out_shape=(dataset.count, 32, 32),
            resampling=Resampling.bilinear)

    u_ideal = cv2.cvtColor(data.T, cv2.COLOR_BGR2GRAY)
    plot_3dsurf(u_ideal, make_grids(u_ideal))


path_u = "D:\\Users\\Ivik12S\\Desktop\\PDE-Net 2.0\\matrices_burgers\\u.npy"
u = np.load(path_u)

u_res = cv2.resize(u, (32, 32), 0, 0, cv2.INTER_LINEAR).astype(np.float32)
plot_3dsurf(u_res, make_grids(u_res))
# plot_3dsurf(u, make_grids(u))
print()