import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)
from PIL import Image
import os


# u(t, x) = x^2 + 5xt^3
def set_u_grid(size=64):
    x = np.linspace(0, 4, size)
    t = np.linspace(0, 10, size)
    grids = np.meshgrid(t, x, indexing='ij')
    # u_ideal = grids[1] ** 2 + 100*np.cos(grids[0]) #* grids[1] + 100*np.cos(grids[1])
    # u_ideal = grids[0] + 100 * np.sin(grids[1])
    # u_ideal = 100 * grids[0]**2 + 1 * grids[1]
    u_ideal = np.cos(10*grids[1])
    return grids, u_ideal


def normalize(u, grids):
    u_ideal = u / np.max(u)
    grids[0] = grids[0] / np.max(grids[0])
    grids[1] = grids[1] / np.max(grids[1])
    return u_ideal, grids


def plot_3dsurf(u, grids):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(grids[0], grids[1], u, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    plt.show()


def plot_heatmap(u_ideal, save=False, name="my_img"):
    fig = plt.figure(figsize=(10, 10))
    # ax_im = plt.imshow(u_ideal, cmap='grey')
    ax_im = plt.imshow(u_ideal)
    colours = ax_im.cmap(ax_im.norm(u_ideal))
    image_arr = (colours[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(image_arr, mode="RGB")
    if save:
        img.save(name)
    plt.clf()
    plt.imshow(img)
    plt.show()


# path_u = "D:\\Users\\Ivik12S\\Desktop\\PDE-Net 2.0\\matrices_burgers"
# path_u_full = os.path.join(path_u, 'u.npy')
# u_ideal = np.load(path_u_full)
# u_ideal = u_ideal[:64, :64]

# x = np.arange(u_ideal.shape[0])
# t = np.arange(u_ideal.shape[0])
# grids = np.meshgrid(t, x, indexing='ij')

grids, u_ideal = set_u_grid(1024)
# u_ideal, grids = normalize(u_ideal, grids)
plot_heatmap(u_ideal, True, 'cos10x1024x1024yb.jpg')
# plot_heatmap(u_ideal)
