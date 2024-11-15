from tokenlearner_pytorch import TokenLearner
import numpy as np
import torch
import matplotlib.pyplot as plt


# u(t, x) = x^2 + 5xt^3
def set_u_grid():
    np.random.seed(10)
    x = np.linspace(0, 4, 100)
    t = np.linspace(0, 10, 100)
    grids = np.meshgrid(t, x, indexing='ij')
    u_ideal = grids[1] ** 2 + 100*np.cos(grids[0])
    return grids, u_ideal


def make_rgb(C: np.ndarray):
    ax_im = plt.imshow(C, cmap='grey')
    colours = ax_im.cmap(ax_im.norm(C))
    image = colours[:, :, :3]
    plt.clf()
    return image


def plot_function(C, grids):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    # ax.view_init(20, 35)
    ax.plot_surface(grids[0], grids[1], C, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    plt.show()


def draw_img(img):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()


grids, u = set_u_grid()
u_rgb = make_rgb(u)
u_tensor = torch.tensor(u_rgb, dtype=torch.float).reshape((1, u.shape[0], u.shape[1], 3))

tklr = TokenLearner(S=100)
u_tl = tklr(u_tensor)


print()