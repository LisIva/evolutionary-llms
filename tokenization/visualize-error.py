from titok_pytorch import TiTokTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt


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

u_tensor = torch.tensor(u_rgb, dtype=torch.float).reshape((1, 3, u_rgb.shape[0], u_rgb.shape[1]))
images = u_tensor

titok = TiTokTokenizer(
    dim = 1024,
    patch_size = 10,
    num_latent_tokens = 300,   # they claim only 32 tokens needed
    codebook_size = 4096,      # codebook size 4096
    image_size = u.shape[0],
    channels=3
)

loss = titok(images)
loss.backward()

codes = titok.tokenize(images) # (2, 32)
recon_images = titok.codebook_ids_to_images(codes)
err = recon_images - u_tensor
recon_u = recon_images.cpu().detach().numpy().reshape((u.shape[0], u.shape[1], 3))
draw_img(recon_u)
