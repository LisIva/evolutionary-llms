from titok_pytorch import TiTokTokenizer
import numpy as np
import torch


def set_u_grid():
    np.random.seed(10)
    x = np.linspace(0, 4, 100)
    t = np.linspace(0, 10, 100)
    grids = np.meshgrid(t, x, indexing='ij')
    u_ideal = grids[1] ** 2 + 5 * grids[1] * grids[0] ** 3
    u_ideal = u_ideal / np.max(u_ideal)
    grids[0] = grids[0] / np.max(grids[0])
    grids[1] = grids[1] / np.max(grids[1])
    return grids, u_ideal


grids, u = set_u_grid()
x = grids[1]
t = grids[0]

x_tensor = torch.tensor(x, dtype=torch.float).reshape((1, 1, x.shape[0], x.shape[1]))
t_tensor = torch.tensor(t, dtype=torch.float).reshape((1, 1, t.shape[0], t.shape[1]))
u_tensor = torch.tensor(u, dtype=torch.float).reshape((1, 1, u.shape[0], u.shape[1]))
images = u_tensor

titok = TiTokTokenizer(
    dim = 128,
    patch_size = 5,
    num_latent_tokens = 20,   # they claim only 32 tokens needed
    codebook_size = 4096,      # codebook size 4096
    image_size = u.shape[0],
    channels=1
)

loss = titok(images)
loss.backward()

codes = titok.tokenize(images) # (2, 32)
recon_images = titok.codebook_ids_to_images(codes)
err = recon_images - u_tensor
assert recon_images.shape == images.shape
print()
