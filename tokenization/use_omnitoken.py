from tokenization.OmniTokenizer.OmniTokenizer import OmniTokenizer_VQGAN
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt


def open_img(name):
    path_im = os.path.join(os.getcwd(), "images", name)
    img = Image.open(path_im)
    convert_tensor = ToTensor()
    img_tensor = convert_tensor(img)
    return img_tensor.reshape((1, 3, img_tensor.shape[1], img_tensor.shape[2]))


def draw_tensor(image):
    im4show = image.cpu().detach().numpy().reshape((image.shape[2], image.shape[3], image.shape[1]))

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(im4show)
    plt.show()


def draw_layer(tokens, id):
    layer = tokens[0, id, :, :]
    data = layer.cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    grids = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    # ax.view_init(20, 35)
    ax.plot_surface(grids[0], grids[1], data, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('z')
    plt.show()


# img = open_img("burg64x64.jpg")
img = open_img("doggy128x128.jpg")
# mean = torch.mean(img)
vqgan_ckpt = os.path.join(os.getcwd(), "imagenet_ucf_vae.ckpt")
vqgan = OmniTokenizer_VQGAN.load_from_checkpoint(vqgan_ckpt, strict=False)

tokens = vqgan.encode(img, is_image=True)

# draw_layer(tokens, 7)
recons = vqgan.decode(tokens, is_image=True)
if torch.min(recons) < 0:
    recons = recons + torch.abs(torch.min(recons))
draw_tensor(recons)
print("all done")