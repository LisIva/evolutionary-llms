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