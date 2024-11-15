from imagetokenizer.model.magvit2 import Magvit2Tokenizer
import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def show_img(path):
    img = Image.open(path)
    img.show()

# Initialize the tokenizer
image_tokenizer = Magvit2Tokenizer()

# Tokenize an image
path_im = os.path.join(os.getcwd(), "images", 'im1.jpg')
img = Image.open(path_im)
convert_tensor = ToTensor()
img_tensor = convert_tensor(img)
img_tensor = img_tensor.reshape((1, 3, img_tensor.shape[1], img_tensor.shape[2]))

quants, embedding, codebook_indices = image_tokenizer.encode(img_tensor)

# Print the tokens
image = image_tokenizer.decode(quants)
im4show = image.cpu().detach().numpy().reshape((image.shape[2], image.shape[3], image.shape[1]))

fig = plt.figure(figsize=(10, 10))
plt.imshow(im4show)
plt.show()
print()