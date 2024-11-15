import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from d1_tokenizer.modeling.maskgit import ImageBert
from d1_tokenizer.modeling.titok import TiTok


def show_img(path):
    img = Image.open(path)
    img.show()


path_im = os.path.join(os.getcwd(), "images", 'im1.jpg')
img = Image.open(path_im)
convert_tensor = ToTensor()
img_tensor = convert_tensor(img)
img_tensor = img_tensor.reshape((3, img_tensor.shape[1], img_tensor.shape[2]))


image = torch.from_numpy(np.array(Image.open(path_im)).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0


titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
titok_tokenizer.eval()
titok_tokenizer.requires_grad_(False)

device = "cuda"
titok_tokenizer = titok_tokenizer.to(device)

# tokenization
encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]

# image assets/ILSVRC2012_val_00010240.png is encoded into tokens tensor([[[ 887, 3979,  349,  720, 2809, 2743, 2101,  603, 2205, 1508, 1891, 4015, 1317, 2956, 3774, 2296,  484, 2612, 3472, 2330, 3140, 3113, 1056, 3779,  654, 2360, 1901, 2908, 2169,  953, 1326, 2598]]], device='cuda:0'), with shape torch.Size([1, 1, 32])
print(f"image {path_im} is encoded into tokens {encoded_tokens}, with shape {encoded_tokens.shape}")

# de-tokenization
reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
# reconstructed_image = Image.fromarray(reconstructed_image).save("assets/ILSVRC2012_val_00010240_recon.png")

