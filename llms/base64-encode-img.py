import base64
from openai import OpenAI
import os
import numpy as np


def format_arrays(array):
    to_string = np.array2string(array, separator=',', max_line_width=1000)
    return to_string


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
# image_path = "data\\simple_burg\\trigon.jpg"
# image_path = os.path.join(os.getcwd(), image_path)

# Getting the base64 string
# base64_image = encode_image(image_path)
# length = len(base64_image)
# length1 = len("https://drive.google.com/file/d/1L0vjBCqgtbTrQu8Rfl9ZXQWXXK31DRDV/view?usp=drive_link")
# https://drive.google.com/file/d/1L0vjBCqgtbTrQu8Rfl9ZXQWXXK31DRDV/view?usp=drive_link


t = np.arange(10000, dtype=np.float64).reshape((100, 100))
t_str = len(format_arrays(t))
s = len(str(base64.b64encode(t)))
s1 = len(str(base64.b64encode(format_arrays(t))))
print()
