from openai import OpenAI
from langchain_core.prompts import PromptTemplate
import numpy as np
import os
import sys
import rasterio
from rasterio.enums import Resampling
import cv2
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import creds
np.set_printoptions(threshold=sys.maxsize)


client = OpenAI(
    api_key=creds.api_key,
    base_url="https://api.vsegpt.ru/v1",
)

path = os.path.join(Path().absolute().parent, "prompts", "vis-llms", "prompt-vis-qwen3.txt")
file = open(path, 'r')
prompt_raw = file.read()
images_url = {"simple eq": "1_-UVYLsBcvgFf7lcsGN6DNWjZ7vinzbs",
              'cos2.5x': "1Qd4Qavj9plipXhZLqNbNELQno3FU7Qdi",
              'cos10x': "1C-KFEWHmcXj5kmDYyGEimUGTtVm48Wfb",
              'cos10xyb': "1RQEU61gBoojKpncEG5ouL6xMo4qmsmge",
              'flowers': "1FXVZoQVpwfKrYLJw8QpIXmFi0SyKbv7f",
              'cos10xyb512': "1fNjmbeHUWFCMywUNAZ2BLqsWrZb-FjdR",
              'cos10xyb1024': "18LsZvka_oNDhEICcegKOF4k8-b7qMiik"}

# prompt_test = "What are the possible functions (like u(x) = 2 * x**3) that could be represented by the gradient in the image? Describe the gradient with function and words"
image_url = f"https://drive.usercontent.google.com/u/0/uc?id={images_url['flowers']}&export=download"
# prompt_test = "Considering that the white stripes (intensity 255 or (255, 255, 255) in RGB) represent maximum and the black ones minimum (intensity = 0 or (0, 0, 0) in RGB), how many maximums and minimums does the image have?"

# prompt_test = "How many #450053 stripes and how many #FCE923 stripes are there on the picture?"
prompt_test = "How many flowers are on the picture?"
response = client.chat.completions.create(
    model="vis-meta-llama/llama-3.2-90b-vision-instruct", #  vis-google/gemini-pro-vision  vis-qwen/qwen-2-vl-72b-instruct  vis-meta-llama/llama-3.2-90b-vision-instruct
    messages=[
        # {"role": "system", "content": "You are an experienced mathematician, "
        #                               "especially familiar with physical meaning of functions and equations"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_test},
                {
                    "type": "image_url",
                    "image_url": image_url,
                },
            ],
        }
    ],
    max_tokens=1000,
)

print("Answer of the model:\n")
print(response.choices[0].message.content)
