

from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageOps
import tensorflow_hub as hub

# Stuff that was there:
import os
# import detect <--- what is this
import tflite_runtime.interpreter as tflite
import platform
import datetime
import cv2
import time
import numpy as np
import io
from io import BytesIO
import random
import re
import tensorflow as tf


import matplotlib.pyplot as plt

from app import detection_loop

data_input = os.path.join('data', 'test_small')  # file or directory
output = 'out/'  # request.values.get('output')

path = data_input
filename_images = {}

input_format = ["jpg", "png", "jpeg"]
if data_input.find(".") != -1:  # if its a file
    print(data_input + " is a file")
    split_data_input = data_input.split(".", 1)
    if data_input.endswith(tuple(input_format)):
        print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
        # path_splitted = []    <--- pointless, should we delete?
        path_splitted = re.split('/', data_input)
        filename = path_splitted[len(path_splitted) - 1]
        # filename_images[filename] = Image.open(data_input)
        filename_images[filename] = None
        path = os.path.dirname(data_input) + "/"
else:  # if its a directory
    print(data_input + " is a path with the following files: ")
    for filename in os.listdir(data_input):
        image_path = os.path.join(data_input, filename)
        filename_images[filename] = None
        print("  " + filename)

detection_loop(filename_images, path, output)
