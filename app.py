# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Stuff i have added
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageOps
import tensorflow_hub as hub

from werkzeug.local import Local

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
from flask import Flask, request, Response, jsonify, g
import json
import random
import re
import tensorflow as tf
from utils import draw_boxes
from icecream import ic


import matplotlib.pyplot as plt


def get_module_handle(module_name: str) -> str:
    '''
    :param module_name: String containing one of the valid module names.
    :return: handle to the model object loaded from TFHub
    '''

    if module_name == 'FasterRCNN+InceptionResNetV2':
        handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    elif module_name == 'ssd+mobilenetV2':
        handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    else:
        raise RuntimeError('Unknown module name.')

    return handle


app = Flask(__name__)


def get_model():
    """
        Load model from application context
    """
    if 'model' not in g:
        ic("Loading Model!")
        detector = hub.load(get_module_handle('ssd+mobilenetV2')
                            ).signatures['serving_default']
        g.model = detector
        ic(g.model)
    else:
        ic("Model loaded!")
        detector = g.model
    return detector


def detection_loop(filename_images: dict, input_path: str, output_path='out', module_name='ssd+mobilenetV2', output_type=None, verbose=False):
    '''
    :param filename_images: dict in the format {str:PIL image} containing all the images in the input directory.
    :param input_path: directory containing the input data (str)
    :param output_path: directory in which to store output data (str)
    :param module_name: name of the module to load from TFHub (str)
    :param output_type: toggles between the types of output (str). Valid values are 'graphic' | 'text'
    :param verbose: toggles intermediate console output (bool)
    :return mean_inference_time: mean inference time of the model over the input images (np.float64)
    '''

    # Getting the tfhub link corresponding to the module name provided as input
    # module_handle = get_module_handle(module_name)

    # Loading actual model object from the handle
    detector = get_model()

    # Initializing the inference times storage
    inference_times = []

    # Iterates over every item in the dictionary provided as input. Doesn't really use the values.
    for filename, image_handle in filename_images.items():

        # Reading RGB jpeg into a NxMx3 tensor
        img = tf.io.read_file(os.path.join(input_path, filename))
        img = tf.image.decode_jpeg(img, channels=3)

        # Adding a new axis to the tensor, as the model expects it to be of shape [1,length, width, 3]
        converted_img = tf.image.convert_image_dtype(img, tf.uint8)[
            tf.newaxis, ...]

        # Running inference and measuring the time right before and right after
        # Results of object detection are stored in a dictionary with several tensors corresponding to
        # e.g. the bounding box locations, the object classes and their respective classification scores
        start_time = time.time()
        result = detector(converted_img)
        end_time = time.time()

        # Computing inference time from our measurements
        inference_times.append(end_time-start_time)
        print(end_time-start_time)

        # If desired, writes a txt with the bounding box information to the output directory.
        if output_type == 'text':
            # We then convert the tf tensors in the results dictionary into np arrays so that we can produce
            # the desired outputs.
            # When converting, we must remove the extra dimension we added to the tensors before running the inference,
            # as the graphical output code expects a shape of [length, width, 3], and there is no point to keeping
            # that extra dimension in the text output.
            result_np = {key: value.numpy().reshape(
                list(value.shape[1:])) for key, value in result.items()}

            # Save boxes to file
            out_filename = filename.split('.')[0]+'.txt'
            with open(os.path.join(output_path, out_filename), 'w') as fp:
                fp.write(str(result_np["detection_boxes"])+'\n' +
                         str(result_np["detection_classes"])+'\n'+str(result_np["detection_scores"]))

    # Returns the mean inference time over the provided images
    # mean_inference_time = np.mean(inference_times)
    return inference_times


@app.route('/', methods=['GET'])
def root_handle():
    return 'Group 40'


@app.route('/api/detect', methods=['POST'])
def main():
    # Local file path to the images
    data_input = request.values.get('input')
    # Binary variable whether to save the bounding boxes or not
    output = 'text' if request.values.get('output') == '1' else None

    path = data_input
    filename_images = {}
    input_format = ["jpg", "png", "jpeg"]
    # Input is a file
    if data_input.find(".") != -1:
        print(data_input + " is a file")
        split_data_input = data_input.split(".", 1)
        if data_input.endswith(tuple(input_format)):
            print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
            path_splitted = re.split('/', data_input)
            filename = path_splitted[len(path_splitted)-1]
            filename_images[filename] = None
            path = os.path.dirname(data_input)+"/"
    # Input is a directory
    else:
        print(data_input + " is a path with the following files: ")
        for filename in os.listdir(data_input):
            # image_path = data_input + filename
            filename_images[filename] = None
            print("  " + filename)

    inference_times = detection_loop(
        filename_images, path, output_type=output)

    # Serialize output and send response to client
    res = Response(json.dumps(inference_times),
                   status=200, mimetype='application/json')
    return res


@app.teardown_appcontext
def teardown_model(exception):
    _ = g.pop('model', None)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
