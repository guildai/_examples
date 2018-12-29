# Copyright 2017-2019 TensorHub, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import train

def write_image(image, path):
    f = plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.savefig(path)
    plt.close(f)

def write_image_grid(images, labels, path):
    f = plt.figure(figsize=(10, 10))
    for i, image, label in zip(range(len(images)), images, labels):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(label)
        plt.savefig(path)
    plt.close(f)

def write_image_prediction(prediction, image, image_label, path):
    f = plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    _plot_predicted_image(prediction, image, image_label)
    plt.subplot(1, 2, 2)
    _plot_predicted_label(prediction, image_label)
    plt.savefig(path)
    plt.close(f)

def _plot_predicted_image(prediction, image, image_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction)
    if predicted_label == image_label:
        color = "blue"
        error_suffix = ""
    else:
        color = "red"
        error_suffix = "\n(actually %s)" % train.class_names[image_label]
    caption = "%s %0.2f%s" % (
        train.class_names[predicted_label],
        100 * np.max(prediction),
        error_suffix)
    plt.xlabel(caption, color=color)

def _plot_predicted_label(prediction, image_label):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    class_count = len(train.class_names)
    chart = plt.bar(range(class_count), prediction, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)
    chart[predicted_label].set_color("red")
    chart[image_label].set_color("blue")
