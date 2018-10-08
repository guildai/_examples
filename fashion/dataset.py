# Copyright 2017-2018 TensorHub, Inc.
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

import os

import numpy as np

from tensorflow import keras

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]

def load(from_dir=None):
    if from_dir:
        return _load_from_dir(from_dir)
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

def _load_from_dir(dir):
    return (
        (np.load(os.path.join(dir, "train-images.npy")),
         np.load(os.path.join(dir, "train-labels.npy"))),
        (np.load(os.path.join(dir, "test-images.npy")),
         np.load(os.path.join(dir, "test-labels.npy"))))

def save(data, output_dir):
    (train_images, train_labels), (test_images, test_labels) = data
    np.save(os.path.join(output_dir, "train-images"), train_images)
    np.save(os.path.join(output_dir, "train-labels"), train_labels)
    np.save(os.path.join(output_dir, "test-images"), test_images)
    np.save(os.path.join(output_dir, "test-labels"), test_labels)
