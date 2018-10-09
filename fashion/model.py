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

import glob
import os
import re

import tensorflow as tf

from tensorflow import keras

def init(**optimizer_kw):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(
        optimizer=tf.train.AdamOptimizer(**optimizer_kw),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

def checkpoint_callback(checkpoint_dir):
    file_pattern = os.path.join(
        checkpoint_dir,
        "weights-{epoch:04d}-{loss:0.3f}.hdf5")
    return keras.callbacks.ModelCheckpoint(file_pattern)

def load(checkpoint_dir, epoch=None):
    model = init()
    model.load_weights(_checkpoint_path(checkpoint_dir, epoch))
    return model

def _checkpoint_path(checkpoint_dir, epoch):
    if epoch:
        epoch = "{:04d}".format(epoch)
    else:
        epoch = _latest_epoch(checkpoint_dir)
    pattern = os.path.join(checkpoint_dir, "weights-%s-*.hdf5" % epoch)
    matches = glob.glob(pattern)
    if not matches:
        raise RuntimeError(
            "cannot find Keras checkpoint matching %s"
            % pattern)
    return matches[0]

def _latest_epoch(checkpoint_dir):
    latest = None
    for name in os.listdir(checkpoint_dir):
        m = re.search(r"weights-(\d+)-.+\.hdf5$", name)
        if m:
            if latest is None or int(m.group(1)) > int(latest):
                latest = m.group(1)
    if latest is None:
        raise RuntimeError(
            "cannot find latest Keras checkpoint in %s"
            % checkpoint_dir)
    return latest
