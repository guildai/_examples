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

import argparse
import logging
import os
import sys

import numpy as np

import tensorflow as tf

from tensorflow import keras

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s")

log = logging.getLogger()

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

def main(argv):
    args = _init_args(argv)
    data = load_data(args.data_dir)
    model = init_model(learning_rate=args.learning_rate)
    _init_output_dirs(args)
    _train_model(model, data, args)
    _test_model(model, data)

def _init_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-e", "--epochs",
        default=5, type=int,
        help="number of epochs to train (default is 5)")
    p.add_argument(
        "-r", "--learning-rate", default=0.001, type=float,
        help="learning rate (default is 0.001)")
    p.add_argument(
        "-d", "--data-dir",
        help=(
            "directory containing prepare data (default is to "
            "download raw data)"))
    p.add_argument(
        "-c", "--checkpoint-dir", default=".",
        help="directory to write checkpoints (default is current directory)")
    p.add_argument(
        "-l", "--log-dir", default=".",
        help="directory to write logs (default is current directory)")
    return p.parse_args(argv[1:])

def load_data(from_dir=None):
    if from_dir:
        log.info("Loading data from %s", from_dir)
        return _load_prepared_data(from_dir)
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

def _load_prepared_data(dir):
    return (
        (np.load(os.path.join(dir, "train-images.npy")),
         np.load(os.path.join(dir, "train-labels.npy"))),
        (np.load(os.path.join(dir, "test-images.npy")),
         np.load(os.path.join(dir, "test-labels.npy"))))

def init_model(**optimizer_kw):
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

def _init_output_dirs(args):
    log.info("Checkpoints will be written to %s", args.checkpoint_dir)
    tf.gfile.MakeDirs(args.checkpoint_dir)
    log.info("Logs will be written to %s", args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)

def _train_model(model, data, args):
    log.info("Training model")
    (train_images, train_labels), _ = data
    model.fit(
        train_images,
        train_labels,
        epochs=args.epochs,
        callbacks=_train_callbacks(args))

def _train_callbacks(args):
    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)
    ]
    checkpoint_cb = _try_checkpoint_callback(args.checkpoint_dir)
    if checkpoint_cb:
        cbs.append(checkpoint_cb)
    return cbs

def _try_checkpoint_callback(checkpoint_dir):
    try:
        import h5py
    except ImportError:
        log.warning("h5py is not available - checkpoints are disabled")
        return None
    else:
        file_pattern = os.path.join(
            checkpoint_dir,
            "weights-{epoch:04d}-{loss:0.3f}.hdf5")
        return keras.callbacks.ModelCheckpoint(file_pattern)

def _test_model(model, data):
    log.info("Evaluating trained model")
    _, (test_images, test_labels) = data
    _loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy: %s" % test_acc)

if __name__ == "__main__":
    main(sys.argv)
