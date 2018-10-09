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
import sys

import tensorflow as tf

import dataset
import model

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s")

log = logging.getLogger()

DEFAULT_EPOCHS = 5
DEFAULT_LR = 0.001
DEFAULT_DATA_DIR = "/tmp/fashion-data"
DEFAULT_CHECKPOINT_DIR = "/tmp/fashion-train"
DEFAULT_LOG_DIR = "/tmp/fashion-train"

def main(argv):
    args = _init_args(argv)
    data = _load_data(args)
    model = _init_model(args)
    _init_output_dirs(args)
    _train_model(model, data, args)
    _test_model(model, data)

def _init_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-e", "--epochs",
        default=DEFAULT_EPOCHS, type=int,
        help="number of epochs to train (%s)" % DEFAULT_EPOCHS)
    p.add_argument(
        "-r", "--learning-rate", default=DEFAULT_LR, type=float,
        help="learning rate")
    p.add_argument(
        "-d", "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory containing prepare data (%s)" % DEFAULT_DATA_DIR)
    p.add_argument(
        "-c", "--checkpoint-dir",
        default=DEFAULT_DATA_DIR,
        help="directory to write checkpoints (%s)" % DEFAULT_CHECKPOINT_DIR)
    p.add_argument(
        "-l", "--log-dir",
        default=DEFAULT_DATA_DIR,
        help="directory to write logs (%s)" % DEFAULT_LOG_DIR)
    return p.parse_args(argv[1:])

def _load_data(args):
    log.info("Loading data from %s", args.data_dir)
    return dataset.load(args.data_dir)

def _init_model(args):
    return model.init(learning_rate=args.learning_rate)

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
    return [
        _tensorboard_callback(args),
        model.checkpoint_callback(args.checkpoint_dir),
    ]

def _tensorboard_callback(args):
    return tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)

def _test_model(model, data):
    log.info("Evaluating trained model")
    _, (test_images, test_labels) = data
    _loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy: %s" % test_acc)

if __name__ == "__main__":
    main(sys.argv)
