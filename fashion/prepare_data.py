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

import argparse
import logging
import os
import sys

import numpy as np

import tensorflow as tf

import fig
import train

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s")

log = logging.getLogger()

def main(argv):
    args = _init_args(argv)
    raw_data = _load_raw_data()
    _init_output_dir(args)
    _write_sample_raw_images(raw_data, args)
    processed_data = _process_data(raw_data)
    _write_sample_processed_images(processed_data, args)
    _save_data(processed_data, args.output_dir)

def _init_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o", "--output-dir", default=".",
        help="Directory to save output")
    return p.parse_args(argv[1:])

def _init_output_dir(args):
    log.info("Output directory is %s", args.output_dir)
    tf.gfile.MakeDirs(args.output_dir)

def _load_raw_data():
    log.info("Loading raw data")
    return train.load_data()

def _write_sample_raw_images(data, args):
    log.info("Writing sample raw images")
    (train_images, _), _ = data
    for i in range(25):
        fig.write_image(
            train_images[i],
            _output_path("sample-raw-%0.2i.png" % i, args))

def _process_data(data):
    log.info("Processing data")
    (train_images, train_labels), (test_images, test_labels) = data
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

def _write_sample_processed_images(data, args):
    log.info("Writing sample processed images")
    (train_images, train_labels), _ = data
    _write_sample_images(
        train_images[:25],
        train_labels[:25],
        _output_path("sample-processed.png", args))

def _save_data(data, output_dir):
    log.info("Writing processed data")
    (train_images, train_labels), (test_images, test_labels) = data
    np.save(os.path.join(output_dir, "train-images"), train_images)
    np.save(os.path.join(output_dir, "train-labels"), train_labels)
    np.save(os.path.join(output_dir, "test-images"), test_images)
    np.save(os.path.join(output_dir, "test-labels"), test_labels)

def _write_sample_images(images, labels, name):
    labels = [train.class_names[label_id] for label_id in labels]
    fig.write_image_grid(images, labels, name)

def _output_path(name, args):
    return os.path.join(args.output_dir, name)

if __name__ == "__main__":
    main(sys.argv)
