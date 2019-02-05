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
import glob
import logging
import os
import re
import sys
import time

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
    model = _load_model(args)
    data = _load_data(args)
    _init_output_dir(args)
    _predict(model, data, args)

def _init_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-n", "--count", default=None, type=int,
        help="Number of randomly selected examples to predict")
    p.add_argument(
        "-r", "--range",
        help=("Range of examples to predict in the format START[:STOP] "
              "where START is zero-based inclusive and STOP is "
              "zero-based exclusive; if STOP is omitted, only one example "
              "START is predicted"))
    p.add_argument(
        "-E", "--errors-only", action="store_true",
        help="Only show incorrect predictions")
    p.add_argument(
        "-d", "--data-dir",
        help=(
            "Directory containing prepare data; if not specified, "
            "downloads raw data"))
    p.add_argument(
        "-c", "--checkpoint-dir",
        default="model",
        help="Directory containing model checkpoints")
    p.add_argument(
        "-e", "--checkpoint-epoch", type=int,
        help="Checkpoint epoch to use; default is latest available")
    p.add_argument(
        "-o", "--output-dir", default=".",
        help="Directory to save output")
    return p.parse_args(argv[1:])

def _load_model(args):
    if not os.path.exists(args.checkpoint_dir):
        raise SystemExit(
            "predict: checkpoint directory %s does not exist"
            % args.checkpoint_dir)
    log.info("Loading trained model from %s", args.checkpoint_dir)
    model = train.init_model()
    model.load_weights(_checkpoint_path(args))
    return model

def _checkpoint_path(args):
    if args.checkpoint_epoch:
        epoch = "{:04d}".format(args.checkpoint_epoch)
    else:
        epoch = _latest_epoch(args.checkpoint_dir)
    pattern = os.path.join(args.checkpoint_dir, "weights-%s-*.hdf5" % epoch)
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(
            "predict: cannot find Keras checkpoint matching %s"
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
        raise SystemExit(
            "predict: cannot find latest Keras checkpoint in %s"
            % checkpoint_dir)
    return latest

def _load_data(args):
    return train.load_data(args.data_dir)

def _init_output_dir(args):
    log.info("Output directory is %s", args.output_dir)
    tf.gfile.MakeDirs(args.output_dir)

def _predict(model, data, args):
    _, (test_images, test_labels) = data
    predictions = model.predict(test_images)
    name_pattern = os.path.join(args.output_dir, "{:05d}{}")
    indexes = _example_indexes(predictions, test_labels, args)
    log.info("Generating predictions for %i test image(s)", len(indexes))
    start = time.time()
    for i in indexes:
        suffix = _predict_file_suffix(predictions[i], test_labels[i])
        fig.write_image_prediction(
            predictions[i],
            test_images[i],
            test_labels[i],
            name_pattern.format(i, suffix)
        )
        _tick(start)
    _tick(start, nl=True)

def _example_indexes(predictions, labels, args):
    assert len(predictions) == len(labels)
    if args.range:
        range_indexes = _example_range(args.range, len(predictions))
        return _filter_errors(range_indexes, predictions, labels, args)
    else:
        return _random_indexes(
            args.count or 5,
            predictions,
            labels,
            args.errors_only)

def _example_range(range_s, examples_count):
    m = re.search(r"(\d+)(?::(\d+))?$", range_s)
    if not m:
        raise SystemExit(
            "predict: invalid range %s: must be in the "
            "format START[:STOP]" % range_s)
    start = max(0, int(m.group(1)))
    stop = min((m.group(2) and int(m.group(2)) or (start + 1)), examples_count)
    return range(start, stop)

def _filter_errors(indexes, predictions, labels, args):
    if not args.errors_only:
        return indexes
    return [i for i in indexes if np.argmax(predictions[i]) != labels[i]]

def _random_indexes(index_count, predictions, labels, errors_only=False):
    assert len(predictions) == len(labels)
    if not errors_only:
        return np.random.randint(len(predictions) - 1, size=index_count)
    return _random_error_indexes(index_count, predictions, labels)

def _random_error_indexes(index_count, predictions, labels):
    errors = [
        i for i in range(len(predictions))
        if np.argmax(predictions[i]) != labels[i]]
    indexes = []
    while len(indexes) < index_count and errors:
        indexes.append(errors.pop(np.random.randint(len(errors))))
    return indexes

def _predict_file_suffix(prediction, label):
    if np.argmax(prediction) != label:
        return "-error"
    else:
        return ""

def _tick(start, nl=False):
    if time.time() > start + 2:
        sys.stdout.write(".")
        if nl:
            sys.stdout.write("\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main(sys.argv)
