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
DEFAULT_DATA_DIR = "/tmp/fashion-data"
DEFAULT_CHECKPOINT_DIR = "/tmp/fashion-train"
DEFAULT_LOG_DIR = "/tmp/fashion-train"

def main(argv):
    args = _init_args(argv)
    data = _load_data(args)
    model = _init_model()
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

def _init_model():
    return model.init()

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
         _checkpoint_callback(args),
     ]

def _tensorboard_callback(args):
    return tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)

def _checkpoint_callback(args):
    file_pattern = os.path.join(
        args.checkpoint_dir,
        "weights-{epoch:02d}-{loss:0.3f}.hdf5")
    return tf.keras.callbacks.ModelCheckpoint(file_pattern)

def _test_model(model, data):
    log.info("Evaluating trained model")
    _, (test_images, test_labels) = data
    _loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy: %s" % test_acc)

def blah():
    """



    predictions = model.predict(test_images)

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)
    plt.savefig("huh.png")

    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, test_labels)
    plt.savefig("huh-2.png")


    # Plot the first X test images, their predicted label, and the
    # true label# Plot Color correct predictions in blue, incorrect
    # predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
    plt.savedif("huh-3.png")

    img = test_images[0]

    print(img.shape)

    img = (np.expand_dims(img, 0))

    print(img.shape)

    predictions_single = model.predict(img)

    print(predictions_single)

    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.savefig("huh-4.png")

    np.argmax(predictions_single[0])
    """

delme = """
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
"""

if __name__ == "__main__":
    main(sys.argv)
