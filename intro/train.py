"""Simple MNIST learner adapted from Hvass-Labs/TensorFlow-Tutorials:

https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb

Copyright 2019 TensorHub, Inc.

Copyright (c) 2016-2018 by Magnus Erik Hvass Pedersen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function

import argparse
import os
import time

from tensorflow.python.keras import callbacks
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Data dimensions

IMG_SIZE = 28
IMG_SIZE_flat = IMG_SIZE * IMG_SIZE
##IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
IMG_SHAPE_FULL = (IMG_SIZE, IMG_SIZE, 1)
##NUM_CHANNELS = 1
NUM_CLASSES = 10

def init_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--dense-layers', type=int, default=1)
    p.add_argument('--dense-nodes', type=int, default=16)
    p.add_argument('--activation', choices=('relu', 'sigmoid'), default='relu')
    p.add_argument('--log-dir', default='./logs')
    p.add_argument('--data-dir', default='/tmp/MNIST_data')
    p.add_argument('--checkpoint-dir', default='./ckpt')
    p.add_argument('--plots-dir', default='plots')
    return p.parse_args()

def init_model(learning_rate, dense_layers, dense_nodes, activation):
    """
    Hyper-parameters:
    learning_rate: Learning-rate for the optimizer.
    dense_layers:  Number of dense layers.
    dense_nodes:   Number of nodes in each dense layer.
    activation:    Activation function for all layers.
    """

    # Start construction of a Keras Sequential model.
    model = models.Sequential()

    # Add an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    model.add(layers.InputLayer(input_shape=(IMG_SIZE_flat,)))

    # The input from MNIST is a flattened array with 784 elements,
    # but the convolutional layers expect images with shape (28, 28, 1)
    model.add(layers.Reshape(IMG_SHAPE_FULL))

    # First convolutional layer.
    # There are many hyper-parameters in this layer, but we only
    # want to optimize the activation-function in this example.
    model.add(layers.Conv2D(
        kernel_size=5, strides=1, filters=16, padding='same',
        activation=activation, name='layer_conv1'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))

    # Second convolutional layer.
    # Again, we only want to optimize the activation-function here.
    model.add(layers.Conv2D(
        kernel_size=5, strides=1, filters=36, padding='same',
        activation=activation, name='layer_conv2'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))

    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(layers.Flatten())

    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(layers.Dense(
            dense_nodes, activation=activation, name=name))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    # Use the Adam method for training the network.
    # We want to find the best learning-rate for the Adam method.
    optimizer = optimizers.Adam(lr=learning_rate)

    # In Keras we need to compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def init_data(args):
    data = mnist_data.read_data_sets(args.data_dir, one_hot=True)
    plot_samples(data, args)

def plot_samples(data, args):
    images = data.test.images[0:9]
    cls_true = data.test.cls[0:9]
    path = os.path.join(args.plot_dir, "samples.png")
    util.plot_images(path, images, cls_true)

def train_model(model, data, args):
    callbacks = [
        tensorboard_callback(args),
        checkpoint_callback(args)]
    return model.fit(
        x=data.train.images,
        y=data.train.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            data.validation.images,
            data.validation.labels),
        callbacks=callbacks)

def tensorboard_callback(args):
    return callbacks.TensorBoard(
        log_dir=args.log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)

def checkpoint_callback(args):
    try:
        os.makedirs(args.checkpoint_dir)
    except OSError:
        pass
    timestamp = int(time.time())
    pattern = "weights-%i-{epoch:05d}.h5" % timestamp
    filepath = os.path.join(args.checkpoint_dir, pattern)
    return callbacks.ModelCheckpoint(filepath)

args = init_args()

model = init_model(
    args.learning_rate,
    args.dense_layers,
    args.dense_nodes,
    args.activation)

data = init_data(args)

train_model(model, data, args)
