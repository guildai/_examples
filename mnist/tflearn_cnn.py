# -*- coding: utf-8 -*-

"""Convolutional Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

This code was original copied from
https://raw.githubusercontent.com/tflearn/tflearn/master/examples/images/convnet_mnist.py

"""

from __future__ import division, print_function, absolute_import

import argparse
import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.normalization import local_response_normalization
#from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

def train(flags):
    # Data loading and preprocessing
    X, Y, testX, testY = mnist.load_data(flags.datadir, one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])

    # Building convolutional network
    net = tflearn.input_data(shape=[None, 28, 28, 1], name='input')
    net = tflearn.conv_2d(net, 32, 3, activation='relu', regularizer="L2")
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.conv_2d(net, 64, 3, activation='relu', regularizer="L2")
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.local_response_normalization(net)
    net = tflearn.fully_connected(net, 128, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 256, activation='tanh')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(
        net, optimizer='adam', learning_rate=0.01,
        loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(net, tensorboard_dir=flags.rundir)
    model.fit(
        {'input': X},
        {'target': Y},
        n_epoch=flags.epochs,
        validation_set=({'input': testX}, {'target': testY}),
        snapshot_step=100, show_metric=True, run_id='convnet_mnist')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", default="/tmp/MNIST_train")
    parser.add_argument("--datadir", default="/tmp/MNIST_data",)
    parser.add_argument("--epochs", type=int, default=10)
    flags, _ = parser.parse_known_args()
    train(flags)
