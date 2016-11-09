# -*- coding: utf-8 -*-

""" Deep Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""
from __future__ import division, print_function, absolute_import

import argparse
import tflearn
import tflearn.datasets.mnist as mnist

def train(flags):
    # Data loading and preprocessing
    X, Y, testX, testY = mnist.load_data(flags.datadir, one_hot=True)

    # Building deep neural network
    input_layer = tflearn.input_data(shape=[None, 784])
    dense1 = tflearn.fully_connected(
        input_layer, 64, activation='tanh',
        regularizer='L2', weight_decay=0.001)
    dropout1 = tflearn.dropout(dense1, 0.8)
    dense2 = tflearn.fully_connected(
        dropout1, 64, activation='tanh',
        regularizer='L2', weight_decay=0.001)
    dropout2 = tflearn.dropout(dense2, 0.8)
    softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(
        softmax, optimizer=sgd, metric=top_k,
        loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_dir=flags.rundir)
    model.fit(
        X, Y,
        n_epoch=flags.epochs,
        validation_set=(testX, testY),
        show_metric=True,
        run_id="dense_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", default="/tmp/MNIST_train")
    parser.add_argument("--datadir", default="/tmp/MNIST_data",)
    parser.add_argument("--epochs", type=int, default=10)
    flags, _ = parser.parse_known_args()
    train(flags)
