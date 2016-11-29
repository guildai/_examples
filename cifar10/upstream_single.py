from __future__ import print_function

import argparse
import os

import tensorflow as tf
import tensorflow.models.image.cifar10.cifar10 as cifar10
import tensorflow.models.image.cifar10.cifar10_train as cifar10_train

FLAGS = None

def main(_):
    cifar10_train.FLAGS.data_dir = FLAGS.datadir
    cifar10_train.FLAGS.train_dir = FLAGS.rundir + "/train"
    cifar10_train.FLAGS.max_steps = (FLAGS.epochs
                                     * cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    cifar10_train.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/tmp/CIFAR10_data",)
    parser.add_argument("--rundir", default="/tmp/CIFAR10_train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    FLAGS, _ = parser.parse_known_args()
    tf.app.run()
