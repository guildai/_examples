from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

import support

FLAGS = None

def train():

    # Training data
    train_images, train_labels = support.inputs(
        FLAGS.datadir,
        support.TRAINING_DATA,
        FLAGS.batch_size)

    # Validation data
    validate_images, validate_labels = support.inputs(
        FLAGS.datadir,
        support.VALIDATION_DATA,
        FLAGS.batch_size)

    # Model and training ops
    validate_flag = tf.placeholder(tf.bool, ())
    predict = support.inference(
        train_images, validate_images, FLAGS.batch_size, validate_flag)
    loss = support.loss(predict, train_labels)
    global_step = tf.Variable(0, trainable=False)
    train, learning_rate = support.train(loss, FLAGS.batch_size, global_step)

    # Accuracy
    accuracy = support.accuracy(
        predict, train_labels, validate_labels, validate_flag)

    # Summaries
    tf.scalar_summary("loss", loss)
    tf.scalar_summary("accuracy", accuracy)
    tf.scalar_summary("learning_rate", learning_rate)
    summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.rundir + "/train")
    validate_writer = tf.train.SummaryWriter(FLAGS.rundir + "/validation")

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialize queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Helpers to log status
    def log_status(train_step, validate=False):
        print("Step %i:" % train_step, end="")
        log_train_status(train_step)
        if validate:
            log_validate_status(train_step)
        print()

    def log_train_status(step):
        summaries_, accuracy_ = sess.run(
            [summaries, accuracy],
            feed_dict={validate_flag: False})
        train_writer.add_summary(summaries_, step)
        train_writer.flush()
        print(" training=%f" % accuracy_, end="")

    def log_validate_status(train_step):
        accuracies = []
        validate_steps = support.VALIDATION_IMAGES_COUNT // FLAGS.batch_size
        step = 0
        while step < validate_steps:
            accuracy_ = sess.run(accuracy, feed_dict={validate_flag: True})
            accuracies.append(accuracy_)
            step += 1
        validation_accuracy = float(np.mean(accuracies))
        summary = tf.Summary()
        summary.value.add(tag="accuracy", simple_value=validation_accuracy)
        validate_writer.add_summary(summary, train_step)
        validate_writer.flush()
        print(" validation=%f" % validation_accuracy, end="")

    # Training loop
    steps_per_epoch = support.TRAINING_IMAGES_COUNT // FLAGS.batch_size
    train_steps = steps_per_epoch * FLAGS.epochs
    step = 0
    while step < train_steps:
        sess.run(train, feed_dict={validate_flag: False})
        if step % 20 == 0:
            validate = step % steps_per_epoch == 0
            log_status(step, validate)
        step += 1

    # Final status
    log_status(step, True)

    # Stop queue runners
    coord.request_stop()
    coord.join(threads)

def evaluate():
    print("TODO evaluate")

def main(_):
    support.ensure_data(FLAGS.datadir)
    if FLAGS.prepare:
        pass
    elif FLAGS.evaluate:
        evaluate()
    else:
        train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/tmp/CIFAR10_data",)
    parser.add_argument("--rundir", default="/tmp/CIFAR10_train")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    FLAGS, _ = parser.parse_known_args()
    tf.app.run()
