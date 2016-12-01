from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

import support

FLAGS = None

def train():

    # Placeholder inputs
    images, labels = support.placeholder_inputs()

    # Training data
    train_images, train_labels = support.data_inputs(
        FLAGS.datadir,
        support.TRAINING_DATA,
        FLAGS.batch_size)

    # Validation data
    validate_images, validate_labels = support.data_inputs(
        FLAGS.datadir,
        support.VALIDATION_DATA,
        FLAGS.batch_size)

    # Model and training ops
    predict = support.inference(images)
    loss = support.loss(predict, labels)
    global_step = tf.Variable(0, trainable=False)
    train, learning_rate = support.train(loss, global_step, FLAGS.batch_size)

    # Accuracy
    accuracy = support.accuracy(predict, labels)

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

    # Helper to read next batch
    def next_batch(images_source, labels_source):
        batch_images, batch_labels = sess.run([images_source, labels_source])
        return {
            images: batch_images,
            labels: batch_labels
        }

    # Helpder to validate
    def validate():
        step = 0
        steps = support.VALIDATION_IMAGES_COUNT // FLAGS.batch_size
        validate_accuracy = 0.0
        while step < steps:
            batch_accuracy = sess.run(
                accuracy, next_batch(validate_images, validate_labels))
            validate_accuracy += batch_accuracy / steps
            step += 1
        summary = tf.Summary()
        summary.value.add(tag="accuracy", simple_value=validate_accuracy)
        return summary, validate_accuracy

    # Helper to log status
    def log_status(step, train_summary, train_accuracy,
                   validate_summary=None, validate_accuracy=None):
        train_writer.add_summary(train_summary, step)
        if validate_summary is not None:
            validate_writer.add_summary(validate_summary)
            validate_writer.flush()
        print("Step %i: training=%f" % (step, train_accuracy), end="")
        if validate_accuracy is not None:
            print(" validation=%f" % validate_accuracy, end="")
        print()

    # Helper to save model
    saver = tf.train.Saver()
    def save_model():
        print("Saving trained model")
        tf.gfile.MakeDirs(FLAGS.rundir + "/model")
        saver.save(sess, FLAGS.rundir + "/model/export")

    # Training loop
    steps_per_epoch = support.TRAINING_IMAGES_COUNT // FLAGS.batch_size
    steps = steps_per_epoch * FLAGS.epochs
    step = 0
    while step < steps:
        _, train_summary, train_accuracy = sess.run(
            [train, summaries, accuracy],
            next_batch(train_images, train_labels))
        if step % 20 == 0:
            if step % steps_per_epoch == 0:
                validate_summary, validate_accuracy = validate()
                log_status(
                    step, train_summary, train_accuracy,
                    validate_summary, validate_accuracy)
                save_model()
            else:
                log_status(
                    step, train_summary, train_accuracy)
        step += 1

    # Final status
    train_summary, train_accuracy = sess.run(
        [summaries, accuracy],
        next_batch(train_images, train_labels))
    validate_summary, validate_accuracy = validate()
    log_status(
        step, train_summary, train_accuracy,
        validate_summary, validate_accuracy)

    # Save trained model
    save_model()

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
