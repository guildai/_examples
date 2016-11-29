from __future__ import print_function

import argparse
import os

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
    train_predict = support.inference(train_images, FLAGS.batch_size)
    loss = support.loss(train_predict, train_labels)
    global_step = tf.Variable(0, trainable=False)
    train, learning_rate = support.train(loss, FLAGS.batch_size, global_step)

    # Accuracy
    train_accuracy = support.accuracy(train_predict, train_labels)
    validate_predict = support.inference(validate_images, FLAGS.batch_size)
    validate_accuracy = support.accuracy(validate_predict, validate_labels)

    # Summaries
    tf.scalar_summary("loss", loss)
    tf.scalar_summary("accuracy", train_accuracy)
    tf.scalar_summary("learning_rate", learning_rate)
    train_summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.rundir + "/train")

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialize queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Helper to log status
    def log_status(step):
        loss_, summaries, accuracy = sess.run(
            [loss, train_summaries, train_accuracy])
        train_writer.add_summary(summaries, step)
        print("Step %i: loss=%f accuracy=%s" % (step, loss_, accuracy))

    # Training loop
    steps = (support.TRAINING_IMAGES_COUNT // FLAGS.batch_size) * FLAGS.epochs
    step = 0
    while step < steps:
        sess.run(train)
        if step % 20 == 0:
            log_status(step)
        step += 1

    # Final status
    log_status(step)

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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    FLAGS, _ = parser.parse_known_args()
    tf.app.run()
