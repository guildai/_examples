from __future__ import print_function

import argparse
import os

import tensorflow as tf

import support

FLAGS = None

def train():

    # Training data
    images, labels = support.inputs(
        FLAGS.datadir,
        support.TRAINING_DATA,
        FLAGS.batch_size)

    # Model and training ops
    predictions = support.inference(images, FLAGS.batch_size)
    loss = support.loss(predictions, labels)
    global_step = tf.Variable(0, trainable=False)
    train = support.train(loss, FLAGS.batch_size, global_step)

    # Summaries
    tf.scalar_summary("loss", loss)
    summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.rundir + "/train")

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialize queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Helper to log status
    def train_and_log():
        _, loss_, summaries_ = sess.run([train, loss, summaries])
        log_result(train_writer, step, loss_, summaries_)

    # Training loop
    steps = (support.TRAINING_IMAGES_COUNT // FLAGS.batch_size) * FLAGS.epochs
    step = 0
    while step < steps:
        if step % 20 == 0:
            train_and_log()
        else:
            sess.run(train)
        step += 1

    # Final status
    train_and_log()

    # Stop queue runners
    coord.request_stop()
    coord.join(threads)

def log_result(writer, step, loss, summary):
    writer.add_summary(summary, step)
    print("Step %i: loss=%f" % (step, loss))

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
