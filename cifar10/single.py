from __future__ import print_function

import argparse
import os

import tensorflow as tf

import support

FLAGS = None

def train():

    images, labels = support.inputs(
        FLAGS.datadir,
        support.TRAINING_DATA,
        FLAGS.batch_size)

    predictions = support.inference(images, FLAGS.batch_size)

    loss = support.loss(predictions, labels)

    train = support.train(loss, FLAGS.batch_size)

    with training_session(loss) as session:
        while not session.should_stop():
            session.run(train)

def training_session(loss):
    steps = support.TRAINING_IMAGES_COUNT * FLAGS.epochs
    hooks = [
        tf.train.StopAtStepHook(last_step=steps),
        tf.train.NanTensorHook(loss),
        TrainLogHook(loss)
    ]
    train_dir = os.path.join(FLAGS.rundir, "train")
    return tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=hooks)

class TrainLogHook(tf.train.SessionRunHook):

    def __init__(self, loss):
        super(TrainLogHook, self).__init__()
        self.loss = loss

    def begin(self):
        self.step = -1

    def before_run(self, context):
        self.step += 1
        return tf.train.SessionRunArgs(self.loss)

    def after_run(self, context, values):
        if self.step % 20 == 0:
            loss = values.results
            print("Step %i: loss=%f" % (self.step, loss))

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
