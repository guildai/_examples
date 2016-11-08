import argparse
import json

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def train(mnist):

    # Softmax regression model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Training operation
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                         reduction_indices=[1]))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Evaluation operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summaries
    tf.scalar_summary("loss", loss)
    tf.scalar_summary("accuracy", accuracy)
    summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.rundir + "/train",
                                          tf.get_default_graph())
    validation_writer = tf.train.SummaryWriter(FLAGS.rundir + "/validation")

    # Inputs/outputs for running exported model
    tf.add_to_collection("inputs", json.dumps({"image": x.name}))
    tf.add_to_collection("outputs", json.dumps({"prediction": y.name}))

    # Session and variable init
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Helper to write log performance
    def write_model_status(label, step, train_images, train_labels):
        train_data = {
            x: train_images,
            y_: train_labels
        }
        train_accuracy, train_summary = \
            sess.run([accuracy, summaries], feed_dict=train_data)
        train_writer.add_summary(train_summary, step)
        validation_data = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        validation_accuracy, validation_summary = \
            sess.run([accuracy, summaries], feed_dict=validation_data)
        validation_writer.add_summary(validation_summary, step)
        print "%s (step %i): training=%f validation=%f" % (
            label, step, train_accuracy, validation_accuracy)

    saver = tf.train.Saver()
    def save_model():
        print "Saving trained model"
        tf.gfile.MakeDirs(FLAGS.rundir + "/model")
        saver.save(sess, FLAGS.rundir + "/model/export")

    # Batch training over all training examples per epoch
    steps = (mnist.train.num_examples / FLAGS.batch_size) * FLAGS.epochs
    for step in range(steps):
        images, labels = mnist.train.next_batch(FLAGS.batch_size)
        sess.run(train, feed_dict={x: images, y_: labels})
        if step % 20 == 0:
            write_model_status("Batch", step, images, labels)
        if step != 0 and step % (mnist.train.num_examples /
                                 FLAGS.batch_size) == 0:
            save_model()

    # Final status
    images, labels = mnist.train.next_batch(FLAGS.batch_size)
    write_model_status("Final", step + 1, images, labels)

    # Save trained model
    tf.add_to_collection("x", x.name)
    tf.add_to_collection("y_", y_.name)
    tf.add_to_collection("accuracy", accuracy.name)
    save_model()

def evaluate(mnist):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(FLAGS.rundir + "/model/export.meta")
    saver.restore(sess, FLAGS.rundir + "/model/export")
    accuracy = sess.graph.get_tensor_by_name(tf.get_collection("accuracy")[0])
    x = sess.graph.get_tensor_by_name(tf.get_collection("x")[0])
    y_ = sess.graph.get_tensor_by_name(tf.get_collection("y_")[0])
    test_data = {
        x: mnist.test.images,
        y_: mnist.test.labels
    }
    test_accuracy = sess.run(accuracy, feed_dict=test_data)
    print "Test accuracy: %f" % test_accuracy

def main(_):
    mnist = input_data.read_data_sets(FLAGS.datadir, one_hot=True)
    if FLAGS.prepare:
        pass
    elif FLAGS.evaluate:
        evaluate(mnist)
    else:
        train(mnist)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/tmp/MNIST_data",)
    parser.add_argument("--rundir", default="/tmp/MNIST_train")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    FLAGS, _ = parser.parse_known_args()
    tf.app.run()
