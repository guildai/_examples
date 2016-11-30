import argparse
import json

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def train(mnist):
    # Input layer
    x = tf.placeholder(tf.float32, [None, 784])

    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # First fully connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder_with_default(1.0, [])
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Training operation
    y_ = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Evaluation operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summaries
    tf.scalar_summary("loss", loss)
    tf.scalar_summary("accuracy", accuracy)
    summaries = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.rundir + "/train")
    validation_writer = tf.train.SummaryWriter(FLAGS.rundir + "/validation")

    # Inputs/outputs for running exported model
    tf.add_to_collection("inputs", json.dumps({"image": x.name}))
    tf.add_to_collection("outputs", json.dumps({"prediction": y.name}))

    # Session and variable init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Helper to log status
    def log_model_status(step, train_images, train_labels):
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
        print "Step %i: training=%f validation=%f" % (
            step, train_accuracy, validation_accuracy)

    # Helper to save model
    saver = tf.train.Saver()
    def save_model():
        print "Saving trained model"
        tf.gfile.MakeDirs(FLAGS.rundir + "/model")
        saver.save(sess, FLAGS.rundir + "/model/export")

    # Training loop
    steps_per_epoch = mnist.train.num_examples // FLAGS.batch_size
    steps = steps_per_epoch * FLAGS.epochs
    for step in range(steps):
        images, labels = mnist.train.next_batch(FLAGS.batch_size)
        sess.run(train, feed_dict={x: images, y_: labels})
        if step % 20 == 0:
            log_model_status(step, images, labels)
        if step % steps_per_epoch == 0:
            save_model()

    # Final status
    images, labels = mnist.train.next_batch(FLAGS.batch_size)
    log_model_status(step + 1, images, labels)

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
