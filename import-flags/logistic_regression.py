"""Logistic regression example using TensorFlow on MNIST.

Code is adapted from
https://github.com/aymericdamien/TensorFlow-Examples/ by Aymeric
Damien.

All contributions by Aymeric Damien:
Copyright (c) 2015, Aymeric Damien.
All rights reserved.

All other contributions:
Copyright (c) 2015, the respective contributors.
All rights reserved.

"""
from __future__ import print_function

import argparse

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
p = argparse.ArgumentParser()
p.add_argument(
    "--learning_rate", default=0.01,
    help="Learning rate")
p.add_argument(
    "--training_epochs", default=25,
    help="Number of epochs to train")
p.add_argument(
    "--batch_size", default=100,
    help="Training batch size")
p.add_argument(
    "--display_step", default=1,
    help="Log frequency")
p.add_argument(
    "--data_dir", default="/tmp/data/",
    help="Location of downloaded data")
p.add_argument(
    "--prepare", action="store_true",
    help="Download data without training")
args = p.parse_args()

# Download MNIST data
mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
if args.prepare:
    exit()

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(args.training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/args.batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % args.display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
