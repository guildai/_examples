from __future__ import print_function
from __future__ import division

import os
import sys
import tarfile
import urllib

import tensorflow as tf

DATA_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
DATA_BIN_NAME = "cifar-10-batches-bin"

TRAINING_IMAGES_COUNT = 50000
VALIDATION_IMAGES_COUNT = 10000
CLASS_COUNT = 10

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3

CROPPED_IMAGE_HEIGHT = 24
CROPPED_IMAGE_WIDTH = 24

INPUT_LABEL_BYTES = 1
INPUT_IMAGE_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH
INPUT_RECORD_BYTES = INPUT_LABEL_BYTES + INPUT_IMAGE_BYTES

TRAINING_DATA = 1
AUGMENTED_TRAINING_DATA = 2
VALIDATION_DATA = 3

###################################################################
# Download data
###################################################################

def ensure_data(data_dir):
    data_bin = os.path.join(data_dir, DATA_BIN_NAME)
    if os.path.exists(data_bin):
        return
    data_tar = os.path.join(data_dir, DATA_URL.split("/")[-1])
    if not os.path.exists(data_tar):
        download_data(data_tar)
    print("Extracting files from", data_tar)
    tarfile.open(data_tar, "r:gz").extractall(data_dir)

def download_data(dest):
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    def progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%'
            % (dest, count * block_size / total_size * 100.0))
        sys.stdout.flush()
    urllib.urlretrieve(DATA_URL, dest, progress)
    print()
    statinfo = os.stat(dest)
    print("Downloaded", dest, statinfo.st_size, "bytes")

###################################################################
# Inputs
###################################################################

def placeholder_inputs():
    images = tf.placeholder(tf.float32, [None, CROPPED_IMAGE_HEIGHT,
                                         CROPPED_IMAGE_WIDTH, IMAGE_DEPTH])
    labels = tf.placeholder(tf.int32, [None])
    return images, labels

def data_inputs(data_dir, data_type, batch_size, runner_threads):

    # Input file reader
    filenames = input_filenames(data_dir, data_type)
    queue = tf.train.string_input_producer(filenames)
    reader = tf.FixedLengthRecordReader(record_bytes=INPUT_RECORD_BYTES)

    # Decode label and image
    _key, record_raw = reader.read(queue)
    record = tf.decode_raw(record_raw, tf.uint8)
    label = tf.cast(tf.slice(record, [0], [INPUT_LABEL_BYTES]), tf.int32)
    image = tf.reshape(
        tf.slice(record, [INPUT_LABEL_BYTES], [INPUT_IMAGE_BYTES]),
        [IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH])

    # Transpose image from stored DHW to HWD
    image_hwd = tf.transpose(image, [1, 2, 0])

    # Finalize image
    image_float = tf.cast(image_hwd, tf.float32)
    if data_type == AUGMENTED_TRAINING_DATA:
        image_final = augmented_standardized_image(image_float)
    else:
        image_final = standardized_image(image_float)

    # Process image and labels using queue runner
    images, labels = tf.train.batch(
        [image_final, label],
        batch_size=batch_size,
        num_threads=runner_threads,
        capacity=10 * batch_size)
    return images, tf.reshape(labels, [batch_size])

def input_filenames(data_dir, data_type):
    def data_path(name):
        return os.path.join(data_dir, DATA_BIN_NAME, name)
    if data_type == TRAINING_DATA:
        return [data_path("data_batch_%i.bin" % i) for i in range(1, 6)]
    elif data_type == VALIDATION_DATA:
        return [data_path("test_batch.bin")]
    else:
        raise ValueError(data_type)

def augmented_standardized_image(image_in):
    image_cropped = tf.random_crop(
        image_in,
        [CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH, IMAGE_DEPTH])
    image_flipped = tf.image.random_flip_left_right(image_cropped)
    image_bright = tf.image.random_brightness(image_flipped, max_delta=63)
    image_contrast = tf.image.random_contrast(
        image_bright, lower=0.2, upper=1.8)
    return tf.image.per_image_standardization(image_contrast)

def standardized_image(image_in):
    image_cropped = tf.image.resize_image_with_crop_or_pad(
        image_in,
        CROPPED_IMAGE_HEIGHT,
        CROPPED_IMAGE_WIDTH)
    return tf.image.per_image_standardization(image_cropped)

###################################################################
# Inference
###################################################################

def inference(images):

    # First convolutional layer
    W_conv1 = weight_variable([5, 5, 3, 64], 0.05)
    b_conv1 = bias_variable([64], 0.1)
    h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
    h_pool1 = max_pool_3x3(h_conv1)
    h_norm1 = lrn(h_pool1)

    # Second convolutional layer
    W_conv2 = weight_variable([5, 5, 64, 64], 0.05)
    b_conv2 = bias_variable([64], 0.1)
    h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)
    h_norm2 = lrn(h_conv2)
    h_pool2 = max_pool_3x3(h_norm2)

    # First locally connected layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 64])
    W_local1 = weight_variable([6 * 6 * 64, 384], 0.04, 0.004)
    b_local1 = bias_variable([384], 0.1)
    h_local1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_local1) + b_local1)

    # Second locally connected layer
    W_local2 = weight_variable([384, 192], 0.04, 0.004)
    b_local2 = bias_variable([192], 0.1)
    h_local2 = tf.nn.relu(tf.matmul(h_local1, W_local2) + b_local2)

    # Output layer
    W_out = weight_variable([192, CLASS_COUNT], 1 / 192)
    b_out = bias_variable([CLASS_COUNT], 0.0)
    output = tf.matmul(h_local2, W_out) + b_out

    return output

def weight_variable(shape, stddev, decay=None):
    W = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if decay is not None:
        loss = tf.mul(tf.nn.l2_loss(W), decay)
        tf.add_to_collection("losses", loss)
    return W

def bias_variable(shape, initial_value):
    return tf.Variable(tf.constant(initial_value, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding="SAME")

def max_pool_3x3(x):
    return tf.nn.max_pool(
        x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

def lrn(x):
    return tf.nn.local_response_normalization(
        x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

###################################################################
# Loss
###################################################################

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                       logits, labels)
    loss = tf.reduce_mean(cross_entropy)
    tf.add_to_collection("losses", loss)
    return tf.add_n(tf.get_collection("losses"))

###################################################################
# Train
###################################################################

def learning_rate(global_step, decay_steps):
    return tf.train.exponential_decay(
        0.1, global_step, decay_steps, 0.1, staircase=True)

def train(loss, learning_rate, global_step):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss)
    train = optimizer.apply_gradients(gradients, global_step=global_step)
    return train

###################################################################
# Accuracy
###################################################################

def accuracy(logits, labels):
    top_1 = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_mean(tf.cast(top_1, tf.float16))
