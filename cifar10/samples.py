import argparse
import json

import tensorflow as tf

from tensorflow.core.framework import summary_pb2

import support

FLAGS = None

def prepare_samples(images_op, labels_op):
    # Use summary op to generate a PNG from TF native image
    inputs = tf.placeholder(tf.float32, [None, support.CROPPED_IMAGE_HEIGHT,
                                         support.CROPPED_IMAGE_WIDTH,
                                         support.IMAGE_DEPTH])
    summary = tf.image_summary('input', inputs, 1)

    # Init session
    sess = tf.Session()

    # Initialize queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Save images and labels
    images, labels = sess.run([images_op, labels_op])
    i = 1
    for image, label in zip(images, labels):
        summary_bin = sess.run(summary, feed_dict={inputs: [image]})
        image_summary = summary_pb2.Summary()
        image_summary.ParseFromString(summary_bin)
        basename = FLAGS.sample_dir + "/" + ("%05i" % i)
        image_path = basename + ".png"
        print "Writing %s" % image_path
        with open(image_path, "w") as f:
            f.write(image_summary.value[0].image.encoded_image_string)
        with open(basename + ".json", "w") as f:
            f.write(json.dumps({
                "image": image.tolist(),
                "label": int(label)
            }))
        i += 1

    # Stop queue runners
    coord.request_stop()
    coord.join(threads)

def main(_):
    images, labels = support.data_inputs(
        FLAGS.datadir, support.VALIDATION_DATA, FLAGS.sample_count, 1)
    tf.gfile.MakeDirs(FLAGS.sample_dir)
    prepare_samples(images, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/tmp/CIFAR10_data",)
    parser.add_argument("--sample_dir", default="/tmp/CIFAR10_samples")
    parser.add_argument("--sample_count", type=int, default=100)
    FLAGS, _ = parser.parse_known_args()
    tf.app.run()
