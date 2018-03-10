from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

import iris_data

def main():
    """Main function.

    This function orchestates the process of loading data,
    initializing the model, training the model, and evaluating the
    result.
    """
    args = parse_args()
    data = init_data(args)
    if args.model_version == 1:
        model = init_model(data, args)
    elif args.model_version == 2:
        model = init_model_2(data, args)
    else:
        raise AssertionError(args)
    train(model, data, args)
    evaluate(model, data, args)

def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-steps', default=1000, type=int)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--model-dir', default='model')
    parser.add_argument('--model-version', default=1, type=int, choices=[1, 2])
    return parser.parse_args()

def init_data(_args):
    """Initialize data for training and evaluation.
    """
    return iris_data.load_data()

def init_model(data, args):
    """Initialize the model for training.
    """
    train_x = data[0][0]
    feature_columns = [
        tf.feature_column.numeric_column(key=key)
        for key in train_x.keys()
    ]
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
        model_dir=args.model_dir)

def init_model_2(data, args):
    """Initialize mode 2 for training.
    """

    def model_fn(features, labels, mode, params):
        net = tf.feature_column.input_layer(features, params['feature_columns'])
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        logits = tf.layers.dense(net, params['n_classes'], activation=None)
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=predicted_classes,
            name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    train_x = data[0][0]
    feature_columns = [
        tf.feature_column.numeric_column(key=key)
        for key in train_x.keys()
    ]

    return tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        },
        model_dir=args.model_dir)

def train(model, data, args):
    """Train the model.
    """
    (train_x, train_y), _ = data
    input_fn = lambda: iris_data.train_input_fn(
        train_x, train_y, args.batch_size)
    model.train(input_fn=input_fn, steps=args.train_steps)

def evaluate(model, data, args):
    """Evaluate the trained model.
    """
    _, (test_x, test_y) = data
    input_fn = lambda: iris_data.eval_input_fn(
        test_x, test_y, args.batch_size)
    eval_result = model.evaluate(input_fn=input_fn)
    print('Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    main()
