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
    model = init_model(data, args)
    train(model, data, args)
    evaluate(model, data, args)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-steps', default=1000, type=int,
        help='Training steps')
    parser.add_argument(
        '--batch-size', default=100, type=int,
        help='Training batch size')
    parser.add_argument(
        '--model-dir', default='model',
        help='Directory to save trained models')
    parser.add_argument(
        '--hidden-layers', default=2, type=int,
        help='Number of hidden layers')
    parser.add_argument(
        '--hidden-layer-nodes', default=10, type=int,
        help='Number of nodes per hidden layer')
    parser.add_argument(
        '--optimizer', default='Adagrad',
        help='Optimizer used for training')
    parser.add_argument(
        '--learning-rate', default=0.1, type=float,
        help='Learning rate')
    return parser.parse_args()

def init_data(_args):
    """Initialize data for training and evaluation."""
    return iris_data.load_data()

def init_model(data, args):
    """Initialize the model for training."""
    train_x = data[0][0]
    feature_columns = [
        tf.feature_column.numeric_column(key=key)
        for key in train_x.keys()
    ]
    return tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[args.hidden_layer_nodes] * args.hidden_layers,
        n_classes=3,
        optimizer=_init_optimizer(args),
        model_dir=args.model_dir)

def _init_optimizer(args):
    cls = getattr(tf.train, args.optimizer + 'Optimizer')
    return cls(learning_rate=args.learning_rate)

def train(model, data, args):
    """Train the model."""
    (train_x, train_y), _ = data
    input_fn = lambda: iris_data.train_input_fn(
        train_x, train_y, args.batch_size)
    model.train(input_fn=input_fn, steps=args.train_steps)

def evaluate(model, data, args):
    """Evaluate the trained model."""
    _, (test_x, test_y) = data
    input_fn = lambda: iris_data.eval_input_fn(
        test_x, test_y, args.batch_size)
    eval_result = model.evaluate(input_fn=input_fn)
    print('Test set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    main()
