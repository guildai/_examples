import argparse
import os
import pickle
import warnings

# Silence the ever-present numpy deprecation warnings
warnings.warn = lambda *_args, **_kw: None

import numpy as np

def main():
    args = init_args()
    data = load_data(args)
    model = init_model(args)
    train(model, data)
    ensure_output_dir(args)
    write_model(model, args)
    eval_model(model, data)

def init_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--estimators',
        default=1000,
        type=int,
        help="Number of trees in the forest")
    p.add_argument(
        '--criterion',
        default='mse',
        choices=('mse', 'mae'),
        help="Function to measure the quality of a split")
    p.add_argument("--max_depth", type=int)
    p.add_argument("--min_samples_split", default=2, type=int)
    p.add_argument("--min_samples_leaf", default=1, type=int)
    p.add_argument("--min_weight_fraction_leaf", default=0.0, type=float)
    p.add_argument("--max_features", default="auto")
    p.add_argument("--max_leaf_nodes", type=int)
    p.add_argument("--min_impurity_decrease", default=0.0, type=float)
    p.add_argument(
        '--random_seed',
        type=int,
        help="Random seed used for model init")
    p.add_argument(
        '--data_dir',
        default='data',
        help="Path to prepared data")
    p.add_argument(
        '--output',
        default='model',
        help="Path to directory containing saved model")
    return p.parse_args()

def load_data(args):
    print("Loading data")
    train_features = np.load(os.path.join(args.data_dir, 'train-features.npy'))
    val_features = np.load(os.path.join(args.data_dir, 'val-features.npy'))
    train_labels = np.load(os.path.join(args.data_dir, 'train-labels.npy'))
    val_labels = np.load(os.path.join(args.data_dir, 'val-labels.npy'))
    return (
        train_features,
        val_features,
        train_labels,
        val_labels,
    )

def init_model(args):
    print("Initializing model")
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=args.estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        min_weight_fraction_leaf=args.min_weight_fraction_leaf,
        max_features=args.max_features,
        max_leaf_nodes=args.max_leaf_nodes,
        min_impurity_decrease=args.min_impurity_decrease,
        random_state=args.random_seed)

def train(model, data):
    print("Training model")
    train_features, _val_f, train_labels, _val_l = data
    model.fit(train_features, train_labels)

def ensure_output_dir(args):
    if not os.path.exists(args.output):
        print("Creating output directory %s" % args.output)
        os.makedirs(args.output)

def write_model(model, args):
    print("Saving model")
    with open(os.path.join(args.output, 'model.pickle'), 'wb') as out:
        out.write(pickle.dumps(model))

def eval_model(model, data):
    print("Evaluating model")
    train_features, val_features, train_labels, val_labels = data
    eval_predictions(model, train_features, train_labels, "train")
    eval_predictions(model, val_features, val_labels, "validate")

def eval_predictions(model, features, labels, desc):
    predictions = model.predict(features)
    errors = abs(predictions - labels)
    print("  Average absolute error (%s): %f degrees" % (desc, np.mean(errors)))
    accuracy = 1.0 - np.mean(errors / labels)
    print("  Accuracy (%s): %f" % (desc, accuracy))

if __name__ == '__main__':
    main()
