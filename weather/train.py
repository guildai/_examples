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
    evaluate_model(model, data)

def init_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--estimators",
        default=1000,
        type=int,
        help="Number of trees in the forest")
    p.add_argument(
        '--random-seed',
        type=int,
        help="Random seed used for model init")
    p.add_argument(
        "--data-dir",
        default="data",
        help="Path to prepared data")
    p.add_argument(
        '--output',
        default='model',
        help="Path to directory containing saved model")
    return p.parse_args()

def load_data(args):
    print("Loading data")
    train_features = np.load(os.path.join(args.data_dir, "train-features.npy"))
    val_features = np.load(os.path.join(args.data_dir, "val-features.npy"))
    train_labels = np.load(os.path.join(args.data_dir, "train-labels.npy"))
    val_labels = np.load(os.path.join(args.data_dir, "val-labels.npy"))
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
    with open(os.path.join(args.output, "model.pickle"), "wb") as out:
        out.write(pickle.dumps(model))

def evaluate_model(model, data):
    print("Evaluating model")
    _train_f, val_features, _train_l, val_labels = data
    predictions = model.predict(val_features)
    errors = abs(predictions - val_labels)
    print("  Average absolute error: %f degrees" % np.mean(errors))
    accuracy = 1.0 - np.mean(errors / val_labels)
    print("  Accuracy: %f" % accuracy)

if __name__ == "__main__":
    main()
