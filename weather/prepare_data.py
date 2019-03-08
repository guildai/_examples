import argparse
import datetime
import os

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def main():
    args = init_args()
    data = load_data(args)
    ensure_output_dir(args)
    write_data_desc(data, args)
    write_data_figs(data, args)
    write_split_data(data, args)

def init_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--source',
        default='seattle-2016.csv',
        help="Path to CSV containing weather data")
    p.add_argument(
        '--val-split',
        type=float,
        default=0.25,
        help="Fraction of data to reserve for validation")
    p.add_argument(
        '--random-seed',
        type=int,
        help="Random seed used for validation split")
    p.add_argument(
        '--output',
        default='data',
        help="Path to directory containing prepared data")
    return p.parse_args()

def load_data(args):
    print("Loading data")
    return pd.read_csv(args.source)

def ensure_output_dir(args):
    if not os.path.exists(args.output):
        print("Creating output directory %s" % args.output)
        os.makedirs(args.output)

def write_data_desc(data, args):
    print("Writing data description")
    data_desc = data.describe().to_string()
    with open(os.path.join(args.output, 'data-summary.txt'), 'w') as out:
        out.write(data_desc)

def write_data_figs(data, args):
    write_temps_fig(data, args)
    write_pairplots_fig(data, args)

def write_temps_fig(data, args):
    print("Writing temps figure")

    # Format dates
    years = data['year']
    months = data['month']
    days = data['day']
    dates = [
        str(int(year)) + '-' + str(int(month)) + '-' + str(int(day))
        for year, month, day in zip(years, months, days)
    ]
    dates = [
        datetime.datetime.strptime(date, '%Y-%m-%d')
        for date in dates
    ]

    plt.style.use('fivethirtyeight')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, figsize=(15, 10))
    fig.autofmt_xdate(rotation=45)

    # Actual max temperature measurement
    ax1.plot(dates, data['actual'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Temperature (F)')
    ax1.set_title('Max Temp')

    # Temperature from 1 day ago
    ax2.plot(dates, data['temp_1'])
    ax2.set_xlabel('')
    ax2.set_ylabel('Temperature (F)')
    ax2.set_title('Prior Max Temp')

    # Temperature from 2 days ago
    ax3.plot(dates, data['temp_2'])
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Temperature (F)')
    ax3.set_title('Two Days Prior Max Temp')

    # Friend Estimate
    ax4.plot(dates, data['friend'])
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Temperature (F)')
    ax4.set_title('Friend Estimate')

    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(args.output, 'temps.png'))

def write_pairplots_fig(data, args):
    print("Writing pairplots figure")
    try:
        import seaborn as sns
    except ImportError:
        print(
            "Cannot write pairplots (install seaborn package "
            "to enable this functionality)")
        return

    # Create columns of seasons for pair plotting colors
    seasons = []
    for month in data['month']:
        if month in [1, 2, 12]:
            seasons.append('winter')
        elif month in [3, 4, 5]:
            seasons.append('spring')
        elif month in [6, 7, 8]:
            seasons.append('summer')
        elif month in [9, 10, 11]:
            seasons.append('fall')

    # Plot three cols with seasons
    to_plot = data[['temp_1', 'average', 'actual']].copy()
    to_plot['season'] = seasons

    sns.set(style="ticks", color_codes=True)
    palette = sns.xkcd_palette(['dark blue', 'dark green', 'gold', 'orange'])
    sns_plt = sns.pairplot(
        to_plot,
        hue='season',
        diag_kind='kde',
        palette=palette,
        plot_kws={'alpha': 0.7},
        diag_kws={'shade': True})
    sns_plt.savefig(os.path.join(args.output, "pairplots.png"))

def write_split_data(data, args):
    print("Writing train and validate data")
    (train_features,
     val_features,
     train_labels,
     val_labels), feature_names = split_data(data, args)
    np.save(os.path.join(args.output, "train-features.npy"), train_features)
    np.save(os.path.join(args.output, "val-features.npy"), val_features)
    np.save(os.path.join(args.output, "train-labels.npy"), train_labels)
    np.save(os.path.join(args.output, "val-labels.npy"), val_labels)
    with open(os.path.join(args.output, "feature-names.txt"), "w") as out:
        out.write("\n".join(feature_names))

def split_data(data, args):
    onehot_data = pd.get_dummies(data)
    labels = data['actual']
    features = onehot_data.drop('actual', axis=1)
    split = train_test_split(
         np.array(features),
         np.array(labels),
         test_size=args.val_split,
         random_state=args.random_seed)
    return split, list(features.columns)

if __name__ == "__main__":
    main()
