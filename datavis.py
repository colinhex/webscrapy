# Script to generate data visualizations, DEPRECATED
import argparse
import io
import sys
from typing import Callable, List, Tuple
from numbers import Number
from enum import Enum
import numpy
import skfuzzy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series

sns.set_theme(style="whitegrid")

features: list = [
       'Travel',
       'Social Networking and Messaging',
       'News',
       'Streaming Services',
       'Sports',
       'Photography',
       'Law and Government',
       'Health and Fitness',
       'Games',
       'E-Commerce',
       'Forums',
       'Food',
       'Education',
       'Computers and Technology',
       'Business/Corporate',
       'Adult'
]

columns = ["URL"]
columns.extend(features)
print(columns)

df = pd.read_csv("url_predictions.txt")
df.columns = columns


# helper functions

def fst_snd_max(xs: Series):
    """
    returns url, first maximum, second maximum tuple from observation.
    """""
    xss = sorted(xs[1::])
    return xs[0], xss[-1], xss[-2]


class AxesChoice(Enum):
    FST_SND_MAX = 1


def get_axes_meth(axes: AxesChoice) -> Callable[[Series], Tuple[str, Number, Number]]:
    match axes:
        case AxesChoice.FST_SND_MAX:
            return fst_snd_max


def extract_datapoints(df: pd.DataFrame, methodology: Callable[[Series], Tuple[str, Number, Number]]) \
        -> List[Tuple[str, Tuple[Number, Number]]]:
    xypts = []
    for data, x, y in [methodology(row) for index, row in df.iterrows()]:
        xypts.append((data, (x, y)))
    return xypts


def showcase_fuzzy_data_clusters(path, axes: AxesChoice):
    df = pd.read_csv(path)
    xypts = extract_datapoints(df, get_axes_meth(axes))
    data, pts = zip(*xypts)
    xpts, ypts = zip(*pts)
    xptsn = numpy.array(xpts)
    yptsn = numpy.array(ypts)

    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    all_data = np.vstack((xpts, ypts))
    fpcs = []

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(
            all_data, ncenters, 2, error=0.005, maxiter=1000, init=None)

        fpcs.append(fpc)

        cluster_membership = np.argmax(u, axis=0)
        for j in range(ncenters):
            ax.plot(xptsn[cluster_membership == j],
                    yptsn[cluster_membership == j], '.', color=colors[j])

        for pt in cntr:
            ax.plot(pt[0], pt[1], 'rs')

        ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
        ax.axis('off')

    fig1.tight_layout()

    plt.show()


def showcase_probability_urls(path):
    reader = io.open(path)
    colors = sns.color_palette('pastel')[0:5]
    while True:
        line = reader.readline()
        if not line:
            break
        predictions = line.split(',')[1::]
        plt.pie(predictions, labels=features, colors=colors, autopct='%.0f%%')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="url_texts.txt", help="path of cleaned website text sample.")
    parser.add_argument('-urlt', '--total_urls', type=int, default=1000, help="total number of urls read.")
    parser.add_argument('-m', '--mode', type=int, required=True, help="data visualization mode, 0 <=> cake diagramm of single urls, 1 <=> clustering.")

    args = parser.parse_args()

    match args.mode:
        case 0:
            showcase_probability_urls(args.input)
            sys.exit()
        case 1:
            showcase_fuzzy_data_clusters(args.input, AxesChoice.FST_SND_MAX)
            sys.exit()
