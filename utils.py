import matplotlib.pyplot as plt
import os
import tensorflow as tf
import zipfile
import numpy as np

from six.moves import urllib


def maybe_download(url, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def get_optimal():
    x = np.linspace(0, 1, 100)
    a = 1
    optimals = []
    for b in range(7, 17, 2):
        optimal = np.power(x, a) * np.power((1 - x), b) + 1e-3
        optimal = optimal / np.sum(optimal)
        optimals.append(optimal)
    return optimals


def is_goog_embedding(sigmas):
    threshold = 1e-3
    optimals = get_optimal()
    hist = plt.hist(sigmas, bins=100, color='green', label='sigma values')
    distr = (hist[0] + 1e-5) / np.sum(hist[0])
    distance = 0
    for optimal in optimals:
        distance += -np.sum(optimal * np.log(distr / optimal))
    distance = distance / len(optimals)
    return distance < threshold


def plot_with_labels(low_dim_embs, labels, fname):
    plt.figure(figsize=(28, 28))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(fname)
    plt.close()


def variable_summaries(summary_name, var):
    with tf.name_scope(summary_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

