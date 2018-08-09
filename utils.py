import matplotlib.pyplot as plt
import os
import tensorflow as tf
import zipfile
import numpy as np
import logging

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


def is_good_embedding(sigmas):
    threshold = 1e-3
    distance = get_distance(sigmas)
    return distance < threshold


def get_distance(sigma_array):
    n_bins = 100
    min_value = 0
    max_value = 1.5
    width = max_value / n_bins
    current_bin = min_value
    bins = [current_bin]
    for i in range(n_bins):
        current_bin += width
        bins.append(current_bin)
    x = np.linspace(0,1,100)
    a=1
    m1=plt.hist(sigma_array,bins)
    d1=(m1[0]+1e-5)/np.sum(m1[0])
    distance = []
    for b in range(7,17,2):
        optimal = np.power(x,a)*np.power((1-x),b) + 1e-3
        optimal=optimal/np.sum(optimal)
        plt.plot(x,optimal)
        distance.append(-np.sum(optimal*np.log(d1/optimal)))
    return np.mean(distance)


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


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create logger
    logger = logging.getLogger("logging_songscuncert")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger


def get_n_iters(n_epochs, batch_size, n_samples):
    n_batches = n_samples / batch_size
    if n_samples % batch_size > 0:
        n_batches += 1
    return int(n_batches) * n_epochs, int(n_batches)


def get_kl_weights(n_batches):
    weights = np.full(n_batches, 1./((2 ** 1000) - 1), dtype=np.float64)
    weights_lim = min(n_batches, 1000)
    for i in range(weights_lim):
        weight = (2 ** (weights_lim - i)) / ((2 ** n_batches) - 1)
        weights[i] = weight
    return weights


def clr(clr_iteration, clr_type, step_size, base_lr, max_lr):
    if clr_type == 'triangular':
        scale_fn = lambda x: 1.
    elif clr_type == 'triangular2':
        scale_fn = lambda x: 1 / (2. ** (x - 1))
    elif clr_type == 'exp_range':
        gamma = 1.
        scale_fn = lambda x: gamma ** (x)
    else:
        scale_fn = lambda x: x

    cycle = np.floor(1 + (1. * clr_iteration) / (2 * step_size))
    x = np.abs(1. * clr_iteration / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * scale_fn(cycle)


def get_learning_rates(clr_type, n_iters, n_cycles, base_lr, max_lr, default_lr):
    lrs = []
    for i in range(n_iters):
        if clr_type is not None:
            step_size = n_iters / (2 * n_cycles)
            lrs.append(clr(i, clr_type, step_size, base_lr, max_lr))
        else:
            lrs.append(default_lr)
    return lrs


class EarlyStopping:
    def __init__(self,
                 monitor='distance',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None):

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def is_early_stopping(self, value):
        stop_training = False
        if self.monitor_op(value - self.min_delta, self.best):
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                stop_training = True
        return stop_training