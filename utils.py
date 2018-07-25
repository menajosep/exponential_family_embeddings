import matplotlib.pyplot as plt
import os
import tensorflow as tf
import zipfile
import numpy as np
from itertools import chain

from six.moves import urllib
from pathos.multiprocessing import Pool, cpu_count
from more_itertools import chunked
from typing import List, Callable, Union, Any
import random
from math import ceil
import logging


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
    """Extract the first file enclosed in a zip file as a list of sentences"""
    data = list()
    with zipfile.ZipFile(filename) as z:
        with z.open(z.namelist()[0]) as f:
            for line in f:
                data.append(tf.compat.as_str(line))
    return data[:1000]


def flatten_list(listoflists):
    return list(chain.from_iterable(listoflists))


def process_sentences_constructor(neg_samples:int, dictionary:dict, context_size:int):
    """Generate a function that will clean and tokenize text."""
    def process_sentences(sentences):
        samples = []
        dictionary_keys = list(dictionary.keys())
        try:
            for sentence in sentences:
                padding = ['UNK'] * int(context_size/2)
                words = sentence.split()
                words = padding + words + padding
                if len(words) > context_size:
                    index = 0
                    for word in words[int(context_size/2):len(words)-int(context_size/2)]:
                        if word in dictionary_keys and word != 'UNK':
                            target_word_index = dictionary[word]
                            local_context_words_indexes = [i for i in range(index, index + context_size + 1)]
                            local_context_words_indexes.remove(index+int(context_size/2))
                            index += 1
                            for local_index in local_context_words_indexes:
                                context_word = words[local_index]
                                #prepare positive samples
                                if context_word in dictionary_keys:
                                    context_word_index = dictionary[context_word]
                                else:
                                    context_word_index = dictionary['UNK']
                                samples.append((target_word_index, context_word_index, 1))
                                for i in range(neg_samples):
                                    random_neg_sample = random.randint(0, len(dictionary) - 1)
                                    samples.append((target_word_index, random_neg_sample, 0))
        except Exception as e:
            print('error '+e)
        return samples

    return process_sentences


def apply_parallel(func: Callable,
                   data: List[Any],
                   cpu_cores: int = None) -> List[Any]:
    """
    Apply function to list of elements.

    Automatically determines the chunk size.
    """
    if not cpu_cores:
        cpu_cores = cpu_count()

    try:
        chunk_size = ceil(len(data) / cpu_cores)
        pool = Pool(cpu_cores)
        transformed_data = pool.map(func, chunked(data, chunk_size), chunksize=1)
    finally:
        pool.close()
        pool.join()
        return transformed_data


def get_optimal():
    x = np.linspace(0, 1, 100)
    a = 1
    optimals = []
    for b in range(7, 17, 2):
        optimal = np.power(x, a) * np.power((1 - x), b) + 1e-3
        optimal = optimal / np.sum(optimal)
        optimals.append(optimal)
    return optimals


def is_good_embedding(sigmas):
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