import argparse
import collections
import logging
import os
import random
import re
import time
from itertools import chain
from math import ceil
from typing import List, Callable, Any

import dill
import numpy as np
import tensorflow as tf
from more_itertools import chunked
from pathos.multiprocessing import Pool, cpu_count
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tqdm import tqdm


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create logger
    logger = logging.getLogger("logging_MVP_word_embeddings_context_window")
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


def flatten_list(listoflists):
    return list(chain.from_iterable(listoflists))


def process_sentences_constructor(context_size: int):
    """Generate a function that will clean and tokenize text."""

    def process_sentences(data):
        samples = list()
        if context_size > 0:
            # we traverse all the words in the book
            for i in tqdm(range(context_size, len(data) - (context_size + 1))):
                # For the positive example we pick the letter and the letter after to get
                # the positive bigram
                word = data[i]
                prev_words = data[i - context_size:i]
                next_words = data[i + 1:i + context_size + 1]
                context_words = prev_words + next_words
                for context_word in context_words:
                    samples.append((word, context_word, 1))
            for i in tqdm(range(len(data) - 2)):
                # For the positive example we pick the letter and the letter after to get
                # the positive bigram
                word = data[i]
                next_word = data[i + 1]
                samples.append((word, next_word, 1))
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


def parallel_process_text(bbe_data: List[str], context_size) -> List[List[str]]:
    """Apply cleaner -> tokenizer."""
    process_text = process_sentences_constructor(context_size)
    return flatten_list(apply_parallel(process_text, bbe_data))


def count_bigrams(context_size, file_name, out_file_name):
    CONTEXT_SIZE = context_size  # @param {type:"slider", min:1, max:10, step:1}

    logger = get_logger()
    dir_name = 'fits/fit' + time.strftime("%y_%m_%d_%H_%M_%S")

    os.makedirs(dir_name)
    text_data = list()
    words = list()
    existing_bigrams = list()
    logger.info('Loading data from {}'.format(file_name))
    with open(file_name) as f:
        sentences = f.readlines()
    logger.info('Loaded {} lines'.format(len(sentences)))
    for sentence in sentences:
        sentence_words = re.split(r'\W+', sentence)
        for word in sentence_words:
            word = word.lower()
            if not word.isalpha():
                word = '#NUMBER'
            words.append(word)
    logger.info('Loaded {} words'.format(len(words)))
    count = [['UNK', 0]]
    count.extend(collections.Counter(words).most_common(1000 - 1))
    dictionary = dict()
    counter = dict()
    for character, _ in count:
        dictionary[character] = len(dictionary)
        counter[character] = _
    # we also create a dictionary that maps eahc id with the corresponing letter
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # we now create a vocabulary for our corpus, in this case the letters
    vocabulary = list(dictionary.keys())

    vocab_size = len(vocabulary)
    logger.info('Loaded dictionary of {} words'.format(vocab_size))
    unk_count = 0
    for word_index in range(len(words) - 1):
        if words[word_index] in dictionary:
            index = dictionary[words[word_index]]
        else:
            index = 0
            unk_count += 1
        text_data.append(index)
        next_word = words[word_index + 1]
        if next_word in dictionary:
            next_word_index = dictionary[next_word]
        else:
            next_word_index = 0
        existing_bigrams.append((index, next_word_index))
    text_data.append(next_word_index)
    count[0][1] = unk_count
    logger.info('{} OOV words'.format(unk_count))
    existing_bigrams = list(set(existing_bigrams))
    logger.info('{} existing bigrams'.format(len(existing_bigrams)))
    logger.info('counting bigrams')
    text_bigrams = parallel_process_text(text_data, CONTEXT_SIZE)
    logger.info('{} bigrams counted'.format(len(text_bigrams)))
    bigrams_counter = dict()
    for sample in text_bigrams:
        if sample in bigrams_counter:
            bigrams_counter[sample] += 1
        else:
            bigrams_counter[sample] = 1
    logger.info('counting text_next_word_bigrams')
    text_next_word_bigrams = parallel_process_text(text_data, 0)
    logger.info('{} text_next_word_bigrams counted'.format(len(text_next_word_bigrams)))
    next_word_bigrams_counter = dict()
    for sample in text_next_word_bigrams:
        if sample in bigrams_counter:
            next_word_bigrams_counter[sample] += 1
        else:
            next_word_bigrams_counter[sample] = 1
    logger.info('store bigrams_counter in {}'.format(os.path.join(dir_name, out_file_name)))
    with open(os.path.join(dir_name, out_file_name), "wb") as dill_file:
        dill.dump((count, bigrams_counter, next_word_bigrams_counter), dill_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run the clasifier")
    parser.add_argument('--context_size', type=int, default=0,
                        help='size of context window, use 0 for only taking the nex word into account')
    parser.add_argument('--file_name', type=str, default=None,
                        help='name of the data file')
    parser.add_argument('--out_file_name', type=str, default=None,
                        help='name of the out_file_name')

    args = parser.parse_args()

    count_bigrams(args.context_size, args.file_name, args.out_file_name)
