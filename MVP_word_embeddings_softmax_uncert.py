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
from tensorflow.contrib import distributions
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
        for i in tqdm(range(context_size, len(data) - (context_size + 1))):
            # For the positive example we pick the letter and the letter after to get
            # the positive bigram
            word = data[i]
            prev_words = data[i - context_size:i]
            next_words = data[i + 1:i + context_size + 1]
            context_words = prev_words + next_words
            for context_word in context_words:
                samples.append((word, context_word))
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


class WordData:
    def __init__(self, samples):
        self.samples = samples

    def batch_generator(self, batch_size):
        target_words, context_words = zip(*self.samples)
        word_target = np.array(target_words, dtype="int32")
        word_context = np.array(context_words, dtype="int32")
        data_target = word_target
        data_context = word_context
        while True:
            if data_target.shape[0] < batch_size:
                data_target = np.hstack([data_target, word_target])
                data_context = np.hstack([data_context, word_context])
                if data_target.shape[0] < batch_size:
                    continue
            word_target = data_target[:batch_size]
            word_context = data_context[:batch_size]
            data_target = data_target[batch_size:]
            data_context = data_context[batch_size:]
            yield word_target, word_context

    def feed(self, target_placeholder, context_placeholder, learning_rate_placeholder, learning_rate_value):
        chars_target, chars_context = self.batch.__next__()
        return {target_placeholder: chars_target,
                context_placeholder: chars_context,
                learning_rate_placeholder: learning_rate_value
                }


class EmbModel:
    def __init__(self, vocab_size, dim, graph, batch_size, logdir):
        T = 10

        def variable_summaries(var, name):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries_' + name):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        # for a single monte carlo simulation,
        #   calculate categorical_crossentropy of
        #   predicted logit values plus gaussian
        #   noise vs true values.
        # true - true values. Shape: (N, C)
        # pred - predicted logit values. Shape: (N, C)
        # dist - normal distribution to sample from. Shape: (N, C)
        # undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
        # num_classes - the number of classes. C
        # returns - total differences for all classes (N,)
        def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
            def map_fn(i):
                std_samples = tf.transpose(dist.sample(num_classes))
                std_samples = tf.reshape(std_samples, [-1, 1, num_classes])
                distorted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=true,
                    logits=pred + std_samples,
                    name="sparse_softmax_cross_entropy_with_logits"
                )
                distorted_loss = tf.cast(distorted_loss, tf.float64)
                diff = undistorted_loss - distorted_loss
                return -tf.nn.elu(diff)

            return map_fn

        global_step = variable_scope.get_variable(  # this needs to be defined for tf.contrib.layers.optimize_loss()
            "global_step", [],
            trainable=False,
            dtype=dtypes.int64,
            initializer=init_ops.constant_initializer(0, dtype=dtypes.int64))

        # Data Placeholder
        with tf.name_scope('input'):
            self.target_placeholder = tf.placeholder(tf.int32)
            self.context_placeholder = tf.placeholder(tf.int32)
            self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])

        self.rho_embeddings = tf.Variable(tf.random_uniform([vocab_size, dim], -1.0, 1.0), name='rho_embeddings')
        self.alpha_embeddings = tf.Variable(tf.random_uniform([vocab_size, dim], -1.0, 1.0), name='alpha_embeddings')

        self.var_embeddings = tf.Variable(tf.ones([vocab_size, 1]), name='var_embeddings')

        h_target = tf.nn.embedding_lookup(self.rho_embeddings, self.target_placeholder)
        self.variance = tf.layers.dense(
            tf.reshape(h_target, [-1, dim]),
            1,
            name="variance",
            use_bias=False,
            activation=tf.nn.softplus)
        h_target = tf.reshape(h_target, [batch_size, 1, dim])
        logits = tf.matmul(h_target, [tf.transpose(self.alpha_embeddings)] * batch_size)

        self.std = tf.sqrt(self.variance)
        variance_depressor = tf.exp(self.variance) - tf.ones_like(self.variance)
        variance_depressor = tf.cast(variance_depressor, tf.float64)

        probs = tf.nn.softmax(logits)
        variable_summaries(logits, "logits")
        variable_summaries(probs, "probs")
        variable_summaries(self.variance, "variance")

        self.categorical_crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.context_placeholder, [batch_size, 1]),
            logits=logits,
            name="sparse_softmax_cross_entropy_with_logits"
        )
        self.categorical_crossentropy = tf.cast(self.categorical_crossentropy, tf.float64)

        # shape: (T,)
        iterable = tf.Variable(np.ones(T))
        dist = distributions.Normal(loc=tf.zeros_like(self.std), scale=self.std)
        monte_carlo_results = tf.map_fn(
            gaussian_categorical_crossentropy(
                tf.reshape(self.context_placeholder, [batch_size, 1]),
                logits,
                dist,
                self.categorical_crossentropy,
                vocab_size
            ),
            iterable, name='monte_carlo_results'
        )
        monte_carlo_results_mean = tf.reduce_mean(monte_carlo_results, axis=0)

        variance_loss = tf.multiply(
            monte_carlo_results_mean,
            self.categorical_crossentropy
        )

        self.bayesian_categorical_crossentropy = (
                variance_loss + self.categorical_crossentropy + variance_depressor
        )
        categorical_crossentropy_weight = 1.
        bayesian_categorical_crossentropy_weight = .2
        self.loss = tf.reduce_mean(
            categorical_crossentropy_weight * self.categorical_crossentropy +
            bayesian_categorical_crossentropy_weight * self.bayesian_categorical_crossentropy
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, global_step,
            learning_rate=self.learning_rate_placeholder,
            optimizer='Adam',
            summaries=["gradients"])

        variable_summaries(self.categorical_crossentropy, "categorical_crossentropy")
        tf.summary.scalar("loss", self.loss)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(logdir, graph.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        rho = config.embeddings.add()
        rho.tensor_name = 'rho_embeddings'
        rho.metadata_path = 'vocab.tsv'
        alpha = config.embeddings.add()
        alpha.tensor_name = 'alpha_embeddings'
        alpha.metadata_path = 'vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)
        with graph.as_default():
            tf.global_variables_initializer().run()


def get_n_batches_per_epoch(batch_size, data_size):
    n_batches = data_size / batch_size
    if data_size % batch_size > 0:
        n_batches += 1
    return int(n_batches)


def get_learning_rates(initial_learning_rate, num_batches, num_epochs):
    learning_rates = []
    learning_rate = initial_learning_rate
    decay_factor = 1#1e-1
    for epoch in range(num_epochs):
        learning_rates.extend([learning_rate] * num_batches)
        if epoch in [0, 1]:
            learning_rate *= decay_factor
    return learning_rates


def learn_embeddings(context_size, num_epochs,
                     learning_rate,embedding_dim, batch_size, file_name):
    NUM_EPOCHS = num_epochs  # @param {type:"integer"}
    LEARNING_RATE = learning_rate  # @param {type:"number"}
    EMBEDDING_DIM = embedding_dim  # @param {type:"integer"}
    BATCH_SIZE = batch_size  # @param {type:"number"}
    CONTEXT_SIZE = context_size  # @param {type:"slider", min:1, max:10, step:1}

    logger = get_logger()
    dir_name = 'fits/fit' + time.strftime("%y_%m_%d_%H_%M_%S")

    os.makedirs(dir_name)
    bbe_data = list()
    words = list()
    existing_bigrams = list()
    logger.info('Loading data from {}'.format(file_name))
    with open(file_name) as f:
        sentences = f.readlines()
    logger.info('Loaded {} lines'.format(len(sentences)))
    for sentence in sentences:
        sentence_words = re.split(r'\W+', sentence)
        for word in sentence_words[3:]:
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
    # we now create a vocabulary for our corpus, in this case the letters
    vocabulary = list(dictionary.keys())

    vocab_size = len(vocabulary)
    logger.info('store vocab in {}'.format(os.path.join(dir_name, "vocab.tsv")))
    with open(os.path.join(dir_name, "vocab.tsv"), 'w') as txt:
        for char in vocabulary:
            txt.write(char + '\n')
    logger.info('Loaded dictionary of {} words'.format(vocab_size))
    unk_count = 0
    for word_index in range(len(words) - 1):
        if words[word_index] in dictionary:
            index = dictionary[words[word_index]]
        else:
            index = 0
            unk_count += 1
        bbe_data.append(index)
        next_word = words[word_index + 1]
        if next_word in dictionary:
            next_word_index = dictionary[next_word]
        else:
            next_word_index = 0
        existing_bigrams.append((index, next_word_index))
    bbe_data.append(next_word_index)
    count[0][1] = unk_count
    logger.info('{} OOV words'.format(unk_count))
    unique_existing_bigrams = list(set(existing_bigrams))
    logger.info('{} existing bigrams'.format(len(existing_bigrams)))
    logger.info('{} unique existing bigrams'.format(len(unique_existing_bigrams)))
    logger.info('Generating samples')
    if CONTEXT_SIZE > 0:
        bbe_samples = parallel_process_text(bbe_data, CONTEXT_SIZE)
    else:
        bbe_samples = existing_bigrams
    logger.info('{} samples generated'.format(len(bbe_samples)))

    book_data = WordData(bbe_samples)
    book_data.batch = book_data.batch_generator(BATCH_SIZE)
    n_batches = get_n_batches_per_epoch(BATCH_SIZE, len(bbe_samples))
    num_iters = n_batches * NUM_EPOCHS
    learning_rates = get_learning_rates(LEARNING_RATE, n_batches, NUM_EPOCHS)
    logger.info('training for {} epochs, {} batch per epoch, total of iters {}'.format(NUM_EPOCHS, n_batches, num_iters))
    g1 = tf.Graph()
    with g1.as_default() as g:
        with tf.Session(graph=g) as sess:
            m = EmbModel(vocab_size, EMBEDDING_DIM, sess, BATCH_SIZE, dir_name)
            init = tf.global_variables_initializer()
            sess.run(init)
            iteration = 0
            for epoch in range(NUM_EPOCHS):
                logger.info('epoch {} of {}'.format(epoch + 1, NUM_EPOCHS))
                for batch in tqdm(range(n_batches), desc='epoch {} of {}'.format(epoch, num_epochs)):
                #for batch in range(n_batches):
                    _, loss, rho_embeddings, alpha_embeddings, summaries = sess.run(
                        [m.train_op, m.loss, m.rho_embeddings, m.alpha_embeddings, m.summaries],
                        book_data.feed(
                            m.target_placeholder,
                            m.context_placeholder,
                            m.learning_rate_placeholder,
                            learning_rates[iteration]
                        )
                    )
                    m.train_writer.add_summary(summaries, iteration)
                    iteration += 1
                m.saver.save(sess, os.path.join(dir_name, "model.ckpt"), iteration)

    logger.info('store results in {}'.format(os.path.join(dir_name, "results.db")))
    with open(os.path.join(dir_name, "results.db"), "wb") as dill_file:
        dill.dump((dictionary, counter, vocabulary,
                   unique_existing_bigrams, rho_embeddings, alpha_embeddings), dill_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run the clasifier")
    parser.add_argument('--context_size', type=int, default=0,
                        help='size of context window, use 0 for only taking the nex word into account')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                        help='initial learning rate')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='number of dimensions for vectors')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='size of the batches')
    parser.add_argument('--file_name', type=str, default=None,
                        help='name of the data file')

    args = parser.parse_args()

    learn_embeddings(args.context_size, args.num_epochs,
                     args.learning_rate, args.embedding_dim, args.batch_size, args.file_name)
