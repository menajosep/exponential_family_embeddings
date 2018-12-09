import re
import os
import collections
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
import numpy as np
import time
from tqdm import tqdm
from pathos.multiprocessing import Pool, cpu_count
from more_itertools import chunked
from typing import List, Callable, Union, Any
import random
from math import ceil
from itertools import chain
import dill


NUM_EPOCHS = 50 #@param {type:"integer"}
LEARNING_RATE = 1e-1 #@param {type:"number"}
NEGATIVE_SAMPLES = 2 #@param {type:"slider", min:1, max:10, step:1}
EMBEDDING_DIM = 200 #@param {type:"integer"}
BATCH_SIZE = 1024 #@param {type:"number"}
CONTEXT_SIZE = 1 #@param {type:"slider", min:1, max:10, step:1}

bbe_data = list()
words = list()
existing_bigrams = list()
with open('/Users/jose.mena/dev/personal/data/basic_english/bbe') as f:
  sentences = f.readlines()
for sentence in sentences:
  sentence_words = re.split(r'\W+', sentence)
  for word in sentence_words:
    word = word.lower()
    if not word.isalpha():
      word = '#NUMBER'
    words.append(word)
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
vocabulary = dictionary.keys()

vocab_size = len(vocabulary)

unk_count = 0
for word_index in range(len(words)-1):
    if words[word_index] in dictionary:
        index = dictionary[words[word_index]]
    else:
        index = 0
        unk_count += 1
    bbe_data.append(index)
    next_word = words[word_index+1]
    if next_word in dictionary:
        next_word_index = dictionary[next_word]
    else:
        next_word_index = 0
    existing_bigrams.append((index,next_word_index))
bbe_data.append(next_word_index)
count[0][1] = unk_count
existing_bigrams = list(set(existing_bigrams))


def flatten_list(listoflists):
    return list(chain.from_iterable(listoflists))


def process_sentences_constructor(neg_samples: int, vocab_size: int, existing_bigrams: list, context_size: int):
    """Generate a function that will clean and tokenize text."""

    def process_sentences(data):
        samples = list()
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
            # for each positive example we take NEGATIVE_SAMPLES negative samples
            num_negs = 0
            while num_negs < NEGATIVE_SAMPLES:
                random_word_index = random.randint(0, vocab_size - 1)
                if (word, random_word_index) not in existing_bigrams:
                    num_negs += 1
                    samples.append((word, random_word_index, 0))
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


def parallel_process_text(bbe_data: List[str]) -> List[List[str]]:
    """Apply cleaner -> tokenizer."""
    process_text = process_sentences_constructor(
        NEGATIVE_SAMPLES, vocab_size, existing_bigrams, CONTEXT_SIZE)
    return flatten_list(apply_parallel(process_text, bbe_data))

bbe_samples = parallel_process_text(bbe_data)


class word_data():
    def __init__(self, samples):
        self.samples = samples

    def batch_generator(self, negative_samples, batch_size):
        target_words, context_words, labels = zip(*self.samples)
        labels = np.array(labels)
        word_target = np.array(target_words, dtype="int32")
        word_context = np.array(context_words, dtype="int32")
        data_target = word_target
        data_context = word_context
        data_labels = labels
        while True:
            if data_target.shape[0] < batch_size:
                data_target = np.hstack([data_target, word_target])
                data_context = np.hstack([data_context, word_context])
                data_labels = np.hstack([data_labels, labels])
                if data_target.shape[0] < batch_size:
                    continue
            word_target = data_target[:batch_size]
            word_context = data_context[:batch_size]
            labels = data_labels[:batch_size]
            data_target = data_target[batch_size:]
            data_context = data_context[batch_size:]
            data_labels = data_labels[batch_size:]
            yield word_target, word_context, labels

    def feed(self, target_placeholder, context_placeholder, labels_placeholder,
             learning_rate_placeholder, learning_rate_value):
        chars_target, chars_context, labels = self.batch.__next__()
        return {target_placeholder: chars_target,
                context_placeholder: chars_context,
                labels_placeholder: labels,
                learning_rate_placeholder: learning_rate_value
                }


class emb_model():
    def __init__(self, vocab_size, dim, graph, learning_rate, n_iters,
                 negative_samples, batch_size, logdir, embeddings=None):
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

        global_step = variable_scope.get_variable(  # this needs to be defined for tf.contrib.layers.optimize_loss()
            "global_step", [],
            trainable=False,
            dtype=dtypes.int64,
            initializer=init_ops.constant_initializer(0, dtype=dtypes.int64))

        # Data Placeholder
        with tf.name_scope('input'):
            self.target_placeholder = tf.placeholder(tf.int32)
            self.context_placeholder = tf.placeholder(tf.int32)
            self.labels_placeholder = tf.placeholder(tf.float32, shape=[batch_size])
            self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])

        self.rho_embeddings = tf.Variable(tf.random_uniform([vocab_size, dim], -1.0, 1.0), name='rho_embeddings')
        self.alpha_embeddings = tf.Variable(tf.random_uniform([vocab_size, dim], -1.0, 1.0), name='alpha_embeddings')
        self.rhos = tf.nn.embedding_lookup(self.rho_embeddings, self.target_placeholder)
        self.ctx_alpha = tf.nn.embedding_lookup(self.alpha_embeddings, self.context_placeholder)
        self.dot_product = tf.reduce_sum(tf.multiply(self.rhos, self.ctx_alpha), -1)
        rhos_norm = tf.norm(self.rhos, axis=1)
        ctx_alpha_norm = tf.norm(self.ctx_alpha, axis=1)
        denom = tf.multiply(rhos_norm, ctx_alpha_norm)
        self.cosine = tf.div(self.dot_product, denom)
        self.pos_prob = tf.sigmoid(3 * self.cosine)
        self.neg_prob = tf.sigmoid(-3 * self.cosine)
        variable_summaries(self.cosine, "cosine")
        variable_summaries(self.pos_prob, "pos_prob")
        variable_summaries(self.neg_prob, "neg_prob")
        self.log_likelihood = -(
                tf.reduce_sum(tf.multiply(self.labels_placeholder, tf.log(self.pos_prob))) +
                tf.reduce_sum(tf.multiply((1 - self.labels_placeholder), tf.log(self.neg_prob)))
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
        self.train_op = tf.contrib.layers.optimize_loss(
            self.log_likelihood, global_step,
            learning_rate=self.learning_rate_placeholder,
            optimizer='Adam',
            summaries=["gradients"])

        tf.summary.scalar("loss", self.log_likelihood)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(logdir, graph.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        rho = config.embeddings.add()
        rho.tensor_name = 'rho_embeddings'
        rho.metadata_path = 'vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)
        with graph.as_default():
            tf.global_variables_initializer().run()

dir_name = 'fits/fit' + time.strftime("%y_%m_%d_%H_%M_%S")
os.makedirs(dir_name)

def get_n_batches_per_epoch(ns, n_epochs, batch_size, data_size):
  n_batches = (data_size*(int(ns) +1)) / batch_size
  if data_size % batch_size > 0:
      n_batches += 1
  return int(n_batches)

def get_learning_rates(initial_learning_rate, num_batches, num_epochs):
  learning_rates = []
  learning_rate = initial_learning_rate
  decay_factor = 1e-1
  for epoch in range(num_epochs):
    learning_rates.extend([learning_rate]*num_batches)
    if epoch in [0,1]:
      learning_rate *= decay_factor
  return learning_rates

book_data = word_data(bbe_samples)
book_data.batch = book_data.batch_generator(NEGATIVE_SAMPLES, BATCH_SIZE)
n_batches = get_n_batches_per_epoch(NEGATIVE_SAMPLES, NUM_EPOCHS, BATCH_SIZE, len(bbe_samples))
num_iters = n_batches * NUM_EPOCHS
learning_rates = get_learning_rates(LEARNING_RATE, n_batches, NUM_EPOCHS)
g1 = tf.Graph()
with g1.as_default() as g:
  with tf.Session( graph = g ) as sess:
    m = emb_model(vocab_size, EMBEDDING_DIM, sess,
                  learning_rates, num_iters,
                  NEGATIVE_SAMPLES, BATCH_SIZE, dir_name)
    init = tf.global_variables_initializer()
    sess.run(init)
    iteration = 0
    for epoch in range(NUM_EPOCHS):
      print('epoch {} of {}'.format(epoch+1, NUM_EPOCHS))
      #for batch in tqdm(range(n_batches), desc='epoch {} of {}'.format(epoch, num_epochs)):
      for batch in range(n_batches):
        _, log_likelihood, rho_embeddings, alpha_embeddings, summaries = sess.run(
              [m.train_op, m.log_likelihood, m.rho_embeddings, m.alpha_embeddings, m.summaries],
               book_data.feed(
                   m.target_placeholder,
                   m.context_placeholder,
                   m.labels_placeholder,
                   m.learning_rate_placeholder,
                   learning_rates[iteration]
               )
        )
        m.train_writer.add_summary(summaries, iteration)
        iteration += 1
      m.saver.save(sess, os.path.join(dir_name, "model.ckpt"), iteration)

with open(os.path.join(dir_name, "results.db"), "wb") as dill_file:
    dill.dump((dictionary, counter, vocabulary, existing_bigrams, rho_embeddings, alpha_embeddings), dill_file)

