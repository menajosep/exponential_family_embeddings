import numpy as np
from random import shuffle
import random


class bayessian_bern_emb_data_deterministic():
    def __init__(self, logger, context_size, negative_samples, dimension, minibatch, repetitions, n_random_vectors=0):
        self.logger = logger
        self.cs = context_size
        self.ns = negative_samples
        self.K = dimension
        self.n_minibatch = minibatch
        self.repetitions = repetitions
        self.n_random_vectors = n_random_vectors

        self.dictionary = self.get_dictionary()
        self.L = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        self.embedding_matrix = self.get_embeddings()
        self.positive_word_sampling_indexes, self.negative_word_sampling_indexes = self.get_samples()
        logger.debug('bayessian_bern_emb_data_deterministic is built')

    def get_samples(self):
        positive_word_sampling_indexes = dict()
        negative_word_sampling_indexes = dict()
        positive_word_sampling_indexes[self.dictionary['pos']] = [self.dictionary['pos']] * self.cs
        positive_word_sampling_indexes[self.dictionary['neg']] = [self.dictionary['neg']] * self.cs
        negative_word_sampling_indexes[self.dictionary['pos']] = [self.dictionary['neg']] * self.cs
        negative_word_sampling_indexes[self.dictionary['neg']] = [self.dictionary['pos']] * self.cs
        return positive_word_sampling_indexes, negative_word_sampling_indexes

    def get_embeddings(self):
        embedding_matrix = np.zeros((self.L, self.K), dtype=np.float32)
        embedding_matrix[self.dictionary['pos']] = np.random.rand(self.K)
        embedding_matrix[self.dictionary['neg']] = self.get_ortogonal_vector(
            embedding_matrix[self.dictionary['pos']])
        for i in range(self.n_random_vectors):
            embedding_matrix[self.dictionary['random' + str(i + 1)]] = np.random.rand(self.K)
        return embedding_matrix

    def get_dictionary(self):
        dictionary = dict()
        dictionary['UNK'] = len(dictionary)
        dictionary['pos'] = len(dictionary)
        dictionary['neg'] = len(dictionary)
        for i in range(self.n_random_vectors):
            dictionary['random' + str(i + 1)] = len(dictionary)
        return dictionary

    def get_ortogonal_vector(self, v1):
        v2 = np.random.rand(self.K)
        coff = np.dot(v2, v1) / np.dot(v1, v1)
        # vector ortogonal a v1
        u2 = v2 - coff * v1
        return u2

    def batch_generator(self, batch_size, noise):
        epoch_samples = []
        for word in ['pos', 'neg']:
            for i in range(0, self.repetitions):
                if word != 'UNK':
                    noise_indexes = []
                    if noise > 0:
                        noise_indexes.extend(random.sample(range(0, self.cs), noise))
                    word_index = self.dictionary[word]
                    positive_word_sampling_indexes = self.positive_word_sampling_indexes[word_index]
                    negative_word_sampling_indexes = self.negative_word_sampling_indexes[word_index]
                    if len(positive_word_sampling_indexes) > 0:
                        pos_samples_indexes = []
                        while len(pos_samples_indexes) < self.cs:

                            pos_random_index = random.randint(0, len(positive_word_sampling_indexes) - 1)
                            pos_word = positive_word_sampling_indexes[pos_random_index]
                            if len(pos_samples_indexes) in noise_indexes:
                                pos_word = random.randint(3, self.n_random_vectors + 3 - 1)
                            pos_samples_indexes.append(pos_random_index)
                            epoch_samples.append((word_index, pos_word, 1))
                        neg_samples_indexes = []
                        while len(neg_samples_indexes) < self.ns:
                            neg_random_index = random.randint(0, len(negative_word_sampling_indexes)-1)
                            neg_word = negative_word_sampling_indexes[neg_random_index]
                            if len(pos_samples_indexes) in noise_indexes:
                                neg_word = random.randint(3, self.n_random_vectors + 3 - 1)
                            neg_samples_indexes.append(neg_random_index)
                            epoch_samples.append((word_index, neg_word, 0))
        shuffle(epoch_samples)
        target_words, context_words, labels = zip(*epoch_samples)
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
            words_target = data_target[:batch_size]
            words_context = data_context[:batch_size]
            labels = data_labels[:batch_size]
            data_target = data_target[batch_size:]
            data_context = data_context[batch_size:]
            data_labels = data_labels[batch_size:]
            yield words_target, words_context, labels

    def feed(self, target_placeholder, context_placeholder, labels_placeholder,
             ones_placeholder, zeros_placeholder, learning_rate_placeholder,
             n_minibatch, learning_rate):
        chars_target, chars_context, labels = self.batch.__next__()
        return {target_placeholder: chars_target,
                context_placeholder: chars_context,
                labels_placeholder: labels,
                ones_placeholder: np.ones(n_minibatch, dtype=np.int32),
                zeros_placeholder: np.zeros(n_minibatch, dtype=np.int32),
                learning_rate_placeholder: learning_rate
                }


class bayessian_bern_emb_data_deterministic_inverted(bayessian_bern_emb_data_deterministic):
    def get_samples(self):
        positive_word_sampling_indexes = dict()
        negative_word_sampling_indexes = dict()
        positive_word_sampling_indexes[self.dictionary['pos']] = [self.dictionary['neg']] * self.cs
        positive_word_sampling_indexes[self.dictionary['neg']] = [self.dictionary['pos']] * self.cs
        negative_word_sampling_indexes[self.dictionary['pos']] = [self.dictionary['pos']] * self.cs
        negative_word_sampling_indexes[self.dictionary['neg']] = [self.dictionary['neg']] * self.cs
        return positive_word_sampling_indexes, negative_word_sampling_indexes

