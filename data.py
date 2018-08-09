import numpy as np
from utils import *
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams
import collections
from gensim.models import KeyedVectors
import pickle

class bern_emb_data():
    def __init__(self, cs, ns, n_minibatch, L):
        assert cs%2 == 0
        self.cs = cs
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        #url = 'http://mattmahoney.net/dc/'
        #filename = maybe_download(url, 'text8.zip', 31344016)
        #filename = '/Users/jose.mena/Dropbox/Jose/PhD/data/recipes.txt.zip'
        filename = '../data/wiki/wiki.txt.zip'
        words = read_data(filename)
        self.build_dataset(words)
        self.batch = self.batch_generator()
        self.N = len(self.data)

    def build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.L - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
              index = dictionary[word]
            else:
              index = 0 
              unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.data = np.array(data)
        self.count = count
        self.dictionary = dictionary
        self.words = [reverse_dictionary[x] for x in range(len(reverse_dictionary))]
        sampling_table = sequence.make_sampling_table(len(dictionary))
        couples, labels = skipgrams(data,
                                    len(dictionary),
                                    window_size=self.cs,
                                    sampling_table=sampling_table,
                                    negative_samples=self.ns)
        del data
        self.labels = np.array(labels)
        # labels[labels == 0] = -1
        word_target, word_context = zip(*couples)
        del couples
        self.word_target = np.array(word_target, dtype="int32")
        self.word_context = np.array(word_context, dtype="int32")
        with open('fits/vocab.tsv', 'w') as txt:
            for word in self.words:
                txt.write(word+'\n')

    def batch_generator(self):
        batch_size = self.n_minibatch
        data_target = self.word_target
        data_context = self.word_context
        data_labels = self.labels
        while True:
            if data_target.shape[0] < batch_size:
                data_target = np.hstack([data_target, self.word_target])
                data_context = np.hstack([data_context, self.word_context])
                data_labels = np.hstack([data_labels, self.labels])
                if data_target.shape[0] < batch_size:
                    continue
            words_target = data_target[:batch_size]
            words_context = data_context[:batch_size]
            labels = data_labels[:batch_size]
            data_target = data_target[batch_size:]
            data_context = data_context[batch_size:]
            data_labels = data_labels[batch_size:]
            yield words_target, words_context, labels
    
    def feed(self, target_placeholder, context_placeholder, labels_placeholder, y_ph):
        words_target, words_context, labels = self.batch.next()
        return {target_placeholder: words_target,
                context_placeholder: words_context,
                labels_placeholder: labels,
                y_ph: np.ones((self.n_minibatch), dtype=np.int32)
                }


class bayessian_bern_emb_data():
    def __init__(self, input_file, cs, ns, n_minibatch, L, K,
                 emb_type, word2vec_file, glove_file,
                 fasttext_file, custom_file, dir_name, logger):
        self.logger = logger
        self.logger.debug('initializing bayessian_bern_emb_data with file '+input_file)
        self.logger.debug('neg sampling '+str(ns))
        self.logger.debug('context size of '+str(cs))
        self.logger.debug('working dir '+dir_name)
        assert cs % 2 == 0
        self.cs = cs
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        self.K = K
        self.dir_name = dir_name
        self.word2vec_embedings = None
        self.glove_embedings = None
        self.fasttext_embedings = None
        self.custom_embedings = None
        self.emb_type = emb_type
        self.embedding_matrix = None
        self.logger.debug('....reading data')
        words = read_data(input_file)
        self.logger.debug('....loading embeddings file')
        if emb_type:
            self.word2vec_embedings = self.read_word2vec_embeddings(word2vec_file)
            self.glove_embedings = self.read_embeddings(glove_file)
            self.fasttext_embedings = self.read_embeddings(fasttext_file)
            if custom_file:
                self.custom_embedings = self.read_embeddings(custom_file)
        self.logger.debug('....building corpus')
        self.build_dataset(words)
        self.N = len(self.word_target)
        pickle.dump(self.dictionary, open(dir_name + "/dictionary.dat", "wb+"))
        pickle.dump(self.words, open(dir_name + "/words.dat", "wb+"))
        pickle.dump(self.counter, open(dir_name + "/counter.dat", "wb+"))
        pickle.dump(self.sampling_table, open(dir_name + "/sampling_table.dat", "wb+"))
        pickle.dump(self.labels, open(dir_name + "/labels.dat", "wb+"))
        pickle.dump(self.word_target, open(dir_name + "/word_target.dat", "wb+"))
        pickle.dump(self.word_context, open(dir_name + "/word_context.dat", "wb+"))


    def build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.L - 1))
        self.logger.debug("original count " + str(len(count)))
        dictionary = dict()
        self.counter = dict()
        for word, _ in count:
            if self.emb_type:
                if word == 'UNK' or (word in self.word2vec_embedings.vocab\
                        and word in self.glove_embedings and word in self.fasttext_embedings):
                    dictionary[word] = len(dictionary)
                    self.counter[word] = _
                else:
                    self.logger.debug(word + " not in embeds")
            else:
                dictionary[word] = len(dictionary)
        del (self.word2vec_embedings)
        del (self.glove_embedings)
        del (self.fasttext_embedings)
        del (self.custom_embedings)
        self.L = len(dictionary)
        self.logger.debug("dictionary size" + str(len(dictionary)))
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count

        data = np.array(data)
        self.count = count
        self.dictionary = dictionary
        self.words = [reverse_dictionary[x] for x in range(len(reverse_dictionary))]
        self.logger.debug('....building samples')
        self.sampling_table = sequence.make_sampling_table(len(dictionary))
        couples, labels = skipgrams(data,
                                    len(dictionary),
                                    window_size=self.cs,
                                    sampling_table=self.sampling_table,
                                    negative_samples=self.ns)
        del data
        self.labels = np.array(labels)
        # labels[labels == 0] = -1
        word_target, word_context = zip(*couples)
        del couples
        self.word_target = np.array(word_target, dtype="int32")
        self.word_context = np.array(word_context, dtype="int32")
        self.logger.debug('....corpus generated')
        with open(self.dir_name+'/vocab.tsv', 'w') as txt:
            for word in self.words:
                txt.write(word + '\n')
        self.logger.debug('....vocab writen')

    def read_embeddings(self, emb_file):
        # load  embeddings
        embeddings_index = {}
        f = open(emb_file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    def read_word2vec_embeddings(self, emb_file):
        # load  embeddings
        return KeyedVectors.load_word2vec_format(emb_file, binary=True)

    def load_embeddings(self, emb_type, word2vec_file, glove_file, fasttext_file, custom_file, logger):
        self.logger = logger
        self.word2vec_embedings = None
        self.glove_embedings = None
        self.fasttext_embedings = None
        self.custom_embedings = None
        self.emb_type = emb_type
        self.embedding_matrix = None
        if emb_type:
            self.word2vec_embedings = self.read_word2vec_embeddings(word2vec_file)
            self.glove_embedings = self.read_embeddings(glove_file)
            self.fasttext_embedings = self.read_embeddings(fasttext_file)
            if custom_file:
                self.custom_embedings = self.read_embeddings(custom_file)
            if self.emb_type == 'word2vec':
                self.K = self.word2vec_embedings.vector_size
                # build encoder embedding matrix
                embedding_matrix = np.zeros((self.L, self.K), dtype=np.float32)
                not_found = 0
                for word, index in self.dictionary.items():
                    embedding_index = self.word2vec_embedings.vocab[word].index
                    embedding_vector = self.word2vec_embedings.vectors[embedding_index]
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[index] = embedding_vector
                    else:
                        not_found += 1
                        self.logger.debug('%s word out of the vocab.' % word)
                self.embedding_matrix = embedding_matrix
            else:
                if self.emb_type == 'glove':
                    embeddings = self.glove_embedings
                elif self.emb_type == 'fasttext':
                    embeddings = self.fasttext_embedings
                else:
                    embeddings = self.custom_embedings
                self.K = len(list(embeddings.values())[0])
                self.logger.debug("build encoder embedding matrix")
                embedding_matrix = np.zeros((self.L, self.K), dtype=np.float32)
                not_found = 0
                for word, index in self.dictionary.items():
                    embedding_vector = embeddings.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[index] = embedding_vector
                    else:
                        not_found += 1
                        self.logger.debug('%s word out of the vocab.' % word)
                del (embeddings)
                self.embedding_matrix = embedding_matrix
            del (self.word2vec_embedings)
            del (self.glove_embedings)
            del (self.fasttext_embedings)
            del (self.custom_embedings)

    def batch_generator(self, batch_size):
        data_target = self.word_target
        data_context = self.word_context
        data_labels = self.labels
        while True:
            if data_target.shape[0] < batch_size:
                data_target = np.hstack([data_target, self.word_target])
                data_context = np.hstack([data_context, self.word_context])
                data_labels = np.hstack([data_labels, self.labels])
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
        chars_target, chars_context, labels = self.batch.next()
        return {target_placeholder: chars_target,
                context_placeholder: chars_context,
                labels_placeholder: labels,
                ones_placeholder: np.ones((n_minibatch), dtype=np.int32),
                zeros_placeholder: np.zeros((n_minibatch), dtype=np.int32),
                learning_rate_placeholder: learning_rate
                }

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state