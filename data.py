from utils import *
import collections
from gensim.models import KeyedVectors
from math import sqrt
from random import shuffle
import pickle


class bayessian_bern_emb_data():
    def __init__(self, input_file, cs, ns, n_minibatch, L, K,
                 emb_type, word2vec_file, glove_file,
                 fasttext_file, custom_file, dir_name, logger):
        assert cs % 2 == 0
        self.logger = logger
        self.logger.debug('initializing bayessian_bern_emb_data with file ' + input_file)
        self.logger.debug('neg sampling ' + str(ns))
        self.logger.debug('context size of ' + str(cs))
        self.logger.debug('working dir ' + dir_name)
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
        sentences = read_data(input_file)
        self.logger.debug('....loading embeddings file')
        if emb_type:
            self.word2vec_embedings = self.read_word2vec_embeddings(word2vec_file)
            self.glove_embedings = self.read_embeddings(glove_file)
            self.fasttext_embedings = self.read_embeddings(fasttext_file)
            if custom_file:
                self.custom_embedings = self.read_embeddings(custom_file)
        self.logger.debug('....building corpus')
        self.build_dataset(sentences)
        pickle.dump(self.dictionary, open(dir_name + "/dictionary.dat", "wb+"))
        pickle.dump(self.words, open(dir_name + "/words.dat", "wb+"))
        pickle.dump(self.counter, open(dir_name + "/counter.dat", "wb+"))


    def build_dataset(self, sentences):
        count = [['UNK', -1]]
        count.extend(collections.Counter(''.join(sentences).split()).most_common(self.L - 1))
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
                    print (word + " not in embeds")
            else:
                dictionary[word] = len(dictionary)
        self.L = len(dictionary)
        self.logger.debug("dictionary size" + str(len(dictionary)))
        self.reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        if self.emb_type:
            if self.emb_type == 'word2vec':
                self.K = self.word2vec_embedings.vector_size
                # build encoder embedding matrix
                embedding_matrix = np.zeros((self.L, self.K), dtype=np.float32)
                not_found = 0
                for word, index in dictionary.items():
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
                for word, index in dictionary.items():
                    embedding_vector = embeddings.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[index] = embedding_vector
                    else:
                        not_found += 1
                        self.logger.debug('%s word out of the vocab.' % word)
                del(embeddings)
                self.embedding_matrix = embedding_matrix
            del(self.word2vec_embedings)
            del(self.glove_embedings)
            del(self.fasttext_embedings)
            del(self.custom_embedings)

        self.logger.debug('....Build sampling table')
        #self.build_sampling_table(self.counter)
        self.dictionary = dictionary
        self.words = [self.reverse_dictionary[x] for x in range(len(self.reverse_dictionary))]
        self.logger.debug('....start parallel processing')
        samples = self.parallel_process_text(sentences)
        self.N = len(samples)
        self.positive_word_sampling_indexes = dict()
        self.negative_word_sampling_indexes = dict()
        for key in self.reverse_dictionary:
            self.positive_word_sampling_indexes[key] = []
            self.negative_word_sampling_indexes[key] = []
        for i in range(len(samples)):
            if samples[i][2] == 1:
                self.positive_word_sampling_indexes[samples[i][0]].append(samples[i][1])
            else:
                self.negative_word_sampling_indexes[samples[i][0]].append(samples[i][1])
        del(samples)

        #shuffle(samples)
        self.logger.debug('....finish parallel processing')
        self.logger.debug('....corpus generated')
        self.logger.debug('....store vocab')
        with open(self.dir_name+'/vocab.tsv', 'w') as txt:
            for word in self.words:
                txt.write(word + '\n')
        self.logger.debug('....vocab stored')

    def read_embeddings(self, emb_file):
        # load  embeddings
        embeddings_index = {}
        f = open(emb_file)
        embeddings_size = 0
        for line in f:
            try:
                values = line.split()
                if embeddings_size == 0:
                    embeddings_size = len(values)
                else:
                    if embeddings_size == len(values):
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float64')
                        embeddings_index[word] = coefs
            except ValueError as ve:
                print('error')

        f.close()
        return embeddings_index

    def read_word2vec_embeddings(self, emb_file):
        # load  embeddings
        return KeyedVectors.load_word2vec_format(emb_file, binary=True)

    def parallel_process_text(self, data: List[str]) -> List[List[str]]:
        """Apply cleaner -> tokenizer."""
        process_text = process_sentences_constructor(self.ns, self.dictionary, self.cs)
        return flatten_list(apply_parallel(process_text, data))

    def batch_generator(self, batch_size):
        epoch_samples = []
        for word in self.dictionary:
            if word != 'UNK':
                word_index = self.dictionary[word]
                pos_samples_indexes = []
                while len(pos_samples_indexes) < self.cs:
                    positive_word_sampling_indexes = self.positive_word_sampling_indexes[word_index]
                    pos_random_index = random.randint(0, len(positive_word_sampling_indexes)-1)
                    pos_samples_indexes.append(pos_random_index)
                    epoch_samples.append((word_index, positive_word_sampling_indexes[pos_random_index], 1))
                neg_samples_indexes = []
                while len(neg_samples_indexes) < self.ns:
                    negative_word_sampling_indexes = self.negative_word_sampling_indexes[self.dictionary[word]]
                    neg_random_index = random.randint(0, len(negative_word_sampling_indexes)-1)
                    if neg_random_index not in neg_samples_indexes:
                        neg_samples_indexes.append(neg_random_index)
                        epoch_samples.append((word_index, negative_word_sampling_indexes[neg_random_index], 0))
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

    def build_sampling_table(self, count_words):
        sampling_factor = 1e-5
        sampling_table = dict()
        total_occurrences = sum(count_words.values())
        for word in count_words:
            if word != 'UNK':
                word_frequency = (1. * count_words[word]) / total_occurrences
                sampling_table[word] = max(0., ((word_frequency - sampling_factor) / word_frequency) - sqrt(
                    sampling_factor / word_frequency))
        self.sampling_table = sampling_table

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state