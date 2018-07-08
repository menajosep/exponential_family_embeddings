from utils import *
import collections
from gensim.models import KeyedVectors
from math import sqrt


class bayessian_bern_emb_data():
    def __init__(self, input_file, cs, ns, n_minibatch, L, K,
                 emb_type, word2vec_file, glove_file,
                 fasttext_file, custom_file, dir_name):
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
        sentences = read_data(input_file)
        if emb_type:
            self.word2vec_embedings = self.read_word2vec_embeddings(word2vec_file)
            self.glove_embedings = self.read_embeddings(glove_file)
            self.fasttext_embedings = self.read_embeddings(fasttext_file)
            if custom_file:
                self.custom_embedings = self.read_embeddings(custom_file)
        self.build_dataset(sentences)
        self.batch = self.batch_generator()
        self.N = len(self.word_target)


    def build_dataset(self, sentences):
        count = [['UNK', -1]]
        count.extend(collections.Counter(''.join(sentences).split()).most_common(self.L - 1))
        print("original count " + str(len(count)))
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
        print("dictionary size" + str(len(dictionary)))
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
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
                        print('%s word out of the vocab.' % word)
                self.embedding_matrix = embedding_matrix
            else:
                if self.emb_type == 'glove':
                    embeddings = self.glove_embedings
                elif self.emb_type == 'fasttext':
                    embeddings = self.fasttext_embedings
                else:
                    embeddings = self.custom_embedings
                self.K = len(list(embeddings.values())[0])
                print("build encoder embedding matrix")
                embedding_matrix = np.zeros((self.L, self.K), dtype=np.float32)
                not_found = 0
                for word, index in dictionary.items():
                    embedding_vector = embeddings.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[index] = embedding_vector
                    else:
                        not_found += 1
                        print('%s word out of the vocab.' % word)
                del(embeddings)
                self.embedding_matrix = embedding_matrix
            del(self.word2vec_embedings)
            del(self.glove_embedings)
            del(self.fasttext_embedings)
            del(self.custom_embedings)

        self.build_sampling_table(self.counter)
        self.dictionary = dictionary
        self.words = [reverse_dictionary[x] for x in range(len(reverse_dictionary))]
        samples = self.parallel_process_text(sentences)
        target_words, context_words, labels = zip(*samples)
        self.labels = np.array(labels)
        del samples
        self.word_target = np.array(target_words, dtype="int32")
        self.word_context = np.array(context_words, dtype="int32")
        with open(self.dir_name+'/vocab.tsv', 'w') as txt:
            for word in self.words:
                txt.write(word + '\n')

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
        process_text = process_sentences_constructor(self.ns, self.dictionary, self.cs, self.sampling_table)
        return flatten_list(apply_parallel(process_text, data))

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

    def feed(self, target_placeholder, context_placeholder, labels_placeholder,
             ones_placeholder, zeros_placeholder, shuffling = False):
        chars_target, chars_context, labels = self.batch.next()
        if shuffling:
            labels = np.random.permutation(labels)
        return {target_placeholder: chars_target,
                context_placeholder: chars_context,
                labels_placeholder: labels,
                ones_placeholder: np.ones((self.n_minibatch), dtype=np.int32),
                zeros_placeholder: np.zeros((self.n_minibatch), dtype=np.int32)
                }

    def build_sampling_table(self, count_words):
        sampling_factor = 1e-3
        sampling_table = dict()
        total_occurrences = sum(count_words.values())
        for word in count_words:
            if word != 'UNK':
                word_frequency = (1. * count_words[word]) / total_occurrences
                sampling_table[word] = max(0., ((word_frequency - sampling_factor) / word_frequency) - sqrt(
                    sampling_factor / word_frequency))
        self.sampling_table = sampling_table