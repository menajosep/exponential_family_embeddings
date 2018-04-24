import numpy as np
from utils import *
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams


class bern_emb_data():
    def __init__(self, cs, ns, n_minibatch, L):
        # assert cs % 2 == 0
        self.cs = cs
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        filename = '/Users/jose.mena/dev/personal/data/wiki/wiki106.txt.zip'
        chars = read_data_as_chars(filename)
        self.build_dataset(chars)
        self.batch = self.batch_generator()
        self.N = len(self.data)

    def build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.L - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        self.L = len(dictionary)
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
        self.labels = [reverse_dictionary[x] for x in range(len(reverse_dictionary))]
        unigram_dist = np.array([1.0 * i for ii, i in count])
        unigram_dist = (unigram_dist / unigram_dist.sum()) ** (3.0 / 4)
        self.unigram = unigram_dist / unigram_dist.sum()
        with open('fits/vocab.tsv', 'w') as txt:
            for word in self.labels:
                txt.write(word + '\n')

    def batch_generator(self):
        batch_size = self.n_minibatch + self.cs
        data = self.data
        while True:
            if data.shape[0] < batch_size:
                data = np.hstack([data, self.data])
                if data.shape[0] < batch_size:
                    continue
            words = data[:batch_size]
            data = data[batch_size:]
            yield words

    def feed(self, placeholder):
        return {placeholder: self.batch.next()}

class bayessian_bern_emb_data():
    def __init__(self, cs, ns, n_minibatch, L, in_file):
        # assert cs%2 == 0
        self.cs = cs
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        self.in_file = in_file
        chars = read_data_as_chars(self.in_file)
        self.build_dataset(chars)
        self.batch = self.batch_generator()
        self.N = len(self.data)

    def build_dataset(self, chars):
        test_split = 0.8
        count = [['UNK', -1]]
        count.extend(collections.Counter(chars).most_common(self.L - 1))
        dictionary = dict()
        for char, _ in count:
            dictionary[char] = len(dictionary)
        data = list()
        self.L = len(dictionary)
        unk_count = 0
        for char in chars:
            if char in dictionary:
                index = dictionary[char]
            else:
                index = 0
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.data = np.array(data)
        self.count = count
        self.dictionary = dictionary
        self.chars = [reverse_dictionary[x] for x in range(len(reverse_dictionary))]
        sampling_table = sequence.make_sampling_table(len(dictionary))
        couples, all_labels = skipgrams(data,
                                        len(dictionary),
                                        window_size=self.cs,
                                        sampling_table=sampling_table,
                                        negative_samples=self.ns)
        del data
        self.labels_train = np.array(all_labels[:int(len(all_labels) * test_split)])
        self.labels_test = np.array(all_labels[int(len(all_labels) * test_split):len(all_labels)])
        all_chars_target, all_chars_context = zip(*couples)
        del couples
        self.chars_target_train = np.array(all_chars_target[:int(len(all_chars_target) * test_split)], dtype="int32")
        self.chars_target_test = np.array(
            all_chars_target[int(len(all_chars_target) * test_split):len(all_chars_target)], dtype="int32")
        self.chars_context_train = np.array(all_chars_context[:int(len(all_chars_context) * test_split)], dtype="int32")
        self.chars_context_test = np.array(
            all_chars_context[int(len(all_chars_context) * test_split):len(all_chars_context)], dtype="int32")
        with open('fits/vocab.tsv', 'w') as txt:
            for char in self.chars:
                txt.write(char + '\n')

    def batch_generator(self):
        batch_size = self.n_minibatch
        data_target = self.chars_target_train
        data_context = self.chars_context_train
        data_labels = self.labels_train
        while True:
            if data_target.shape[0] < batch_size:
                data_target = np.hstack([data_target, self.chars_target_train])
                data_context = np.hstack([data_context, self.chars_context_train])
                data_labels = np.hstack([data_labels, self.labels_train])
                if data_target.shape[0] < batch_size:
                    continue
            chars_target = data_target[:batch_size]
            chars_context = data_context[:batch_size]
            labels = data_labels[:batch_size]
            data_target = data_target[batch_size:]
            data_context = data_context[batch_size:]
            data_labels = data_labels[batch_size:]
            yield chars_target, chars_context, labels

    def feed(self, target_placeholder, context_placeholder, labels_placeholder,
             ones_placeholder, zeros_placeholder):
        chars_target, chars_context, labels = self.batch.next()
        return {target_placeholder: chars_target,
                context_placeholder: chars_context,
                labels_placeholder: labels,
                ones_placeholder: np.ones((self.n_minibatch), dtype=np.int32),
                zeros_placeholder: np.zeros((self.n_minibatch), dtype=np.int32)
                }